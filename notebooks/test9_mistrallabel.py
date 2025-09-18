#!/usr/bin/env python3
"""
Synthetic Labeling Experiment
Tests the effect of F1 vs number of labeled examples using LLM labeling
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Add this at the top
import json
import gc
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)

# Import your modules
from config.settings import Settings
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.loader import load_mit_dataset
from evaluation.evaluator import enhanced_evaluate
from generation.mistral_labeler import LabelGenerator
from gliner import GLiNER

# Import the training module
from training.trainer import train_lora_model, intialize_model, load_evaluation_model


def main():
    """Main experiment function"""
    
    # ===============================================================================
    # 1. Setup and Configuration
    # ===============================================================================
    
    # Setup
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    # Load test data
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"📊 Loaded test data: {len(test_data)} examples, {len(entity_types)} entity types")
    
    # Load pre-saved low scoring examples
    logger.info("📂 Loading pre-saved low scoring examples...")
    with open('../results/low_score_1000_examples.json', 'r') as file:
        low_n = json.load(file)
    logger.info(f"📊 Loaded {len(low_n)} low scoring examples")
    
    # Initialize label generator
    generator = LabelGenerator()
    logger.info(f"🤖 Label Generator model: {generator.model_name}")
    
    # ===============================================================================
    # 2. Training Configuration
    # ===============================================================================
    
    training_config = {
        'num_steps': 1000,
        'train_batch_size': 8,
        'gradient_accumulation_steps': 1,
        'learning_rate': 0.00021008343694753508,
        'others_lr': 0.00021008343694753508,
        'warmup_ratio': 0.07064690788186724,
        'eval_steps': 100,
        'save_steps': 100,
        'logging_steps': 10,
        'max_grad_norm': 1,
        'weight_decay': 0.020216630535603918,
        'others_weight_decay': 0.020216630535603918,
        'focal_loss_alpha': 0.75,
        'focal_loss_gamma': 1.0,
        'patience': 3
    }
    
    logger.info("⚙️ Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"   • {key}: {value}")
    
    # ===============================================================================
    # 3. Experiment Parameters
    # ===============================================================================
    
    no_labels_to_test = [5,10,25,50,75,100,250,500,750,1000]
    
    # Results storage
    results = {
        'no_labels': [],
        'f1': [],
        'gliner_f1': [],
        'confidence': [],
        'avg_entities': []
    }
    
    # Calculate total iterations
    total_iterations = len(no_labels_to_test)
    logger.info(f"\n🔬 Experiment Overview:")
    logger.info(f"   • Label amounts to test: {no_labels_to_test}")
    logger.info(f"   • Total experiments: {total_iterations}")
    logger.info(f"   • Baseline F1: 46.95")
    
    # ===============================================================================
    # 4. Main Experiment Loop
    # ===============================================================================
    
    logger.info(f"\n🚀 Starting Labeling Experiment...")
    logger.info("-" * 60)
    
    # Cache for labeled data (reuse across experiments)
    label_cache = []
    
    for num_labels in tqdm(no_labels_to_test, desc="Label Amounts", position=0):
        logger.info(f"\n🎯 Generating {num_labels} labeled examples...")
        
        # Generate labeled data with caching
        labeled_data = generator.generate(
            low_n_examples=low_n,
            num_samples=num_labels,
            entity_types=entity_types,
            label_cache=label_cache
        )
        
        # Calculate average entities
        avg_entities = sum(len(ex['ner']) for ex in labeled_data) / len(labeled_data) if labeled_data else 0
        
        logger.info(f"📊 Generated: {len(labeled_data)} labeled examples, avg entities: {avg_entities:.1f}")
        
        # Define adapter save path
        adapter_path = f"../models/labeled_model_{num_labels}"
        
        # ===============================================================================
        # 5. Training Phase
        # ===============================================================================
        
        logger.info(f"\n🔥 Training Phase: {num_labels} labeled examples")
        
        # Initialize model with LoRA
        model = intialize_model()
        model.to(device)
        
        # Train the model
        train_lora_model(
            model=model,
            train_data=labeled_data,
            eval_data=test_data,
            training_config=training_config,
            adapter_save_path=adapter_path
        )
        
        # Cleanup training model
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ===============================================================================
        # 6. Evaluation Phase
        # ===============================================================================
        
        logger.info(f"\n📊 Evaluation Phase...")
        
        # Load model with trained adapter
        eval_model = load_evaluation_model(adapter_path, device)
        
        # Enhanced evaluation
        with torch.no_grad():
            test_results = enhanced_evaluate(
                eval_model, test_data, entity_types,
                threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
            )
            
            # GLiNER evaluation
            gliner_results, gliner_f1_score = eval_model.evaluate(
                test_data, flat_ner=True, threshold=0.5,
                batch_size=16, entity_types=entity_types
            )
        
        # Extract results
        f1_score = test_results["overall_metrics"]["overall_f1_pct"]
        con_score = test_results["overall_metrics"]["overall_confidence_pct"]
        
        logger.info(f"✅ Results: F1={f1_score:.1f}%, GLiNER_F1={gliner_f1_score:.2%}, Confidence={con_score:.1f}%")
        
        # Store results
        results['no_labels'].append(num_labels)
        results['f1'].append(f1_score)
        results['gliner_f1'].append(f"{gliner_f1_score:.2%}")
        results['confidence'].append(con_score)
        results['avg_entities'].append(avg_entities)
        
        # Cleanup evaluation model
        del eval_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # ===============================================================================
    # 7. Results Analysis and Visualization
    # ===============================================================================
    
    logger.info(f"\n📋 Creating Results DataFrame...")
    
    final_results_df = pd.DataFrame(results)
    
    # Configure pandas to display all columns and rows for logging
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    logger.info("\n" + "="*60)
    logger.info("FINAL LABELING EXPERIMENT RESULTS")
    logger.info("="*60)
    logger.info(final_results_df.to_string(index=False))
    
    # Reset pandas display options to default
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows') 
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # Save results
    results_path = f"../results/mistral/labeling_results1_{generator.model_name}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # ===============================================================================
    # 8. Line Plot Visualization
    # ===============================================================================
    
    logger.info(f"\n📈 Generating Line Plot...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create line plot
    plt.figure(figsize=(12, 8))
    
    # Plot F1 scores
    plt.plot(
        results['no_labels'], results['f1'],
        marker='o', markersize=8, linewidth=3, 
        label='F1 Score (%)', color='#2E86C1', alpha=0.8
    )
    
    # Add baseline line
    plt.axhline(y=46.95, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline F1 (46.95%)')
    
    # Customize plot
    plt.title(f'F1 Score vs Number of Labeled Examples\n({generator.model_name})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Labeled Examples', fontsize=14)
    plt.ylabel('F1 Score (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add value annotations on points
    for i, (x, y) in enumerate(zip(results['no_labels'], results['f1'])):
        plt.annotate(f'{y:.1f}%', (x, y), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=10, fontweight='bold')
    
    # Set x-axis ticks
    plt.xticks(results['no_labels'])
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_path = f"../results/mistral/{generator.model_name}_labeling_results1.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_path}")
    plt.show()
    
    logger.info(f"\n🎉 Labeling Experiment completed successfully!")
    logger.info(f"📋 Best F1 Score: {max(results['f1']):.1f}%")
    
    best_idx = results['f1'].index(max(results['f1']))
    logger.info(f"🏆 Best configuration: {results['no_labels'][best_idx]} labeled examples")
    
    # Show improvement over baseline
    best_f1 = max(results['f1'])
    improvement = best_f1 - 46.95
    logger.info(f"📈 Improvement over baseline: +{improvement:.1f}% F1 score")


if __name__ == "__main__":
    main()