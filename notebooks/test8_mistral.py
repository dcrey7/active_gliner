#!/usr/bin/env python3
"""
Synthetic Data Generation Experiment
Tests the effect of F1 vs number of synthetic data vs number of corrected examples
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
from generation.mistral_simple_gen import SyntheticDataGenerator
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
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    logger.info(f"🤖 Generator model: {generator.model_name}")
    
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
    }
    
    logger.info("⚙️ Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"   • {key}: {value}")
    
    # ===============================================================================
    # 3. Experiment Parameters
    # ===============================================================================
    
    no_syn_train_data = [10, 50, 100]
    no_low_train_data = [10, 50, 100]
    
    # Results storage
    results = {
        'no_corrected_train_data': [],
        'no_syn_train_data': [],
        'f1': [],
        'gliner_f1': [],
        'confidence': [],
        'avg_entities': [],
        'avg_input_tokens': [],
        'model_input_output': [],
        'avg_output_tokens': []
    }
    
    # Calculate total iterations
    total_iterations = len(no_low_train_data) * len(no_syn_train_data)
    logger.info(f"\n🔬 Experiment Overview:")
    logger.info(f"   • Corrected examples to test: {no_low_train_data}")
    logger.info(f"   • Synthetic amounts to test: {no_syn_train_data}")
    logger.info(f"   • Total combinations: {total_iterations}")
    logger.info(f"   • Baseline F1: 46.95")
    
    # ===============================================================================
    # 4. Main Experiment Loop
    # ===============================================================================
    
    logger.info(f"\n🚀 Starting Experiment...")
    logger.info("-" * 60)
    
    for i in tqdm(no_low_train_data, desc="Corrected Examples", position=0):
        logger.info(f"\n📝 Using {i} corrected examples in synthetic prompts")
        
        # Cache for synthetic data (reuse across j values)
        synthetic_data_cache = []
        
        for j in tqdm(no_syn_train_data, desc=f"Synthetic (corr={i})", position=1, leave=False):
            logger.info(f"\n🎯 Generating {j} synthetic examples...")
            
            # Generate synthetic data with caching and token tracking
            synthetic_data, avg_entities, token_metrics = generator.generate(
                corrected_examples=low_n[:i],
                num_samples=j,
                entity_types=entity_types,
                countries=["USA", "Canada", "UK", "India", "France"],
                genres=["action", "comedy", "serious", "adventure", "sports"],
                subject="movie reviews",
                syn_cache=synthetic_data_cache
            )
            
            logger.info(f"📊 Generated: {len(synthetic_data)} examples, avg entities: {avg_entities:.1f}")
            logger.info(f"📊 Token metrics: input={token_metrics['avg_input_tokens']:.0f}, output={token_metrics['avg_output_tokens']:.0f}")
            
            # Define adapter save path
            adapter_path = f"../models/corr_syn_model_{i}/corr_syn_model_{j}"
            
            # ===============================================================================
            # 5. Training Phase
            # ===============================================================================
            
            logger.info(f"\n🔥 Training Phase: {i} corrected → {j} synthetic")
            
            # Initialize model with LoRA
            model = intialize_model()
            model.to(device)
            
            # Train the model
            train_lora_model(
                model=model,
                train_data=synthetic_data,
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
            results['no_corrected_train_data'].append(i)
            results['no_syn_train_data'].append(len(synthetic_data))
            results['f1'].append(f1_score)
            results['gliner_f1'].append(f"{gliner_f1_score:.2%}")
            results['confidence'].append(con_score)
            results['avg_entities'].append(avg_entities)
            results['avg_input_tokens'].append(token_metrics['avg_input_tokens'])
            results['model_input_output'].append(token_metrics['model_input_output'])
            results['avg_output_tokens'].append(token_metrics['avg_output_tokens'])
            
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
    logger.info("FINAL EXPERIMENTAL RESULTS")
    logger.info("="*60)
    logger.info(final_results_df.to_string(index=False))
    
    # Log the complete dataframe
    logger.info(f"\nFINAL EXPERIMENTAL RESULTS:")
    logger.info(f"\n{final_results_df.to_string(index=False)}")
    
    # Reset pandas display options to default
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows') 
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # Save results
    results_path = f"../results/mistral/synthetic_experiment_results_{generator.model_name}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Results saved to: {results_path}")
    logger.info(f"Results saved to: {results_path}")
    
    # ===============================================================================
    # 8. Visualization
    # ===============================================================================
    
    logger.info(f"\n📈 Generating Visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Heatmap
    pivot_f1 = final_results_df.pivot(
        index='no_corrected_train_data',
        columns='no_syn_train_data', 
        values='f1'
    )
    
    sns.heatmap(
        pivot_f1, annot=True, fmt='.1f', cmap='RdYlGn',
        center=55, ax=ax1, cbar_kws={'label': 'F1 Score'}
    )
    ax1.invert_yaxis()
    ax1.set_title(f'F1 Heatmap: Corrected vs Synthetic ({generator.model_name})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Synthetic Examples', fontsize=12)
    ax1.set_ylabel('Number of Corrected Examples in Prompt', fontsize=12)
    
    # Plot 2: Trend lines
    synthetic_amounts = sorted(final_results_df['no_syn_train_data'].unique())
    
    for syn_val in synthetic_amounts:
        subset = final_results_df[final_results_df['no_syn_train_data'] == syn_val]
        subset_sorted = subset.sort_values('no_corrected_train_data')
        
        ax2.plot(
            subset_sorted['no_corrected_train_data'], subset_sorted['f1'],
            marker='o', markersize=6, linewidth=2, 
            label=f'{syn_val} synthetic', alpha=0.8
        )
    
    ax2.axhline(y=46.95, color='red', linestyle='--', alpha=0.7, label='Baseline F1')
    ax2.set_title(f'F1 vs Corrected Examples ({generator.model_name})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Corrected Examples in Prompt', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"../results/mistral/{generator.model_name}_experiment_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 Plots saved to: {plot_path}")
    plt.show()
    
    logger.info(f"\n🎉 Experiment completed successfully!")
    logger.info(f"📋 Best F1 Score: {max(results['f1']):.1f}%")
    
    best_idx = results['f1'].index(max(results['f1']))
    logger.info(f"🏆 Best configuration: {results['no_corrected_train_data'][best_idx]} corrected, {results['no_syn_train_data'][best_idx]} synthetic")


if __name__ == "__main__":
    main()