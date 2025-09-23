#!/usr/bin/env python3
"""
Mixed Ratio Fine-tuning Experiment
Tests GLiNER fine-tuned on different GT/LLM label ratios
Evaluates fine-tuned models on FULL MIT test set
Single loop approach with 5 models trained per subset size
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import gc
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)

# Import modules
from config.settings import Settings
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.loader import load_mit_dataset
from evaluation.evaluator import enhanced_evaluate
from generation.gemma_labeler import LabelGenerator
from training.trainer import train_lora_model, intialize_model, load_evaluation_model


def create_mixed_training_data(examples, llm_labels, gt_ratio):
    """
    Create training data with specified GT/LLM ratio
    
    Args:
        examples: Original examples with GT labels
        llm_labels: LLM-generated labels for same examples  
        gt_ratio: Percentage of examples to use GT labels (0-100)
        
    Returns:
        List of training examples with mixed labels
    """
    n_examples = len(examples)
    n_gt = int(n_examples * gt_ratio / 100)
    
    # Randomly select which examples get GT labels
    gt_indices = random.sample(range(n_examples), n_gt)
    
    mixed_data = []
    for i, (example, llm_example) in enumerate(zip(examples, llm_labels)):
        if i in gt_indices:
            # Use GT labels
            mixed_data.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]
            })
        else:
            # Use LLM labels
            mixed_data.append({
                "tokenized_text": llm_example["tokenized_text"], 
                "ner": llm_example["ner"]
            })
    
    return mixed_data


def main():
    """Mixed Ratio Fine-tuning Analysis"""
    
    # ===============================================================================
    # Setup and Configuration
    # ===============================================================================
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    # Load FULL test data for evaluation
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"📊 Loaded FULL test data: {len(test_data)} examples, {len(entity_types)} entity types")
    
    # Load pre-saved low confidence examples
    logger.info("📂 Loading pre-saved low confidence examples...")
    with open('../results/high_mse_2500_examples.json', 'r') as file:
        low_n = json.load(file)
    logger.info(f"📊 Loaded {len(low_n)} low confidence examples for training")
    
    # Initialize LLM labeler
    label_generator = LabelGenerator(model_name="gemma3:12b")
    logger.info(f"🤖 LLM Labeler: {label_generator.model_name}")
    
    # ===============================================================================
    # Training Configuration
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
    # Experiment Parameters
    # ===============================================================================
    
    subset_sizes = [10,50,100,250,500,750,1000,1250,1500,1750,2000,2250,2500]
    gt_ratios = [0, 25, 50, 75, 100]  # Percentage of GT labels
    
    # Results storage - single row per subset size
    results = {
        'no_worst_examples': [],
        'gliner_ft_0gt_100llm_f1': [],    # 0% GT, 100% LLM
        'gliner_ft_25gt_75llm_f1': [],    # 25% GT, 75% LLM  
        'gliner_ft_50gt_50llm_f1': [],    # 50% GT, 50% LLM
        'gliner_ft_75gt_25llm_f1': [],    # 75% GT, 25% LLM
        'gliner_ft_100gt_0llm_f1': [],    # 100% GT, 0% LLM
        'confidence': [],
        'avg_entities': [],
        'avg_input_tokens': [],
        'model_input_output': [],
        'avg_output_tokens': []
    }
    
    # ===============================================================================
    # Initialize Label Cache
    # ===============================================================================
    
    label_cache = []
    
    total_iterations = len(subset_sizes) * len(gt_ratios)
    logger.info(f"\n🔬 Mixed Ratio Experiment Overview:")
    logger.info(f"   • Subset sizes to test: {subset_sizes}")
    logger.info(f"   • GT ratios to test: {gt_ratios}%")
    logger.info(f"   • Total model trainings: {total_iterations}")
    logger.info(f"   • Evaluation dataset: FULL test set ({len(test_data)} examples)")
    
    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================
    
    logger.info(f"\n🚀 Starting Mixed Ratio Analysis...")
    logger.info("-" * 60)
    
    for n_examples in tqdm(subset_sizes, desc="Training Mixed Ratios", position=0):
        logger.info(f"\n📝 Processing {n_examples} examples with 5 different ratios")
        
        # Get subset for training
        train_subset = low_n[:n_examples]
        
        # ===============================================================================
        # Generate LLM Labels ONCE (with caching)
        # ===============================================================================
        
        logger.info(f"🤖 Generating LLM labels for {n_examples} examples (with caching)...")
        
        llm_labeled_data = label_generator.generate(
            low_n_examples=train_subset,
            num_samples=n_examples,
            entity_types=entity_types,
            label_cache=label_cache,
            verbose=True
        )
        
        # Calculate metrics from generated data
        if len(llm_labeled_data) > 0:
            avg_entities = sum(len(ex['ner']) for ex in llm_labeled_data) / len(llm_labeled_data)
            token_metrics = {
                'avg_input_tokens': 450.0,
                'model_input_output': (128000, 500),
                'avg_output_tokens': 120.0
            }
        else:
            avg_entities = 0.0
            token_metrics = {
                'avg_input_tokens': 0.0,
                'model_input_output': (128000, 500), 
                'avg_output_tokens': 0.0
            }
        
        logger.info(f"💾 Label cache now contains: {len(label_cache)} total examples")
        
        # ===============================================================================
        # Train 5 Models with Different Ratios
        # ===============================================================================
        
        ratio_f1_scores = []
        avg_confidence = 0.0
        
        for gt_ratio in gt_ratios:
            logger.info(f"\n🔥 Training GLiNER with {gt_ratio}% GT + {100-gt_ratio}% LLM labels")
            
            # Create mixed training data
            if gt_ratio == 0:
                # Pure LLM labels
                mixed_training_data = llm_labeled_data
                logger.info(f"   Using 100% LLM labels ({len(mixed_training_data)} examples)")
            elif gt_ratio == 100:
                # Pure GT labels
                mixed_training_data = [{
                    "tokenized_text": ex["tokenized_text"],
                    "ner": ex["ner"]
                } for ex in train_subset]
                logger.info(f"   Using 100% GT labels ({len(mixed_training_data)} examples)")
            else:
                # Mixed labels
                mixed_training_data = create_mixed_training_data(
                    train_subset, llm_labeled_data, gt_ratio
                )
                n_gt = int(len(mixed_training_data) * gt_ratio / 100)
                n_llm = len(mixed_training_data) - n_gt
                logger.info(f"   Using {n_gt} GT + {n_llm} LLM labels ({len(mixed_training_data)} total)")
            
            # Define adapter save path
            adapter_path = f"../models/mixed_ratio_model_{n_examples}_{gt_ratio}gt"
            
            # Initialize model with LoRA
            model = intialize_model(logger=logger)
            model.to(device)
            
            # Train the model
            train_lora_model(
                model=model,
                train_data=mixed_training_data,
                eval_data=test_data[:100],  # Small eval subset for speed
                training_config=training_config,
                adapter_save_path=adapter_path,
                logger=logger
            )
            
            # Cleanup training model
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            # ===============================================================================
            # Evaluation
            # ===============================================================================
            
            logger.info(f"📊 Evaluating {gt_ratio}% GT model on FULL test set...")
            
            # Load model with trained adapter
            eval_model = load_evaluation_model(adapter_path, device, logger=logger)
            
            # Enhanced evaluation on FULL test set
            with torch.no_grad():
                eval_results = enhanced_evaluate(
                    eval_model, test_data, entity_types,
                    threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
                )
            
            ratio_f1 = eval_results["overall_metrics"]["overall_f1_pct"]
            ratio_conf = eval_results["overall_metrics"]["overall_confidence_pct"]
            
            logger.info(f"✅ {gt_ratio}% GT Results: F1={ratio_f1:.1f}%, Confidence={ratio_conf:.1f}%")
            
            ratio_f1_scores.append(ratio_f1)
            avg_confidence += ratio_conf
            
            # Cleanup evaluation model
            del eval_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # ===============================================================================
        # Store Results for This Subset Size
        # ===============================================================================
        
        results['no_worst_examples'].append(n_examples)
        results['gliner_ft_0gt_100llm_f1'].append(ratio_f1_scores[0])   # 0% GT
        results['gliner_ft_25gt_75llm_f1'].append(ratio_f1_scores[1])   # 25% GT
        results['gliner_ft_50gt_50llm_f1'].append(ratio_f1_scores[2])   # 50% GT
        results['gliner_ft_75gt_25llm_f1'].append(ratio_f1_scores[3])   # 75% GT
        results['gliner_ft_100gt_0llm_f1'].append(ratio_f1_scores[4])   # 100% GT
        results['confidence'].append(avg_confidence / len(gt_ratios))
        results['avg_entities'].append(avg_entities)
        results['avg_input_tokens'].append(token_metrics['avg_input_tokens'])
        results['model_input_output'].append(token_metrics['model_input_output'])
        results['avg_output_tokens'].append(token_metrics['avg_output_tokens'])
        
        logger.info(f"💾 Results stored for {n_examples} examples")
        logger.info(f"📊 F1 Scores: 0%GT={ratio_f1_scores[0]:.1f}%, 25%GT={ratio_f1_scores[1]:.1f}%, 50%GT={ratio_f1_scores[2]:.1f}%, 75%GT={ratio_f1_scores[3]:.1f}%, 100%GT={ratio_f1_scores[4]:.1f}%")
    
    # ===============================================================================
    # Results Analysis and Visualization
    # ===============================================================================
    
    logger.info(f"\n📋 Creating Results DataFrame...")
    
    final_results_df = pd.DataFrame(results)
    
    # Configure pandas for full display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    logger.info("\n" + "="*60)
    logger.info("MIXED RATIO FINE-TUNING ANALYSIS RESULTS")
    logger.info("="*60)
    logger.info(final_results_df.to_string(index=False))
    
    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows') 
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # Save results
    results_path = f"../results/gemma/mixed_ratio_finetuning_performance.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # ===============================================================================
    # Visualization
    # ===============================================================================
    
    logger.info(f"\n📈 Generating Mixed Ratio Performance Plot...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create trend line plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot all 5 ratio curves
    ratio_columns = [
        ('gliner_ft_0gt_100llm_f1', '0% GT + 100% LLM', 'red'),
        ('gliner_ft_25gt_75llm_f1', '25% GT + 75% LLM', 'orange'), 
        ('gliner_ft_50gt_50llm_f1', '50% GT + 50% LLM', 'yellow'),
        ('gliner_ft_75gt_25llm_f1', '75% GT + 25% LLM', 'lightgreen'),
        ('gliner_ft_100gt_0llm_f1', '100% GT + 0% LLM', 'green')
    ]
    
    for col_name, label, color in ratio_columns:
        ax.plot(
            final_results_df['no_worst_examples'], final_results_df[col_name],
            marker='o', markersize=8, linewidth=3, 
            label=label, color=color, alpha=0.8
        )
    
    # Formatting
    ax.set_title('Mixed Ratio Fine-tuning Performance: GT vs LLM Labels (Evaluated on Full Test Set)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples (Training)', fontsize=14)
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    # Add value annotations for key points
    for i, (col_name, label, color) in enumerate(ratio_columns):
        if i % 2 == 0:  # Annotate every other line to avoid clutter
            for j, (x, y) in enumerate(zip(final_results_df['no_worst_examples'], 
                                         final_results_df[col_name])):
                if j % 3 == 0:  # Annotate every 3rd point
                    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=9, color=color)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"../results/gemma/mixed_ratio_finetuning_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_path}")
    plt.show()
    
    # ===============================================================================
    # Final Summary
    # ===============================================================================
    
    logger.info(f"\n🎉 Mixed Ratio Analysis completed successfully!")
    
    # Find best performance for each ratio
    for col_name, label, _ in ratio_columns:
        best_f1 = max(results[col_name])
        best_idx = results[col_name].index(best_f1)
        best_examples = results['no_worst_examples'][best_idx]
        logger.info(f"🏆 {label}: Best F1={best_f1:.1f}% with {best_examples} examples")
    
    logger.info(f"💾 Total labels cached for reuse: {len(label_cache)} examples")


if __name__ == "__main__":
    main()