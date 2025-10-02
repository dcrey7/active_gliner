#!/usr/bin/env python3
"""
MSE Bucket Fine-tuning Experiment
Fine-tune GLiNER on GT and LLM labels across MSE confidence buckets
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
from pathlib import Path
from tqdm import tqdm

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


def main():
    """MSE Bucket Fine-tuning Experiment"""
    
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
    
    # Initialize LLM labeler
    label_generator = LabelGenerator(model_name="gemma3:12b")
    logger.info(f"🤖 LLM Labeler: {label_generator.model_name}")
    
    # ===============================================================================
    # Training Configuration (Same as mixed_test_FT.py)
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
    # Load Bucket Files
    # ===============================================================================
    BUCKET_EXAMPLES=500
    results_dir = Path("../results/data")
    bucket_files = sorted(results_dir.glob(f"bucket_*_mse_{BUCKET_EXAMPLES}.json"))
    
    if not bucket_files:
        raise FileNotFoundError(f"No bucket files found in {results_dir}")
    
    logger.info(f"\n📦 Found {len(bucket_files)} bucket files")
    for bf in bucket_files:
        logger.info(f"   • {bf.name}")
    
    # Load bucket summary for metadata
    summary_file = results_dir / "bucket_summary.json"
    with open(summary_file, 'r') as f:
        bucket_summary = json.load(f)
    
    # ===============================================================================
    # Initialize Results Storage
    # ===============================================================================
    
    results = {
        'bucket_number': [],
        'bucket_name': [],
        'mse_range': [],
        'avg_mse': [],
        'avg_confidence_pct': [],
        'num_examples': [],
        'gt_f1': [],
        'gt_confidence': [],
        'llm_f1': [],
        'llm_confidence': []
    }
    
    # Initialize Label Cache (Persistent across all buckets)
    
    
    logger.info(f"\n🔬 Experiment Overview:")
    logger.info(f"   • Number of buckets: {len(bucket_files)}")
    logger.info(f"   • Total experiments: {len(bucket_files) * 2} (GT + LLM per bucket)")
    logger.info(f"   • Evaluation dataset: FULL test set ({len(test_data)} examples)")
    # logger.info(f"   • Label cache initialized: {len(label_cache)} examples")
    
    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================
    
    logger.info(f"\n🚀 Starting MSE Bucket Experiment...")
    logger.info("="*80)
    
    for bucket_idx, bucket_file in enumerate(bucket_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"BUCKET {bucket_idx}/{len(bucket_files)}: {bucket_file.name}")
        logger.info(f"{'='*80}")
        
        label_cache = []
        # Load bucket data
        with open(bucket_file, 'r') as f:
            bucket_examples = json.load(f)
        
        logger.info(f"📊 Loaded {len(bucket_examples)} examples from bucket")
        
        # Get bucket metadata from summary
        bucket_meta = bucket_summary['bucket_summaries'][bucket_idx - 1]
        
        # ===============================================================================
        # Phase 1: Fine-tune on Ground Truth Labels
        # ===============================================================================
        
        logger.info(f"\n🔥 Phase 1: Training on GROUND TRUTH labels")
        
        # Prepare GT training data
        gt_training_data = []
        for example in bucket_examples:
            gt_training_data.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]
            })
        
        logger.info(f"   Prepared {len(gt_training_data)} GT examples")
        
        # Define adapter save path
        gt_adapter_path = f"../models/mse_bucket_{bucket_idx}_gt_model"
        
        # Initialize model with LoRA
        model = intialize_model(logger=logger)
        model.to(device)
        
        # Train on GT labels
        train_lora_model(
            model=model,
            train_data=gt_training_data,
            eval_data=test_data,
            training_config=training_config,
            adapter_save_path=gt_adapter_path,
            logger=logger
        )
        
        # Cleanup training model
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate GT model on FULL test set
        logger.info(f"📊 Evaluating GT model on FULL test set...")
        
        eval_model = load_evaluation_model(gt_adapter_path, device, logger=logger)
        
        with torch.no_grad():
            gt_results = enhanced_evaluate(
                eval_model, test_data, entity_types,
                threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
            )
        
        gt_f1 = gt_results["overall_metrics"]["overall_f1_pct"]
        gt_conf = gt_results["overall_metrics"]["overall_confidence_pct"]
        
        logger.info(f"✅ GT Model Results: F1={gt_f1:.1f}%, Confidence={gt_conf:.1f}%")
        
        # Cleanup evaluation model
        del eval_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ===============================================================================
        # Phase 2: Generate LLM Labels
        # ===============================================================================
        
        logger.info(f"\n🤖 Phase 2: Generating LLM labels (with caching)")
        
        # Prepare examples for labeling (need tokenized_text)
        examples_for_labeling = []
        for example in bucket_examples:
            examples_for_labeling.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]  # Keep GT for reference
            })
        
        # Generate LLM labels using gemma_labeler (with caching)
        llm_labeled_data = label_generator.generate(
            low_n_examples=examples_for_labeling,
            num_samples=len(examples_for_labeling),
            entity_types=entity_types,
            label_cache=label_cache,
            verbose=True
        )
        
        logger.info(f"💾 Label cache now contains: {len(label_cache)} total examples")
        
        # ===============================================================================
        # Phase 3: Fine-tune on LLM Labels
        # ===============================================================================
        
        logger.info(f"\n🔥 Phase 3: Training on LLM labels")
        logger.info(f"   Using {len(llm_labeled_data)} LLM labeled examples")
        
        # Define adapter save path
        llm_adapter_path = f"../models/mse_bucket_{bucket_idx}_llm_model"
        
        # Initialize model with LoRA
        model = intialize_model(logger=logger)
        model.to(device)
        
        # Train on LLM labels
        train_lora_model(
            model=model,
            train_data=llm_labeled_data,
            eval_data=test_data[:100],
            training_config=training_config,
            adapter_save_path=llm_adapter_path,
            logger=logger
        )
        
        # Cleanup training model
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate LLM model on FULL test set
        logger.info(f"📊 Evaluating LLM model on FULL test set...")
        
        eval_model = load_evaluation_model(llm_adapter_path, device, logger=logger)
        
        with torch.no_grad():
            llm_results = enhanced_evaluate(
                eval_model, test_data, entity_types,
                threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
            )
        
        llm_f1 = llm_results["overall_metrics"]["overall_f1_pct"]
        llm_conf = llm_results["overall_metrics"]["overall_confidence_pct"]
        
        logger.info(f"✅ LLM Model Results: F1={llm_f1:.1f}%, Confidence={llm_conf:.1f}%")
        
        # Cleanup evaluation model
        del eval_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ===============================================================================
        # Store Results
        # ===============================================================================
        
        results['bucket_number'].append(bucket_idx)
        results['bucket_name'].append(f"Bucket {bucket_idx}")
        results['mse_range'].append(f"{bucket_meta['mse_range'][0]:.3f}-{bucket_meta['mse_range'][1]:.3f}")
        results['avg_mse'].append(bucket_meta['avg_mse'])
        results['avg_confidence_pct'].append(bucket_meta['avg_confidence_pct'])
        results['num_examples'].append(len(bucket_examples))
        results['gt_f1'].append(gt_f1)
        results['gt_confidence'].append(gt_conf)
        results['llm_f1'].append(llm_f1)
        results['llm_confidence'].append(llm_conf)
        
        logger.info(f"\n💾 Results Summary for Bucket {bucket_idx}:")
        logger.info(f"   MSE Range: {results['mse_range'][-1]}")
        logger.info(f"   Avg Confidence: {results['avg_confidence_pct'][-1]:.1f}%")
        logger.info(f"   GT F1: {gt_f1:.1f}%")
        logger.info(f"   LLM F1: {llm_f1:.1f}%")
        logger.info(f"   Gap: {abs(gt_f1 - llm_f1):.1f}%")
    
    # ===============================================================================
    # Create Results DataFrame
    # ===============================================================================
    
    logger.info(f"\n📋 Creating Results DataFrame...")
    
    final_results_df = pd.DataFrame(results)
    
    # Configure pandas to show ALL rows and columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    logger.info("\n" + "="*80)
    logger.info("MSE BUCKET FINE-TUNING EXPERIMENT RESULTS")
    logger.info("="*80)
    logger.info("\n" + final_results_df.to_string(index=False))
    
    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # Save results
    results_path = f"../results/mse_bucket_experiment_results_{BUCKET_EXAMPLES}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # ===============================================================================
    # Generate Visualization
    # ===============================================================================
    
    logger.info(f"\n📈 Generating Bar Chart Visualization...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = range(len(final_results_df))
    width = 0.35
    
    # Blue bars for GT
    bars1 = ax.bar(
        [i - width/2 for i in x], 
        final_results_df['gt_f1'],
        width,
        label='Ground Truth Fine-tuned',
        color='blue',
        alpha=0.8
    )
    
    # Orange bars for LLM
    bars2 = ax.bar(
        [i + width/2 for i in x],
        final_results_df['llm_f1'],
        width,
        label='LLM Label Fine-tuned',
        color='orange',
        alpha=0.8
    )
    
    # Formatting
    ax.set_xlabel('MSE Confidence Buckets (Highest → Lowest Uncertainty)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14, fontweight='bold')
    ax.set_title('Fine-tuning Performance Across MSE Confidence Buckets', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # X-axis labels with confidence percentages
    x_labels = [
        f"{row['bucket_name']}\n{row['avg_confidence_pct']:.1f}% Conf\nMSE: {row['avg_mse']:.3f}"
        for _, row in final_results_df.iterrows()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"../results/mse_bucket_experiment_plot_{BUCKET_EXAMPLES}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_path}")
    plt.show()
    
    # ===============================================================================
    # Final Summary
    # ===============================================================================
    
    logger.info(f"\n🎉 MSE Bucket Experiment completed successfully!")
    logger.info(f"📊 Total buckets analyzed: {len(final_results_df)}")
    logger.info(f"🏆 Best GT F1: {max(results['gt_f1']):.1f}% (Bucket {results['bucket_number'][results['gt_f1'].index(max(results['gt_f1']))]})")
    logger.info(f"🏆 Best LLM F1: {max(results['llm_f1']):.1f}% (Bucket {results['bucket_number'][results['llm_f1'].index(max(results['llm_f1']))]})")
    
    # Calculate average gap
    avg_gap = sum(abs(gt - llm) for gt, llm in zip(results['gt_f1'], results['llm_f1'])) / len(results['gt_f1'])
    logger.info(f"📈 Average GT-LLM gap: {avg_gap:.1f}%")
    
    logger.info(f"💾 Total labels cached: {len(label_cache)} examples")
    logger.info(f"📁 All results saved to: ../results/")


if __name__ == "__main__":
    main()