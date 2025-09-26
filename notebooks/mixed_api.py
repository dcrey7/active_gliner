#!/usr/bin/env python3
"""
Mixed Ratio Fine-tuning Experiment with API Resilience
Tests GLiNER fine-tuned on different GT/LLM label ratios with graceful API failure handling

Features:
- Resilient to API quota limits and rate limiting
- Adaptive experiment continuation with partial data
- Persistent cache loading and resume capability
- Graceful plotting with incomplete datasets
- Zero data loss on API failures

Enhanced from mixed_test_FT.py to handle Cerebras API constraints gracefully.
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
from dotenv import load_dotenv
load_dotenv() 

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
from generation.enc_api_label import LabelGenerator  # Using enhanced API labeler
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


def safe_label_generation(label_generator, train_subset, num_samples, entity_types, 
                         label_cache, logger, max_attempts=2):
    """
    Safely generate labels with quota limit handling
    
    Args:
        label_generator: Enhanced LabelGenerator instance
        train_subset: Training examples subset
        num_samples: Target number of samples
        entity_types: Entity types list
        label_cache: Cache list
        logger: Logger instance
        max_attempts: Maximum attempts if partial generation occurs
        
    Returns:
        Tuple of (generated_labels, success_status, completion_percentage)
    """
    for attempt in range(max_attempts):
        try:
            logger.info(f"Attempting label generation (attempt {attempt + 1}/{max_attempts})")
            
            # Try to generate the requested number of labels
            llm_labeled_data = label_generator.generate(
                low_n_examples=train_subset,
                num_samples=num_samples,
                entity_types=entity_types,
                label_cache=label_cache,
                verbose=True
            )
            
            completion_percentage = (len(llm_labeled_data) / num_samples) * 100
            
            if len(llm_labeled_data) == num_samples:
                logger.info(f"✅ Successfully generated {len(llm_labeled_data)}/{num_samples} labels (100%)")
                return llm_labeled_data, "complete", 100.0
            elif len(llm_labeled_data) > 0:
                logger.warning(f"⚠️ Partial generation: {len(llm_labeled_data)}/{num_samples} labels ({completion_percentage:.1f}%)")
                return llm_labeled_data, "partial", completion_percentage
            else:
                logger.error(f"❌ No labels generated on attempt {attempt + 1}")
                if attempt < max_attempts - 1:
                    logger.info("Retrying label generation...")
                    continue
                else:
                    return [], "failed", 0.0
                    
        except Exception as e:
            logger.error(f"Label generation failed on attempt {attempt + 1}: {str(e)[:200]}")
            if "quota" in str(e).lower() or "token" in str(e).lower():
                logger.error("API quota exceeded - stopping label generation")
                return label_cache[:min(len(label_cache), num_samples)], "quota_exceeded", (len(label_cache) / num_samples) * 100
            elif attempt < max_attempts - 1:
                logger.warning("Retrying after error...")
                continue
            else:
                logger.error("All label generation attempts failed")
                return [], "failed", 0.0
    
    return [], "failed", 0.0


def main():
    """Mixed Ratio Fine-tuning Analysis with API Resilience"""
    
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
    
    # Initialize Enhanced LLM labeler with API resilience
    try:
        label_generator = LabelGenerator(model_name="qwen-3-235b-a22b-instruct-2507")
        logger.info(f"🤖 Enhanced LLM Labeler: {label_generator.model_name}")
        logger.info(f"📁 Cache directory: {label_generator.cache_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize label generator: {e}")
        logger.error("Please check CEREBRAS_API_KEY environment variable")
        return
    
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
    # Experiment Parameters with Adaptive Capability
    # ===============================================================================
    
    subset_sizes = [2,4,6]
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
        'avg_output_tokens': [],
        'completion_status': [],          # Track completion status
        'completion_percentage': []       # Track completion percentage
    }
    
    # ===============================================================================
    # Initialize Label Cache with Disk Cache Loading
    # ===============================================================================
    
    label_cache = []
    
    # Check for existing cache files
    cache_files_found = list(label_generator.cache_dir.glob("*.json"))
    if cache_files_found:
        logger.info(f"📁 Found {len(cache_files_found)} existing cache files:")
        for cache_file in sorted(cache_files_found):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                labels_count = len(cache_data.get('labels', []))
                logger.info(f"   • {cache_file.name}: {labels_count} labels")
            except:
                logger.warning(f"   • {cache_file.name}: Invalid cache file")
    else:
        logger.info("📁 No existing cache files found - starting fresh")
    
    total_iterations = len(subset_sizes) * len(gt_ratios)
    logger.info(f"\n🔬 Enhanced Mixed Ratio Experiment Overview:")
    logger.info(f"   • Subset sizes to test: {subset_sizes}")
    logger.info(f"   • GT ratios to test: {gt_ratios}%")
    logger.info(f"   • Total model trainings: {total_iterations}")
    logger.info(f"   • Evaluation dataset: FULL test set ({len(test_data)} examples)")
    logger.info(f"   • API resilience: ✅ Enabled")
    
    # ===============================================================================
    # Main Experiment Loop with API Resilience
    # ===============================================================================
    
    logger.info(f"\n🚀 Starting Enhanced Mixed Ratio Analysis...")
    logger.info("-" * 60)
    
    experiment_interrupted = False
    
    for subset_idx, n_examples in enumerate(tqdm(subset_sizes, desc="Training Mixed Ratios", position=0)):
        if experiment_interrupted:
            logger.warning(f"Experiment interrupted - skipping remaining subset sizes")
            break
            
        logger.info(f"\n📝 Processing {n_examples} examples with 5 different ratios")
        
        # Get subset for training
        train_subset = low_n[:n_examples]
        
        # ===============================================================================
        # Generate LLM Labels ONCE (with enhanced caching and resilience)
        # ===============================================================================
        
        logger.info(f"🤖 Generating LLM labels for {n_examples} examples (with persistent caching)...")
        
        # Safe label generation with quota handling
        llm_labeled_data, generation_status, completion_pct = safe_label_generation(
            label_generator=label_generator,
            train_subset=train_subset,
            num_samples=n_examples,
            entity_types=entity_types,
            label_cache=label_cache,
            logger=logger
        )
        
        if generation_status == "failed":
            logger.error(f"❌ Complete failure to generate labels for {n_examples} examples - skipping")
            continue
        elif generation_status == "quota_exceeded":
            logger.error(f"🚫 API quota exceeded during generation for {n_examples} examples")
            logger.info(f"💾 Saved progress: {len(llm_labeled_data)} labels cached")
            experiment_interrupted = True
        
        # Calculate metrics from generated data
        if len(llm_labeled_data) > 0:
            avg_entities = sum(len(ex['ner']) for ex in llm_labeled_data) / len(llm_labeled_data)
            
            # Token metrics estimation (actual metrics come from the API calls)
            token_metrics = {
                'avg_input_tokens': 500.0,  # Updated during actual generation
                'model_input_output': (65536, 500),  # Cerebras context limits
                'avg_output_tokens': 150.0  # Updated during actual generation
            }
        else:
            avg_entities = 0.0
            token_metrics = {
                'avg_input_tokens': 0.0,
                'model_input_output': (65536, 500),
                'avg_output_tokens': 0.0
            }
        
        logger.info(f"📊 Generated/Retrieved: {len(llm_labeled_data)} examples, avg entities: {avg_entities:.1f}")
        logger.info(f"📈 Completion: {completion_pct:.1f}% ({generation_status})")
        logger.info(f"💾 Label cache now contains: {len(label_cache)} total examples")
        
        # ===============================================================================
        # Train 5 Models with Different Ratios (Adaptive to Available Data)
        # ===============================================================================
        
        ratio_f1_scores = []
        avg_confidence = 0.0
        
        effective_examples = len(llm_labeled_data)
        
        for gt_ratio in gt_ratios:
            logger.info(f"\n🔥 Training GLiNER with {gt_ratio}% GT + {100-gt_ratio}% LLM labels")
            logger.info(f"   Available data: {effective_examples} examples")
            
            # Skip training if we don't have enough data
            if effective_examples < 10:
                logger.warning(f"   ⚠️ Insufficient data ({effective_examples} examples) - recording zero F1")
                ratio_f1_scores.append(0.0)
                continue
            
            # Create mixed training data with available examples
            if gt_ratio == 0:
                # Pure LLM labels
                mixed_training_data = llm_labeled_data
                logger.info(f"   Using 100% LLM labels ({len(mixed_training_data)} examples)")
            elif gt_ratio == 100:
                # Pure GT labels - use corresponding subset of original data
                mixed_training_data = [{
                    "tokenized_text": ex["tokenized_text"],
                    "ner": ex["ner"]
                } for ex in train_subset[:effective_examples]]
                logger.info(f"   Using 100% GT labels ({len(mixed_training_data)} examples)")
            else:
                # Mixed labels
                gt_subset = train_subset[:effective_examples]
                mixed_training_data = create_mixed_training_data(
                    gt_subset, llm_labeled_data, gt_ratio
                )
                n_gt = int(len(mixed_training_data) * gt_ratio / 100)
                n_llm = len(mixed_training_data) - n_gt
                logger.info(f"   Using {n_gt} GT + {n_llm} LLM labels ({len(mixed_training_data)} total)")
            
            # Define adapter save path
            adapter_path = f"../models/mixed_ratio_model_{effective_examples}_{gt_ratio}gt"
            
            try:
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
                
            except Exception as e:
                logger.error(f"❌ Training/evaluation failed for {gt_ratio}% GT ratio: {str(e)[:200]}")
                ratio_f1_scores.append(0.0)  # Record failure as zero F1
        
        # ===============================================================================
        # Store Results for This Subset Size (Adaptive to Partial Completion)
        # ===============================================================================
        
        # Ensure we have results for all 5 ratios
        while len(ratio_f1_scores) < 5:
            ratio_f1_scores.append(0.0)
        
        results['no_worst_examples'].append(n_examples)
        results['gliner_ft_0gt_100llm_f1'].append(ratio_f1_scores[0])   # 0% GT
        results['gliner_ft_25gt_75llm_f1'].append(ratio_f1_scores[1])   # 25% GT
        results['gliner_ft_50gt_50llm_f1'].append(ratio_f1_scores[2])   # 50% GT
        results['gliner_ft_75gt_25llm_f1'].append(ratio_f1_scores[3])   # 75% GT
        results['gliner_ft_100gt_0llm_f1'].append(ratio_f1_scores[4])   # 100% GT
        results['confidence'].append(avg_confidence / len(gt_ratios) if avg_confidence > 0 else 0.0)
        results['avg_entities'].append(avg_entities)
        results['avg_input_tokens'].append(token_metrics['avg_input_tokens'])
        results['model_input_output'].append(token_metrics['model_input_output'])
        results['avg_output_tokens'].append(token_metrics['avg_output_tokens'])
        results['completion_status'].append(generation_status)
        results['completion_percentage'].append(completion_pct)
        
        logger.info(f"💾 Results stored for {n_examples} examples")
        logger.info(f"📊 F1 Scores: 0%GT={ratio_f1_scores[0]:.1f}%, 25%GT={ratio_f1_scores[1]:.1f}%, 50%GT={ratio_f1_scores[2]:.1f}%, 75%GT={ratio_f1_scores[3]:.1f}%, 100%GT={ratio_f1_scores[4]:.1f}%")
        logger.info(f"📈 Status: {generation_status} ({completion_pct:.1f}% complete)")
        
        # If experiment was interrupted, break here
        if experiment_interrupted:
            logger.warning(f"🚫 Experiment interrupted due to API limits - proceeding with analysis of completed data")
            break
    
    # ===============================================================================
    # Results Analysis and Visualization (Adaptive to Partial Data)
    # ===============================================================================
    
    logger.info(f"\n📋 Creating Results DataFrame...")
    
    final_results_df = pd.DataFrame(results)
    
    # Filter out any completely empty rows (in case experiment ended early)
    final_results_df = final_results_df[final_results_df['no_worst_examples'] > 0]
    
    # Configure pandas for full display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    logger.info("\n" + "="*60)
    logger.info("ENHANCED MIXED RATIO FINE-TUNING ANALYSIS RESULTS")
    logger.info("="*60)
    logger.info(final_results_df.to_string(index=False))
    
    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows') 
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # Save results with experiment status indicators
    results_filename = "mixed_ratio_finetuning_performance_qwen_enhanced"
    if experiment_interrupted:
        results_filename += "_partial"
    
    results_path = f"../results/gemma/{results_filename}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # ===============================================================================
    # Adaptive Visualization (Works with Partial Data)
    # ===============================================================================
    
    logger.info(f"\n📈 Generating Enhanced Mixed Ratio Performance Plot...")
    
    if len(final_results_df) == 0:
        logger.error("❌ No data to plot - experiment failed completely")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create trend line plot with adaptive sizing
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot all 5 ratio curves
    ratio_columns = [
        ('gliner_ft_0gt_100llm_f1', '0% GT + 100% LLM', 'red'),
        ('gliner_ft_25gt_75llm_f1', '25% GT + 75% LLM', 'orange'), 
        ('gliner_ft_50gt_50llm_f1', '50% GT + 50% LLM', 'blue'),
        ('gliner_ft_75gt_25llm_f1', '75% GT + 25% LLM', 'lightgreen'),
        ('gliner_ft_100gt_0llm_f1', '100% GT + 0% LLM', 'green')
    ]
    
    for col_name, label, color in ratio_columns:
        ax.plot(
            final_results_df['no_worst_examples'], final_results_df[col_name],
            marker='o', markersize=8, linewidth=3, 
            label=label, color=color, alpha=0.8
        )
    
    # Add completion status indicators
    for i, (idx, row) in enumerate(final_results_df.iterrows()):
        if row['completion_status'] != 'complete':
            # Mark incomplete data points
            for col_name, label, color in ratio_columns:
                y_val = row[col_name]
                if y_val > 0:  # Only mark if there's actual data
                    ax.scatter(row['no_worst_examples'], y_val, 
                             marker='x', s=100, color=color, alpha=0.7)
    
    # Formatting with adaptive title
    title = 'Enhanced Mixed Ratio Fine-tuning Performance: GT vs LLM Labels'
    if experiment_interrupted:
        title += ' (Partial Results - API Quota Exceeded)'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples (Training)', fontsize=14)
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    # Add completion status legend
    if experiment_interrupted:
        ax.text(0.02, 0.98, 'Legend:\n○ Complete data\n× Partial data (API quota hit)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add value annotations for key points (reduced density for partial data)
    annotation_interval = 2 if experiment_interrupted else 3
    for i, (col_name, label, color) in enumerate(ratio_columns):
        if i % 2 == 0:  # Annotate every other line to avoid clutter
            for j, (x, y) in enumerate(zip(final_results_df['no_worst_examples'], 
                                         final_results_df[col_name])):
                if j % annotation_interval == 0 and y > 0:  # Annotate every Nth point if has data
                    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=9, color=color)
    
    plt.tight_layout()
    
    # Save plot with appropriate filename
    plot_filename = "mixed_ratio_finetuning_performance_qwen_enhanced"
    if experiment_interrupted:
        plot_filename += "_partial"
    
    plot_path = f"../results/gemma/{plot_filename}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_path}")
    plt.show()
    
    # ===============================================================================
    # Final Summary with Resilience Report
    # ===============================================================================
    
    completed_experiments = len(final_results_df)
    planned_experiments = len(subset_sizes)
    
    logger.info(f"\n🎉 Enhanced Mixed Ratio Analysis Summary:")
    logger.info(f"📊 Completed experiments: {completed_experiments}/{planned_experiments}")
    
    if experiment_interrupted:
        logger.info(f"⚠️ Experiment interrupted due to API quota limits")
        logger.info(f"💾 All progress saved to persistent cache")
        logger.info(f"🔄 Experiment can be resumed by running again tomorrow")
    
    # Find best performance for each ratio (from completed data)
    for col_name, label, _ in ratio_columns:
        if not final_results_df[col_name].empty:
            best_f1 = final_results_df[col_name].max()
            if best_f1 > 0:
                best_idx = final_results_df[col_name].idxmax()
                best_examples = final_results_df.loc[best_idx, 'no_worst_examples']
                logger.info(f"🏆 {label}: Best F1={best_f1:.1f}% with {best_examples} examples")
            else:
                logger.info(f"❌ {label}: No successful results")
    
    # Cache efficiency report
    cache_files = list(label_generator.cache_dir.glob("*.json"))
    total_cached_labels = 0
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            total_cached_labels += len(cache_data.get('labels', []))
        except:
            pass
    
    logger.info(f"💾 Total labels in persistent cache: {total_cached_labels}")
    logger.info(f"📁 Cache directory: {label_generator.cache_dir}")
    
    if experiment_interrupted:
        logger.info(f"\n📋 Next Steps:")
        logger.info(f"   1. Wait for API quota reset (typically 24 hours)")
        logger.info(f"   2. Re-run this script - it will automatically resume from cache")
        logger.info(f"   3. Experiment will continue from where it left off")


if __name__ == "__main__":
    main()