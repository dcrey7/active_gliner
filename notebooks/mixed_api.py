#!/usr/bin/env python3
"""
Mixed Ratio Fine-tuning Experiment with Enhanced API Quota Handling
Tests GLiNER fine-tuned on different GT/LLM label ratios with graceful failure handling
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

warnings.filterwarnings('ignore')

src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)

from config.settings import Settings
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.loader import load_mit_dataset
from evaluation.evaluator import enhanced_evaluate
from generation.enc_api_label import LabelGenerator, QuotaExceededException
from training.trainer import train_lora_model, intialize_model, load_evaluation_model


def create_mixed_training_data(examples, llm_labels, gt_ratio):
    """Create training data with specified GT/LLM ratio"""
    n_examples = len(examples)
    n_gt = int(n_examples * gt_ratio / 100)
    
    gt_indices = random.sample(range(n_examples), n_gt)
    
    mixed_data = []
    for i, (example, llm_example) in enumerate(zip(examples, llm_labels)):
        if i in gt_indices:
            mixed_data.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]
            })
        else:
            mixed_data.append({
                "tokenized_text": llm_example["tokenized_text"], 
                "ner": llm_example["ner"]
            })
    
    return mixed_data


def main():
    """Enhanced Mixed Ratio Fine-tuning Analysis with Quota Handling"""
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"📊 Loaded FULL test data: {len(test_data)} examples, {len(entity_types)} entity types")
    
    logger.info("📂 Loading pre-saved low confidence examples...")
    with open('../results/high_mse_2500_examples.json', 'r') as file:
        low_n = json.load(file)
    logger.info(f"📊 Loaded {len(low_n)} low confidence examples for training")
    
    try:
        label_generator = LabelGenerator(model_name="qwen-3-235b-a22b-thinking-2507")
        logger.info(f"🤖 Enhanced LLM Labeler: {label_generator.model_name}")
        logger.info(f"📁 Cache directory: {label_generator.cache_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize label generator: {e}")
        logger.error("Please check CEREBRAS_API_KEY environment variable")
        return
    
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
    

    
    subset_sizes = [10,50,100,247,507,750,1000,1250,1500,1750,2000,2250,2500]

    # subset_sizes = [10,20]
    gt_ratios = [0, 25, 50, 75, 100]
    
    results = {
        'no_worst_examples': [],
        'gliner_ft_0gt_100llm_f1': [],
        'gliner_ft_25gt_75llm_f1': [],
        'gliner_ft_50gt_50llm_f1': [],
        'gliner_ft_75gt_25llm_f1': [],
        'gliner_ft_100gt_0llm_f1': [],
        'confidence': [],
        'avg_entities': [],
        'avg_input_tokens': [],
        'model_input_output': [],
        'avg_output_tokens': [],
        'completion_status': [],
        'completion_percentage': []
    }
    
    label_cache = []
    
    total_iterations = len(subset_sizes) * len(gt_ratios)
    logger.info(f"\n🔬 Enhanced Mixed Ratio Experiment Overview:")
    logger.info(f"   • Subset sizes to test: {subset_sizes}")
    logger.info(f"   • GT ratios to test: {gt_ratios}%")
    logger.info(f"   • Total model trainings: {total_iterations}")
    logger.info(f"   • Evaluation dataset: FULL test set ({len(test_data)} examples)")
    logger.info(f"   • API resilience: ✅ Enabled")
    logger.info(f"   • Incremental saving: ✅ Enabled")
    
    logger.info(f"\n🚀 Starting Enhanced Mixed Ratio Analysis...")
    logger.info("-" * 60)
    
    experiment_interrupted = False
    
    for subset_idx, n_examples in enumerate(tqdm(subset_sizes, desc="Training Mixed Ratios", position=0)):
        if experiment_interrupted:
            logger.warning(f"⏭️  Skipping remaining subset sizes due to quota limit")
            break
            
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 ITERATION {subset_idx+1}/{len(subset_sizes)}: Processing {n_examples} examples")
        logger.info(f"{'='*60}")
        
        train_subset = low_n[:n_examples]
        
        logger.info(f"🤖 Generating LLM labels for {n_examples} examples...")
        
        try:
            llm_labeled_data = label_generator.generate(
                low_n_examples=train_subset,
                num_samples=n_examples,
                entity_types=entity_types,
                label_cache=label_cache,
                verbose=True
            )
            
            actual_examples = n_examples
            completion_status = "complete"
            completion_pct = 100.0
            
            logger.info(f"✅ Successfully generated {len(llm_labeled_data)} labels")
            
        except QuotaExceededException as qe:
            logger.error("="*60)
            logger.error("🚨 QUOTA EXCEEDED - HANDLING GRACEFULLY")
            logger.error("="*60)
            logger.error(f"Full message: {qe.message}")
            logger.error(f"Requested: {qe.requested} labels")
            logger.error(f"Generated: {qe.actual} labels")
            logger.error(f"Completion: {(qe.actual/qe.requested)*100:.1f}%")
            logger.error("="*60)
            
            llm_labeled_data = qe.partial_labels
            actual_examples = qe.actual
            completion_status = "quota_exceeded"
            completion_pct = (qe.actual / qe.requested) * 100
            experiment_interrupted = True
            
            logger.warning(f"⚠️  Will complete current iteration with {actual_examples} labels, then stop experiment")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error during label generation: {str(e)}")
            logger.error("Skipping this iteration and continuing...")
            continue
        
        if len(llm_labeled_data) > 0:
            avg_entities = sum(len(ex['ner']) for ex in llm_labeled_data) / len(llm_labeled_data)
            
            # Calculate ACTUAL token metrics from cached labels with metadata
            samples_with_tokens = [ex for ex in llm_labeled_data if '_token_input' in ex]
            
            if samples_with_tokens:
                avg_input = sum(ex['_token_input'] for ex in samples_with_tokens) / len(samples_with_tokens)
                avg_output = sum(ex['_token_output'] for ex in samples_with_tokens) / len(samples_with_tokens)
                
                token_metrics = {
                    'avg_input_tokens': avg_input,
                    'model_input_output': (65536, 60000),
                    'avg_output_tokens': avg_output
                }
            else:
                # Fallback if no token metadata (shouldn't happen with fixed enc_api_label)
                token_metrics = {
                    'avg_input_tokens': 0,
                    'model_input_output': (65536, 60000),
                    'avg_output_tokens': 0
                }
        else:
            logger.error(f"❌ No valid LLM labeled data for {n_examples} examples")
            logger.error("Skipping this iteration...")
            continue
        
        logger.info(f"📊 Metrics: avg_entities={avg_entities:.1f}, completion={completion_pct:.1f}%")
        logger.info(f"📊 Token metrics: input={token_metrics['avg_input_tokens']:.0f}, output={token_metrics['avg_output_tokens']:.0f}")
        
        ratio_f1_scores = []
        avg_confidence = 0.0
        
        effective_examples = len(llm_labeled_data)
        
        for gt_ratio in gt_ratios:
            logger.info(f"\n🔥 Training GLiNER with {gt_ratio}% GT + {100-gt_ratio}% LLM labels")
            logger.info(f"   Available data: {effective_examples} examples")
            
            if effective_examples < 1:
                logger.warning(f"   ⚠️  Insufficient data ({effective_examples} examples) - recording zero F1")
                ratio_f1_scores.append(0.0)
                continue
            
            if gt_ratio == 0:
                mixed_training_data = llm_labeled_data
                logger.info(f"   Using 100% LLM labels ({len(mixed_training_data)} examples)")
            elif gt_ratio == 100:
                mixed_training_data = [{
                    "tokenized_text": ex["tokenized_text"],
                    "ner": ex["ner"]
                } for ex in train_subset[:effective_examples]]
                logger.info(f"   Using 100% GT labels ({len(mixed_training_data)} examples)")
            else:
                gt_subset = train_subset[:effective_examples]
                mixed_training_data = create_mixed_training_data(
                    gt_subset, llm_labeled_data, gt_ratio
                )
                n_gt = int(len(mixed_training_data) * gt_ratio / 100)
                n_llm = len(mixed_training_data) - n_gt
                logger.info(f"   Using {n_gt} GT + {n_llm} LLM labels ({len(mixed_training_data)} total)")
            
            adapter_path = f"../models/mixed_ratio_model_{effective_examples}_{gt_ratio}gt"
            
            try:
                model = intialize_model(logger=logger)
                model.to(device)
                
                train_lora_model(
                    model=model,
                    train_data=mixed_training_data,
                    eval_data=test_data[:100],
                    training_config=training_config,
                    adapter_save_path=adapter_path,
                    logger=logger
                )
                
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
                logger.info(f"📊 Evaluating {gt_ratio}% GT model on FULL test set...")
                
                eval_model = load_evaluation_model(adapter_path, device, logger=logger)
                
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
                
                del eval_model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Training/evaluation failed for {gt_ratio}% GT ratio: {str(e)}")
                ratio_f1_scores.append(0.0)
        
        while len(ratio_f1_scores) < 5:
            ratio_f1_scores.append(0.0)
        
        results['no_worst_examples'].append(actual_examples)
        results['gliner_ft_0gt_100llm_f1'].append(ratio_f1_scores[0])
        results['gliner_ft_25gt_75llm_f1'].append(ratio_f1_scores[1])
        results['gliner_ft_50gt_50llm_f1'].append(ratio_f1_scores[2])
        results['gliner_ft_75gt_25llm_f1'].append(ratio_f1_scores[3])
        results['gliner_ft_100gt_0llm_f1'].append(ratio_f1_scores[4])
        results['confidence'].append(avg_confidence / len(gt_ratios) if avg_confidence > 0 else 0.0)
        results['avg_entities'].append(avg_entities)
        results['avg_input_tokens'].append(token_metrics['avg_input_tokens'])
        results['model_input_output'].append(token_metrics['model_input_output'])
        results['avg_output_tokens'].append(token_metrics['avg_output_tokens'])
        results['completion_status'].append(completion_status)
        results['completion_percentage'].append(completion_pct)
        
        logger.info(f"💾 Results stored for {actual_examples} examples")
        logger.info(f"📊 F1 Scores: 0%GT={ratio_f1_scores[0]:.1f}%, 25%GT={ratio_f1_scores[1]:.1f}%, 50%GT={ratio_f1_scores[2]:.1f}%, 75%GT={ratio_f1_scores[3]:.1f}%, 100%GT={ratio_f1_scores[4]:.1f}%")
        logger.info(f"📈 Status: {completion_status} ({completion_pct:.1f}% complete)")
        
        temp_df = pd.DataFrame(results)
        incremental_path = "../results/api/mixed_ratio_performance_incremental.csv"
        os.makedirs(os.path.dirname(incremental_path), exist_ok=True)
        temp_df.to_csv(incremental_path, index=False)
        logger.info(f"💾 Saved incremental results: {len(temp_df)} iterations completed")
        
        if experiment_interrupted:
            logger.error("="*60)
            logger.error("🛑 EXPERIMENT INTERRUPTED DUE TO QUOTA EXCEEDED")
            logger.error("="*60)
            logger.error(f"Completed iterations: {len(results['no_worst_examples'])}")
            logger.error(f"Last successful size: {actual_examples} examples")
            logger.error("Proceeding to final analysis with collected data...")
            logger.error("="*60)
            break
    
    logger.info(f"\n📋 Creating Final Results DataFrame...")
    
    final_results_df = pd.DataFrame(results)
    final_results_df = final_results_df[final_results_df['no_worst_examples'] > 0]
    
    completed_iterations = len(final_results_df)
    planned_iterations = len(subset_sizes)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    logger.info("\n" + "="*60)
    logger.info("ENHANCED MIXED RATIO FINE-TUNING ANALYSIS RESULTS")
    if completed_iterations < planned_iterations:
        logger.warning(f"⚠️  PARTIAL RESULTS: {completed_iterations}/{planned_iterations} iterations")
        logger.warning(f"Experiment stopped at {final_results_df['no_worst_examples'].iloc[-1]} examples")
    else:
        logger.info(f"✅ COMPLETE RESULTS: {completed_iterations}/{planned_iterations} iterations")
    logger.info("="*60)
    logger.info("\n" + final_results_df.to_string(index=False))
    
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows') 
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    results_filename = "mixed_ratio_finetuning_performance_qwen_enhanced"
    if completed_iterations < planned_iterations:
        results_filename += "_partial"
    
    results_path = f"../results/api/{results_filename}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Final results saved to: {results_path}")
    
    logger.info(f"\n📈 Generating Enhanced Mixed Ratio Performance Plot...")
    
    if len(final_results_df) == 0:
        logger.error("❌ No data to plot - experiment failed completely")
        return
    
    plt.style.use('default')
    sns.set_palette("viridis")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
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
    
    for i, (idx, row) in enumerate(final_results_df.iterrows()):
        if row['completion_status'] != 'complete':
            for col_name, label, color in ratio_columns:
                y_val = row[col_name]
                if y_val > 0:
                    ax.scatter(row['no_worst_examples'], y_val, 
                             marker='x', s=100, color=color, alpha=0.7)
    
    title = 'Enhanced Mixed Ratio Fine-tuning Performance: GT vs LLM Labels'
    if completed_iterations < planned_iterations:
        title += f'\n(Partial Results - Stopped at {final_results_df["no_worst_examples"].iloc[-1]} examples)'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples (Training)', fontsize=14)
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    if completed_iterations < planned_iterations:
        ax.text(0.02, 0.98, 'Legend:\n○ Complete data\n× Partial data (API quota hit)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    annotation_interval = 2 if completed_iterations < planned_iterations else 3
    for i, (col_name, label, color) in enumerate(ratio_columns):
        if i % 2 == 0:
            for j, (x, y) in enumerate(zip(final_results_df['no_worst_examples'], 
                                         final_results_df[col_name])):
                if j % annotation_interval == 0 and y > 0:
                    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=9, color=color)
    
    plt.tight_layout()
    
    plot_filename = f"../results/api/{results_filename}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_filename}")
    plt.show()
    
    logger.info(f"\n🎉 Enhanced Mixed Ratio Analysis Summary:")
    logger.info(f"📊 Completed iterations: {completed_iterations}/{planned_iterations}")
    
    if completed_iterations < planned_iterations:
        logger.warning("="*60)
        logger.warning("⚠️  EXPERIMENT INCOMPLETE")
        logger.warning("="*60)
        logger.warning("API quota was exceeded during execution")
        logger.warning(f"Results available for: {list(final_results_df['no_worst_examples'])}")
        logger.warning("To complete experiment:")
        logger.warning("  1. Wait for quota reset (typically 24 hours)")
        logger.warning("  2. Re-run script - will resume from disk cache")
        logger.warning("  3. Or use partial results for preliminary analysis")
        logger.warning("="*60)
    else:
        logger.info("✅ All iterations completed successfully!")
    
    for col_name, label, _ in ratio_columns:
        if not final_results_df[col_name].empty:
            best_f1 = final_results_df[col_name].max()
            if best_f1 > 0:
                best_idx = final_results_df[col_name].idxmax()
                best_examples = final_results_df.loc[best_idx, 'no_worst_examples']
                logger.info(f"🏆 {label}: Best F1={best_f1:.1f}% with {best_examples} examples")
            else:
                logger.info(f"❌ {label}: No successful results")
    
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
    
    if completed_iterations < planned_iterations:
        logger.info(f"\n📋 Next Steps:")
        logger.info(f"   1. Wait for API quota reset (typically 24 hours)")
        logger.info(f"   2. Re-run this script - it will automatically resume from cache")
        logger.info(f"   3. Experiment will continue from where it left off")
        logger.info(f"   4. Or analyze partial results for preliminary insights")


if __name__ == "__main__":
    main()