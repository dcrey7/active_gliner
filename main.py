"""
Complete Active Learning Pipeline with Proper Training Pool and Test Separation
Fixed data leakage issue - low confidence examples come from training pool, not test data
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import ollama
from tqdm import tqdm
import re
import json
import random
import numpy as np
import torch
from gliner import GLiNER
import traceback
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import gc
import time
from peft import LoraConfig, get_peft_model, TaskType
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from transformers import TrainerCallback
import warnings
import psutil
import seaborn as sns
import hashlib

# ===============================================================================
# GLOBAL SETUP
# ===============================================================================

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 only
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

warnings.filterwarnings('ignore')

# Cache for storing analysis results and synthetic data with BATCH-LEVEL OPTIMIZATION
BATCH_ANALYSIS_CACHE: Dict[str, Dict] = {}  # Key: batch_hash, Value: batch analysis
FINAL_SUMMARY_CACHE: Dict[int, Dict] = {}   # Key: num_examples, Value: final summary
SYNTHETIC_CACHE: Dict[int, List] = {}       # Key: num_examples, Value: synthetic data list

# Global constants
BATCH_SIZE = 8  # Consistent batch size for training and evaluation
GLOBAL_SEED = 42

# ===============================================================================
# LOGGING SETUP
# ===============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging for the pipeline"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/active_learning_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger('ActiveLearning')
    logger.info("="*80)
    logger.info("ACTIVE LEARNING PIPELINE WITH PROPER TRAIN/TEST SEPARATION")
    logger.info("="*80)
    logger.info(f"Log file: {log_filename}")
    
    return logger

logger = setup_logging()

# ===============================================================================
# REPRODUCIBILITY SETUP
# ===============================================================================

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    logger.info(f"Setting all seeds to {seed} for reproducibility...")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds at the very beginning
set_all_seeds(GLOBAL_SEED)

# ===============================================================================
# DEVICE SETUP
# ===============================================================================

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Verify GPU setup
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"Number of GPUs visible: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"Current GPU: {torch.cuda.current_device()}")
    logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

logger.info(f"Using consistent batch size: {BATCH_SIZE}")

# ===============================================================================
# MEMORY CLEANUP UTILITY
# ===============================================================================

def cleanup_memory():
    """Clean up memory between experiments"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Memory cleanup completed")

# ===============================================================================
# DATA LOADING
# ===============================================================================

def load_mit_dataset(data_path, labels_path, split_name="train"):
    """Load and process MIT dataset in GLiNER format"""
    logger.info(f"Loading {split_name} data from: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    processed_data = []
    
    for item in data:
        words = item['sentence'].split()
        entities = []
        
        for entity in item['entities']:
            start_char, end_char = entity['pos']
            char_count = 0
            start_word = None
            end_word = None
            
            for i, word in enumerate(words):
                word_length = len(word)
                if char_count == start_char:
                    start_word = i
                if char_count + word_length == end_char:
                    end_word = i
                    break
                char_count += word_length + 1
            
            if start_word is not None and end_word is not None:
                entities.append((start_word, end_word, entity['type'].lower()))
        
        processed_data.append({
            "tokenized_text": words,
            "ner": entities
        })
    
    entity_types = [label.lower() for label in labels]
    logger.info(f"Processed {len(processed_data)} examples")
    logger.info(f"Entity types: {entity_types}")
    
    return processed_data, entity_types

# ===============================================================================
# ENHANCED EVALUATION (adapted from user's code)
# ===============================================================================

def compare_entities(ground_truth, predictions):
    """Compare ground truth and predictions to identify errors"""
    
    gt_set = set()
    pred_set = set()
    pred_scores = {}
    
    # Process ground truth
    for ent in ground_truth:
        gt_set.add((ent[0], ent[1], ent[2]))
    
    # Process predictions
    for i, ent in enumerate(predictions):
        span_tuple = (ent[0], ent[1], ent[2])
        pred_set.add(span_tuple)
        pred_scores[span_tuple] = ent[4] if len(ent) > 4 else 1.0
    
    # Calculate confusion matrix elements
    false_negatives = gt_set - pred_set
    false_positives = pred_set - gt_set
    true_positives = gt_set & pred_set
    
    return {
        'true_positives': list(true_positives),
        'false_positives': list(false_positives), 
        'false_negatives': list(false_negatives),
        'pred_scores': pred_scores
    }

def calculate_overall_metrics(raw_predictions, ground_truths=None):
    """Calculate overall accuracy and confidence metrics"""
    
    # Calculate overall confidence using raw predictions from model.run()
    all_scores = []
    total_predictions = 0
    
    for pred_batch in raw_predictions:
        for pred in pred_batch:
            # Raw predictions are always in dictionary format
            if 'score' in pred:
                all_scores.append(pred['score'])
            total_predictions += 1
    
    overall_confidence = np.mean(all_scores) if all_scores else 0.0
    
    metrics = {
        'total_predictions': total_predictions,
        'overall_confidence': overall_confidence,
        'overall_confidence_pct': overall_confidence * 100
    }
    
    if ground_truths is not None:
        # Calculate accuracy metrics when ground truth is available
        total_examples = len(ground_truths)
        # Note: This would need full comparison logic
        metrics.update({
            'total_examples': total_examples,
            'entity_level_accuracy': 0,  # Placeholder
            'entity_level_accuracy_pct': 0,  # Placeholder
            'example_level_accuracy': 0,  # Placeholder  
            'example_level_accuracy_pct': 0,  # Placeholder
        })
    
    return metrics

def generate_classification_report(entity_stats, entity_prediction_scores):
    """Generate enhanced classification report with confidence metrics"""
    
    df_data = []
    total_tp = total_fp = total_fn = 0
    all_prediction_scores = []
    
    for entity_type, stats in entity_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        # Calculate average prediction confidence
        avg_prediction_confidence = np.mean(entity_prediction_scores[entity_type]) if entity_prediction_scores[entity_type] else 0.0
        all_prediction_scores.extend(entity_prediction_scores[entity_type])
        
        df_data.append({
            'entity_type': entity_type,
            'tp': tp,
            'fp': fp, 
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'avg_prediction_confidence': avg_prediction_confidence
        })
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Add aggregate metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Macro averages
    macro_precision = np.mean([row['precision'] for row in df_data])
    macro_recall = np.mean([row['recall'] for row in df_data]) 
    macro_f1 = np.mean([row['f1'] for row in df_data])
    
    # Confidence averages
    micro_avg_confidence = np.mean(all_prediction_scores) if all_prediction_scores else 0.0
    macro_avg_confidence = np.mean([row['avg_prediction_confidence'] for row in df_data if row['avg_prediction_confidence'] > 0])
    
    # Add aggregate rows
    df_data.extend([
        {
            'entity_type': 'micro_avg',
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1,
            'support': total_tp + total_fn,
            'avg_prediction_confidence': micro_avg_confidence
        },
        {
            'entity_type': 'macro_avg',
            'tp': '-', 'fp': '-', 'fn': '-',
            'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1,
            'support': total_tp + total_fn,
            'avg_prediction_confidence': macro_avg_confidence if not np.isnan(macro_avg_confidence) else 0.0
        }
    ])
    
    return pd.DataFrame(df_data)

def enhanced_evaluate(model, data, entity_types, threshold=0.5, batch_size=16):
    """
    Enhanced evaluation with proper NER metrics
    
    Args:
        model: GLiNER model
        data: List of examples with tokenized_text and ner (ground truth)
        entity_types: List of entity types to predict
        threshold: Prediction threshold
        batch_size: Batch size for inference
    
    Returns:
        Dictionary containing analysis results
    """
    
    logger.info("Running enhanced evaluation...")
    
    # Prepare data for model inference
    texts = []
    ground_truths = []
    
    for example in data:
        text = " ".join(example["tokenized_text"])
        texts.append(text)
        ground_truths.append(example["ner"])
    
    logger.info(f"Processing {len(texts)} examples...")
    
    # Run model predictions
    with torch.no_grad():
        all_predictions = model.run(
            texts, entity_types, 
            flat_ner=True, threshold=threshold, batch_size=batch_size
        )
    
    # Process predictions to match ground truth format
    processed_predictions = []
    for i, predictions in enumerate(all_predictions):
        tokenized_text = data[i]["tokenized_text"]
        pred_entities = []
        
        for pred in predictions:
            # Convert character positions back to word positions
            text = " ".join(tokenized_text)
            char_start = pred['start']
            char_end = pred['end']
            
            # Find word positions
            word_start = None
            word_end = None
            char_pos = 0
            
            for word_idx, word in enumerate(tokenized_text):
                word_len = len(word)
                if char_pos <= char_start < char_pos + word_len:
                    word_start = word_idx
                if char_pos < char_end <= char_pos + word_len:
                    word_end = word_idx
                    break
                char_pos += word_len + 1  # +1 for space
            
            if word_start is not None and word_end is not None:
                pred_entities.append([
                    word_start, word_end, pred['label'].lower(), 
                    pred['text'], pred['score']
                ])
        
        processed_predictions.append(pred_entities)
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(all_predictions, ground_truths)
    
    # Full analysis with ground truth
    entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    entity_prediction_scores = defaultdict(list)
    
    logger.info("Analyzing errors with ground truth...")
    
    for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
        comparison = compare_entities(gt, pred)

        # Update entity statistics
        for tp in comparison['true_positives']:
            entity_type = tp[2]
            entity_stats[entity_type]['tp'] += 1
            score = comparison['pred_scores'].get((tp[0], tp[1], tp[2]), 1.0)
            entity_prediction_scores[entity_type].append(score)
            
        for fp in comparison['false_positives']:
            entity_type = fp[2]
            entity_stats[entity_type]['fp'] += 1
            score = comparison['pred_scores'].get((fp[0], fp[1], fp[2]), 1.0)
            entity_prediction_scores[entity_type].append(score)
            
        for fn in comparison['false_negatives']:
            entity_stats[fn[2]]['fn'] += 1
    
    # Create full predictions with ground truth info for low confidence extraction
    full_predictions_gt = []
    for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
        full_predictions_gt.append({
            "tokenized_text": data[i]["tokenized_text"],
            "ner": gt,
            "predictions": [[p[0], p[1], p[2]] for p in pred],
            "scores": [p[4] for p in pred]
        })
    
    # Generate classification report
    classification_report = generate_classification_report(entity_stats, entity_prediction_scores)
    
    # Extract overall F1 and confidence
    overall_f1 = classification_report[classification_report['entity_type'] == 'micro_avg']['f1'].iloc[0]
    
    # Update overall metrics
    overall_metrics.update({
        'overall_f1': overall_f1,
        'overall_f1_pct': overall_f1 * 100,
    })
    
    results = {
        'overall_metrics': overall_metrics,
        'classification_report_df': classification_report,  # Store raw dataframe for F1 extraction
        'all_predictions': full_predictions_gt
    }
    
    return results

# ===============================================================================
# EXTRACT LOWEST CONFIDENCE EXAMPLES FROM TRAINING POOL EVALUATION RESULTS
# ===============================================================================

def get_lowest_score_examples_sorted(training_pool_results, n=5):
    """Get n examples with lowest minimum scores from training pool evaluation results. Returns empty list if n=0."""
    if n == 0:
        logger.info("Requested 0 examples - returning empty list")
        return []
        
    logger.info(f"Extracting {n} lowest confidence examples from training pool evaluation results...")
    examples = training_pool_results['all_predictions']
    
    # Sort examples by their minimum score
    sorted_examples = sorted(
        examples, 
        key=lambda x: min(x['scores']) if x['scores'] else 1.0
    )
    
    logger.info(f"Total examples available: {len(examples)}")
    logger.info(f"Returning {min(n, len(sorted_examples))} lowest confidence examples")
    
    # Log sample of lowest confidence examples
    for i, ex in enumerate(sorted_examples[:min(3, n)]):
        min_score = min(ex['scores']) if ex['scores'] else 0.0
        text_preview = ' '.join(ex['tokenized_text'][:8])
        logger.info(f"  Example {i+1}: score={min_score:.3f}, text='{text_preview}...'")
    
    return sorted_examples[:n]

# ===============================================================================
# BATCH CACHING UTILITIES
# ===============================================================================

def get_batch_cache_key(batch_examples):
    """Create a unique hash key for a batch based on its content"""
    # Create a string representation of the batch content
    batch_content = []
    for example in batch_examples:
        text = " ".join(example['tokenized_text'])
        ner = str(example['ner'])
        predictions = str(example['predictions'])
        scores = str(example['scores'])
        batch_content.append(f"{text}|{ner}|{predictions}|{scores}")
    
    # Create hash of the batch content
    batch_string = "||".join(batch_content)
    batch_hash = hashlib.md5(batch_string.encode()).hexdigest()
    return batch_hash

def log_cache_status():
    """Log current cache status"""
    logger.info(f"Cache status:")
    logger.info(f"  Batch analysis cache: {len(BATCH_ANALYSIS_CACHE)} batches")
    logger.info(f"  Final summary cache: {len(FINAL_SUMMARY_CACHE)} summaries")
    logger.info(f"  Synthetic data cache: {len(SYNTHETIC_CACHE)} datasets")

# ===============================================================================
# OPTIMIZED ANALYSIS PIPELINE WITH BATCH CACHING
# ===============================================================================

def analyze_single_batch_with_retry(batch_examples, entity_types, batch_num, max_retries=3):
    """Analyze a single batch with retry logic"""
    
    logger.info(f"Analyzing batch {batch_num} ({len(batch_examples)} examples)")
    
    # Build prompt for this batch
    prompt = f"""CRITICAL: You are an expert NER analyst. This is a PRODUCTION system.

TARGET ENTITY TYPES: {', '.join(entity_types)}

BATCH EXAMPLES:
"""
    
    # Add batch examples
    for i, example in enumerate(batch_examples):
        text = " ".join(example['tokenized_text'])
        min_score = min(example['scores']) if example['scores'] else 0.0
        
        prompt += f"""
Example {i + 1}:
Text: {text}
Ground Truth: {example['ner']}
Predictions: {example['predictions']}
Scores: {example['scores']}
Min Score: {min_score:.3f}
"""
    
    # Add structured analysis request
    prompt += f"""

MANDATORY TASK: Provide domain summary, then analyze each entity type.

CRITICAL: OUTPUT MUST BE VALID JSON ONLY. NO OTHER TEXT ALLOWED.

MANDATORY OUTPUT JSON FORMAT:
{{
  "domain_summary": "brief description of domain patterns in this batch",
  "entity_analysis": {{"""
    
    # Add structure for each entity type
    for i, entity_type in enumerate(entity_types):
        if i > 0:
            prompt += ","
        prompt += f"""
    "{entity_type}": {{
      "position_analysis": "boundary/position issues for {entity_type} with specific text examples",
      "good_predicted_examples": ["actual text from examples that worked well", "another good text example"],
      "bad_predicted_examples": ["actual text from examples that failed", "another bad text example"],
      "variation_analysis": "variations needed for {entity_type} with specific examples"
    }}"""
    
    prompt += f"""
  }}
}}

CRITICAL REQUIREMENTS:
- MUST OUTPUT ONLY VALID JSON
- Focus ONLY on: {', '.join(entity_types)}
- Use ACTUAL TEXT from examples, not "example 1" or "example 2"  
- Show specific text snippets that worked/failed
- If entity not in batch, put "No examples in this batch"
- Keep it concise but use real text examples

MANDATORY: Generate ONLY the JSON structure above, nothing else:
"""
    
    # Retry logic for this batch
    for attempt in range(max_retries):
        logger.info(f"Batch {batch_num} - Attempt {attempt + 1}/{max_retries}")
        
        try:
            response = ollama.generate(
                model='mistral:latest',
                prompt=prompt,
                options={
                    'top_k': 50,
                    'top_p': 0.9,
                    'num_predict': 2500,
                    'temperature': 0.2,
                }
            )
            
            response_text = response['response'].strip()
            logger.info(f"Received response for batch {batch_num} attempt {attempt + 1} (length: {len(response_text)})")
            
            # Try to extract JSON
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                batch_result = json.loads(json_text)
                batch_result['batch_number'] = batch_num
                logger.info(f"✅ Batch {batch_num} completed successfully on attempt {attempt + 1}")
                return batch_result
            else:
                logger.warning(f"❌ No JSON found in batch {batch_num} attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    logger.error(f"Response preview: {response_text[:200]}...")
                
        except json.JSONDecodeError as e:
            logger.warning(f"❌ JSON parsing failed for batch {batch_num} attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Response preview: {response_text[:200]}...")
        except Exception as e:
            logger.warning(f"❌ LLM call failed for batch {batch_num} attempt {attempt + 1}: {e}")
    
    logger.error(f"❌❌❌ CRITICAL: Batch {batch_num} failed after {max_retries} attempts")
    return None


def analyze_domain_with_batch_caching(low_confidence_examples, entity_types, batch_size=10):
    """Process examples in batches with intelligent batch caching"""
    
    total_examples = len(low_confidence_examples)
    logger.info("="*60)
    logger.info("DOMAIN ANALYSIS WITH BATCH CACHING")
    logger.info("="*60)
    logger.info(f"Analyzing {total_examples} examples in batches of {batch_size}")
    logger.info(f"Entity types: {entity_types}")
    
    all_results = []
    total_batches = (total_examples - 1) // batch_size + 1
    cache_hits = 0
    new_analyses = 0
    
    # Process in batches with caching
    for batch_start in range(0, total_examples, batch_size):
        batch_end = min(batch_start + batch_size, total_examples)
        batch_examples = low_confidence_examples[batch_start:batch_end]
        batch_num = batch_start//batch_size + 1
        
        # Check if this batch has been analyzed before
        batch_key = get_batch_cache_key(batch_examples)
        
        if batch_key in BATCH_ANALYSIS_CACHE:
            logger.info(f"📋 Batch {batch_num}/{total_batches}: Using cached analysis (examples {batch_start+1}-{batch_end})")
            cached_result = BATCH_ANALYSIS_CACHE[batch_key].copy()
            cached_result['batch_number'] = batch_num  # Update batch number
            all_results.append(cached_result)
            cache_hits += 1
        else:
            logger.info(f"🔄 Batch {batch_num}/{total_batches}: New analysis needed (examples {batch_start+1}-{batch_end})")
            
            batch_result = analyze_single_batch_with_retry(batch_examples, entity_types, batch_num)
            
            if batch_result is None:
                logger.error(f"❌❌❌ CRITICAL: Batch {batch_num} analysis failed")
                return None
            
            # Cache the result for future use
            BATCH_ANALYSIS_CACHE[batch_key] = batch_result.copy()
            all_results.append(batch_result)
            new_analyses += 1
    
    logger.info("="*60)
    logger.info("BATCH ANALYSIS COMPLETED")
    logger.info("="*60)
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Cache hits: {cache_hits}")
    logger.info(f"New analyses: {new_analyses}")
    logger.info(f"Cache efficiency: {cache_hits / total_batches * 100:.1f}%")
    
    return all_results


def combine_all_batch_results(all_batch_results, entity_types):
    """Combine results from all batches into one structure"""
    
    logger.info("="*60)
    logger.info("COMBINING BATCH RESULTS")
    logger.info("="*60)
    logger.info(f"Combining {len(all_batch_results)} batch results for entity types: {entity_types}")
    
    combined_result = {
        'total_batches': len(all_batch_results),
        'entity_types': entity_types,
        'domain_summaries': [],
        'combined_entity_analysis': {}
    }
    
    # Initialize combined entity analysis
    for entity_type in entity_types:
        combined_result['combined_entity_analysis'][entity_type] = {
            'position_analysis': [],
            'good_predicted_examples': [],
            'bad_predicted_examples': [],
            'variation_analysis': []
        }
    
    # Combine all batch results
    for result in all_batch_results:
        domain_summary = result.get('domain_summary', '')
        if domain_summary:
            combined_result['domain_summaries'].append(f"Batch {result.get('batch_number', '?')}: {domain_summary}")
        
        entity_analysis = result.get('entity_analysis', {})
        for entity_type in entity_types:
            if entity_type in entity_analysis:
                entity_data = entity_analysis[entity_type]
                
                pos_analysis = entity_data.get('position_analysis', '')
                if pos_analysis and 'No examples' not in pos_analysis:
                    combined_result['combined_entity_analysis'][entity_type]['position_analysis'].append(pos_analysis)
                
                good_examples = entity_data.get('good_predicted_examples', [])
                if isinstance(good_examples, list):
                    for example in good_examples:
                        if example and 'No examples' not in example:
                            combined_result['combined_entity_analysis'][entity_type]['good_predicted_examples'].append(example)
                
                bad_examples = entity_data.get('bad_predicted_examples', [])
                if isinstance(bad_examples, list):
                    for example in bad_examples:
                        if example and 'No examples' not in example:
                            combined_result['combined_entity_analysis'][entity_type]['bad_predicted_examples'].append(example)
                
                var_analysis = entity_data.get('variation_analysis', '')
                if var_analysis and 'No examples' not in var_analysis:
                    combined_result['combined_entity_analysis'][entity_type]['variation_analysis'].append(var_analysis)
    
    logger.info(f"✅ Combined {len(all_batch_results)} batches successfully")
    return combined_result


def create_final_summary_with_retry(combined_result, max_retries=3):
    """Send combined results to LLM for final reasoning and summarization with retry logic"""
    
    logger.info("="*60)
    logger.info("CREATING FINAL SUMMARY (WITH RETRY LOGIC)")
    logger.info("="*60)
    logger.info(f"Max retries: {max_retries}")
    
    prompt = f"""CRITICAL: You are an expert NER analyst. This is a PRODUCTION system.

I have analyzed multiple batches of NER examples. 

ENTITY TYPES ANALYZED: {', '.join(combined_result['entity_types'])}

DOMAIN SUMMARIES FROM BATCHES:
{chr(10).join(f"- {summary}" for summary in combined_result['domain_summaries'])}

DETAILED ENTITY ANALYSIS:
"""
    
    for entity_type, data in combined_result['combined_entity_analysis'].items():
        if any([data['position_analysis'], data['good_predicted_examples'], 
               data['bad_predicted_examples'], data['variation_analysis']]):
            prompt += f"""
{entity_type.upper()}:
Position Analysis: {'; '.join(data['position_analysis'])}
Good Examples: {'; '.join(data['good_predicted_examples'])}
Bad Examples: {'; '.join(data['bad_predicted_examples'])}  
Variation Analysis: {'; '.join(data['variation_analysis'])}
"""
    
    prompt += f"""

MANDATORY TASK: Based on all this data, provide a final reasoned summary.

CRITICAL: OUTPUT MUST BE VALID JSON ONLY. NO OTHER TEXT ALLOWED.

MANDATORY OUTPUT JSON FORMAT:
{{
  "domain_summary": "Overall domain description and main patterns",
  "entity_summaries": {{"""
    
    for i, entity_type in enumerate(combined_result['entity_types']):
        if i > 0:
            prompt += ","
        prompt += f"""
    "{entity_type}": {{
      "position_summary": "Key position/boundary issues for {entity_type}",
      "good_examples_summary": "What works well for {entity_type} prediction",
      "bad_examples_summary": "What fails for {entity_type} prediction", 
      "variations_summary": "Key variations needed for {entity_type}"
    }}"""
    
    prompt += f"""
  }}
}}

CRITICAL REQUIREMENTS:
- MUST OUTPUT ONLY VALID JSON
- Synthesize findings from all batches
- Focus on patterns across all examples
- Keep summaries concise but informative
- If no data for an entity, put "No significant findings"

MANDATORY: Generate ONLY the JSON structure above, nothing else:
"""
    
    # Retry logic for final summary
    for attempt in range(max_retries):
        logger.info(f"Final Summary - Attempt {attempt + 1}/{max_retries}")
        
        try:
            response = ollama.generate(
                model='mistral:latest',
                prompt=prompt,
                options={
                    'top_k': 50,
                    'top_p': 0.9,
                    'num_predict': 2000,
                    'temperature': 0.1,
                }
            )
            
            response_text = response['response'].strip()
            logger.info(f"Received final summary response attempt {attempt + 1} (length: {len(response_text)})")
            
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                final_summary = json.loads(json_text)
                logger.info(f"✅ Final summary created successfully on attempt {attempt + 1}!")
                return final_summary
            else:
                logger.warning(f"❌ No JSON found in final summary attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    logger.error(f"Response preview: {response_text[:200]}...")
                
        except json.JSONDecodeError as e:
            logger.warning(f"❌ JSON parsing failed for final summary attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Response preview: {response_text[:200]}...")
        except Exception as e:
            logger.warning(f"❌ LLM call failed for final summary attempt {attempt + 1}: {e}")
    
    logger.error("❌❌❌ CRITICAL: Final summary failed after all retry attempts")
    return None

# ===============================================================================
# SYNTHETIC DATA GENERATION WITH VALIDATION AND ZERO CORRECTED EXAMPLES SUPPORT
# ===============================================================================

def create_baseline_synthetic_prompt(entity_types, **kwargs):
    """Create baseline prompt for synthetic data when no corrected examples available"""
    
    # Filter attributes
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}
    
    # Build base prompt
    prompt = """CRITICAL: This is a PRODUCTION system for generating training data.

**Objective:**
Generate realistic movie review text passages that include clearly identified named entities. 

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled in the 'entities' list
- Follow the exact format shown in the examples below

**Entity Types to Focus On:**
"""
    
    # DYNAMIC ENTITY TYPES
    for entity_type in entity_types:
        prompt += f"- {entity_type}: Entities of type {entity_type}\n"
    
    # Add baseline examples (hardcoded since no corrected examples available)
    prompt += """
**BASELINE EXAMPLES:**
Here are examples showing the expected format and entity types:

Example 1:
{
  "text": "I loved the movie starring Tom Hanks and directed by Steven Spielberg in 1998.",
  "entities": [
    {"entity": "Tom Hanks", "types": ["actor"]},
    {"entity": "Steven Spielberg", "types": ["director"]},
    {"entity": "1998", "types": ["year"]}
  ]
}

Example 2:
{
  "text": "The comedy film was shot in Los Angeles and featured great performances.",
  "entities": [
    {"entity": "comedy", "types": ["genre"]},
    {"entity": "Los Angeles", "types": ["location"]}
  ]
}

"""
    
    # Add generation instructions
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])
    
    prompt += f"""

**MANDATORY Task:**
Generate a NEW movie review text similar to the examples above but with different content.
Use the following attributes for variation: {attributes_string}

**CRITICAL Variation Requirements:**
- MUST include entities from these types: {', '.join(entity_types)}
- Create diverse expressions and formats for each entity type
- Use clear, explicit language for entity identification
- Provide sufficient context for each entity
- Make entities easily distinguishable in the text

**MANDATORY Output Format:**
<start {attributes_string}>
{{
  "text": "your generated text here",
  "entities": [
    {{"entity": "entity name", "types": ["entity type"]}},
    ...
  ]
}}
<end>

CRITICAL: Generate ONLY ONE example in the specified JSON format.

<start {attributes_string}>
"""
    
    return prompt


def create_targeted_prompt_with_analysis(low_confidence_examples, entity_types, final_summary=None, **kwargs):
    """Create prompt with analysis integration and dynamic entity types"""
    
    # Filter attributes
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}
    
    # Build base prompt
    prompt = """CRITICAL: This is a PRODUCTION system for generating training data.

**Objective:**
Generate realistic movie review text passages that include clearly identified named entities. Focus on creating diverse examples based on domain analysis and provided templates.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled in the 'entities' list
- Follow the exact format shown in the examples below

**Entity Types to Focus On:**
"""
    
    # DYNAMIC ENTITY TYPES - using the entity_types variable
    for entity_type in entity_types:
        prompt += f"- {entity_type}: Entities of type {entity_type}\n"
    
    # ADD DOMAIN ANALYSIS INSIGHTS (NEW)
    if final_summary:
        prompt += f"""
**DOMAIN ANALYSIS INSIGHTS:**
Based on analysis of low-confidence examples, here are key insights to incorporate:

Domain Summary: {final_summary.get('domain_summary', 'N/A')}

Entity-Specific Insights:
"""
        for entity_type in entity_types:
            entity_data = final_summary.get('entity_summaries', {}).get(entity_type, {})
            if entity_data:
                prompt += f"""
{entity_type.upper()}:
- Position Issues: {entity_data.get('position_summary', 'None identified')}
- What Works Well: {entity_data.get('good_examples_summary', 'N/A')}
- What Fails: {entity_data.get('bad_examples_summary', 'N/A')}
- Needed Variations: {entity_data.get('variations_summary', 'N/A')}
"""
    
    prompt += """
**TEMPLATE EXAMPLES:**
Here are some real examples showing the expected format and entity types:

"""
    
    # Add low confidence examples as templates
    for i, example in enumerate(low_confidence_examples):
        text = " ".join(example['tokenized_text'])
        entities = []
        
        # Convert NER format to JSON entities
        for start, end, label in example['ner']:
            entity_text = " ".join(example['tokenized_text'][start:end+1])
            entities.append({
                "entity": entity_text,
                "types": [label]
            })
        
        prompt += f"""
Example {i+1}:
{{
  "text": "{text}",
  "entities": {json.dumps(entities, indent=2)}
}}
"""
    
    # Add generation instructions
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])
    
    prompt += f"""

**MANDATORY Task:**
Generate a NEW movie review text similar to the examples above but with different content.
Use the following attributes for variation: {attributes_string}
"""
    
    # Add analysis-based instructions if available
    if final_summary:
        prompt += f"""
IMPORTANT: Incorporate the domain analysis insights above to:
- Address the position/boundary issues identified
- Use patterns from "what works well" examples
- Avoid patterns from "what fails" examples  
- Include the needed variations identified in the analysis
"""
    
    prompt += f"""

**CRITICAL Variation Requirements:**
- MUST include entities from these types: {', '.join(entity_types)}
- Create diverse expressions and formats for each entity type
- Use clear, explicit language for entity identification
- Provide sufficient context for each entity
- Make entities easily distinguishable in the text

**MANDATORY Output Format:**
<start {attributes_string}>
{{
  "text": "your generated text here",
  "entities": [
    {{"entity": "entity name", "types": ["entity type"]}},
    ...
  ]
}}
<end>

CRITICAL: Generate ONLY ONE example in the specified JSON format.

<start {attributes_string}>
"""
    
    return prompt


def tokenize_text(text):
    """Tokenize the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def convert_synthetic_to_ner_format(synthetic_data):
    """Convert synthetic JSON data to NER format"""
    logger.info(f"Converting {len(synthetic_data)} synthetic examples to NER format...")
    
    all_examples = []
    conversion_errors = 0

    for i, dt in enumerate(synthetic_data):
        try:
            tokens = tokenize_text(dt['text'])
            ents = [(k["entity"], k["types"]) for k in dt['entities']]
        except Exception as e:
            logger.warning(f"Error processing synthetic example {i}: {e}")
            conversion_errors += 1
            continue

        spans = []
        for entity in ents:
            entity_tokens = tokenize_text(str(entity[0]))

            # Find the start and end indices of each entity in the tokenized text
            for j in range(len(tokens) - len(entity_tokens) + 1):
                if " ".join(tokens[j:j + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                    for el in entity[1]:
                        spans.append((j, j + len(entity_tokens) - 1, el.lower().replace('_', ' ')))

        # Append the tokenized text and its corresponding named entity recognition data
        all_examples.append({"tokenized_text": tokens, "ner": spans})

    logger.info(f"Conversion completed: {len(all_examples)} examples, {conversion_errors} errors")
    return all_examples


def validate_and_clean_ner_data(ner_data, valid_entity_types):
    """Simple validation and cleaning of NER data to prevent training crashes"""
    
    logger.info(f"Validating and cleaning {len(ner_data)} NER examples...")
    
    cleaned_data = []
    stats = {
        'examples_removed': 0,
        'entities_removed': 0,
        'out_of_bounds': 0,
        'invalid_order': 0,
        'invalid_types': 0,
        'empty_after_cleaning': 0
    }
    
    invalid_types_found = set()
    
    for i, example in enumerate(ner_data):
        try:
            tokenized_text = example['tokenized_text'] 
            ner = example['ner']
            text_len = len(tokenized_text)
            
            # Skip very short texts (likely errors)
            if text_len < 2:
                stats['examples_removed'] += 1
                continue
            
            cleaned_entities = []
            
            for entity in ner:
                # Check if entity has correct format [start, end, type]
                if not isinstance(entity, (list, tuple)) or len(entity) != 3:
                    stats['entities_removed'] += 1
                    continue
                    
                start, end, entity_type = entity
                
                # Check index types
                if not isinstance(start, int) or not isinstance(end, int):
                    stats['entities_removed'] += 1
                    continue
                
                # Check index order (start should not be greater than end)
                if start > end:
                    stats['invalid_order'] += 1
                    stats['entities_removed'] += 1
                    continue
                
                # Check index bounds (most critical - this was causing the crash)
                if start < 0 or end >= text_len:
                    stats['out_of_bounds'] += 1
                    stats['entities_removed'] += 1
                    continue
                
                # Check for extremely long spans (likely errors)
                if (end - start) > 15:  # More than 15 tokens is suspicious
                    stats['entities_removed'] += 1
                    continue
                
                # Check entity type validity
                if entity_type not in valid_entity_types:
                    invalid_types_found.add(entity_type)
                    stats['invalid_types'] += 1
                    stats['entities_removed'] += 1
                    continue
                
                # If we get here, entity is valid
                cleaned_entities.append([start, end, entity_type])
            
            # Only keep examples with at least one valid entity
            if len(cleaned_entities) > 0:
                cleaned_data.append({
                    "tokenized_text": tokenized_text,
                    "ner": cleaned_entities
                })
            else:
                stats['empty_after_cleaning'] += 1
                stats['examples_removed'] += 1
                
        except Exception as e:
            logger.warning(f"Error validating example {i}: {e}")
            stats['examples_removed'] += 1
            continue
    
    # Log cleaning results
    logger.info(f"Validation completed:")
    logger.info(f"  Original examples: {len(ner_data)}")
    logger.info(f"  Cleaned examples: {len(cleaned_data)}")
    logger.info(f"  Examples removed: {stats['examples_removed']}")
    logger.info(f"  Entities removed: {stats['entities_removed']}")
    
    if stats['out_of_bounds'] > 0:
        logger.info(f"  - Out of bounds indices: {stats['out_of_bounds']}")
    if stats['invalid_order'] > 0:
        logger.info(f"  - Invalid index order: {stats['invalid_order']}")
    if stats['invalid_types'] > 0:
        logger.info(f"  - Invalid entity types: {stats['invalid_types']}")
    if stats['empty_after_cleaning'] > 0:
        logger.info(f"  - Empty after cleaning: {stats['empty_after_cleaning']}")
        
    if invalid_types_found:
        logger.info(f"  Invalid types found: {sorted(invalid_types_found)}")
    
    # Calculate and log statistics (as requested)
    if cleaned_data:
        logger.info("="*50)
        logger.info("CLEANED DATA STATISTICS")
        logger.info("="*50)
        
        lengths = [len(d["tokenized_text"]) for d in cleaned_data]
        len_ner = [len(d["ner"]) for d in cleaned_data]
        unique_entities = [str(n[2]).lower() for d in cleaned_data for n in d["ner"]]
        
        logger.info(f"Avg num tokens: {sum(lengths) / len(lengths):.2f}")
        logger.info(f"Avg num of entities: {sum(len_ner) / len(len_ner):.2f}")
        logger.info(f"Total entities: {len(unique_entities)}")
        logger.info(f"Unique entity types: {len(set(unique_entities))}")
        
        # Top entity types
        entity_counts = Counter(unique_entities)
        top_types = entity_counts.most_common(10)
        logger.info(f"Top 10 entity types: {top_types}")
    
    return cleaned_data


def generate_synthetic_data_incremental(low_confidence_examples, entity_types, num_examples, num_samples_needed, final_summary=None):
    """Generate synthetic training data incrementally with analysis integration and validation.
    
    FIXED: Now properly handles the case where low_confidence_examples is empty (num_examples=0)
    """
    
    # Updated countries list - primarily English-speaking countries
    countries = ["usa", "uk", "australia", "canada", "ireland", "new zealand", "south africa", "india"]
    
    logger.info("="*60)
    logger.info("INCREMENTAL SYNTHETIC DATA GENERATION")
    logger.info("="*60)
    logger.info(f"Need {num_samples_needed} new synthetic examples")
    logger.info(f"Using entity types: {entity_types}")
    logger.info(f"Corrected examples available: {len(low_confidence_examples)}")
    logger.info(f"Using domain analysis: {'YES' if final_summary else 'NO'}")
    
    # NEW: Handle zero corrected examples case
    use_baseline_prompt = len(low_confidence_examples) == 0
    if use_baseline_prompt:
        logger.info("⚠️ ZERO CORRECTED EXAMPLES - Using baseline synthetic data generation")
    else:
        logger.info(f"✅ Using {len(low_confidence_examples)} corrected examples as templates")
    
    synthetic_outputs = []
    
    for i in tqdm(range(num_samples_needed), desc="Generating synthetic data"):
        # Random variation
        country = random.choice(countries)
        
        logger.debug(f"Generating example {i+1}/{num_samples_needed} with country: {country}")
        
        # Create appropriate prompt based on available corrected examples
        if use_baseline_prompt:
            # NEW: Use baseline prompt when no corrected examples available
            prompt = create_baseline_synthetic_prompt(
                entity_types,
                language="english",
                country=country,
                focus="movie reviews"
            )
        else:
            # Use targeted prompt with analysis integration when corrected examples available
            prompt = create_targeted_prompt_with_analysis(
                low_confidence_examples,
                entity_types,
                final_summary=final_summary,
                language="english",
                country=country,
                focus="movie reviews"
            )
        
        try:
            response = ollama.generate(
                model='mistral:latest',
                prompt=prompt,
                options={
                    'top_k': 100,
                    'top_p': 0.8,
                    'num_predict': 800,
                    'temperature': 0.7,
                    'stop': ['<end>']
                }
            )
            
            # Parse JSON from response
            response_text = response['response'].strip()
            
            js = json.loads(response_text)
            synthetic_outputs.append(js)
            logger.debug(f"✅ Example {i+1} generated successfully")
            
        except json.JSONDecodeError as e:
            logger.warning(f"❌ Failed to parse JSON for sample {i+1}: {e}")
            logger.debug(f"Response preview: {response_text[:200]}...")
            continue
        except Exception as e:
            logger.error(f"❌ Failed to generate sample {i+1}: {e}")
            continue
    
    logger.info("="*60)
    logger.info("INCREMENTAL GENERATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Successfully generated {len(synthetic_outputs)}/{num_samples_needed} new synthetic examples")
    logger.info(f"Prompt type used: {'BASELINE' if use_baseline_prompt else 'TEMPLATE-BASED'}")
    
    # Convert to NER format
    ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
    logger.info(f"Initial NER formatted examples: {len(ner_formatted_data)}")
    
    # VALIDATE AND CLEAN THE DATA (Simple validation after conversion)
    cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types)
    
    if len(cleaned_data) == 0:
        logger.error("❌ No valid examples remain after cleaning")
        return synthetic_outputs, []
    
    # Check if we lost too many examples
    loss_rate = (len(ner_formatted_data) - len(cleaned_data)) / len(ner_formatted_data)
    if loss_rate > 0.5:  # More than 50% loss
        logger.warning(f"⚠️ High data loss during cleaning: {loss_rate*100:.1f}% of examples removed")
    
    logger.info(f"Final cleaned examples: {len(cleaned_data)}")
    return synthetic_outputs, cleaned_data

# ===============================================================================
# OPTIMIZED PIPELINE MANAGEMENT WITH BATCH CACHING AND ZERO EXAMPLES SUPPORT
# ===============================================================================

def get_or_create_analysis(num_examples, low_confidence_examples, entity_types, batch_size=10, skip_analysis=False):
    """Get analysis from cache or create new analysis with batch caching optimization.
    
    FIXED: Now properly handles num_examples=0 case
    """
    
    # NEW: Handle zero examples case
    if num_examples == 0:
        logger.info(f"⏭️ Skipping analysis pipeline (num_examples=0 - no corrected examples to analyze)")
        return {'final_summary': None, 'combined_result': None, 'all_batch_results': None}
    
    if skip_analysis:
        logger.info(f"⏭️ Skipping analysis pipeline (skip_analysis=True)")
        return {'final_summary': None, 'combined_result': None, 'all_batch_results': None}
    
    cache_key = num_examples
    
    if cache_key in FINAL_SUMMARY_CACHE:
        logger.info(f"📋 Using cached final summary for {num_examples} examples")
        return FINAL_SUMMARY_CACHE[cache_key]
    
    logger.info(f"🔄 Creating new analysis for {num_examples} examples")
    
    # Process examples in batches WITH BATCH CACHING
    all_batch_results = analyze_domain_with_batch_caching(low_confidence_examples, entity_types, batch_size)
    
    if not all_batch_results:
        logger.error("❌❌❌ CRITICAL: Batch analysis failed after retries - STOPPING PIPELINE")
        return None
    
    # Combine all batch results
    combined_result = combine_all_batch_results(all_batch_results, entity_types)
    
    # Create final summary WITH RETRY LOGIC
    final_summary = create_final_summary_with_retry(combined_result)
    
    if not final_summary:
        logger.error("❌❌❌ CRITICAL: Final summary failed after retries - STOPPING PIPELINE")
        return None
    
    logger.info(f"📋 Analysis Summary: {final_summary.get('domain_summary', 'N/A')[:100]}...")
    
    # Cache the results at the final summary level
    FINAL_SUMMARY_CACHE[cache_key] = {
        'final_summary': final_summary,
        'combined_result': combined_result,
        'all_batch_results': all_batch_results
    }
    
    logger.info(f"💾 Cached final summary for {num_examples} examples")
    log_cache_status()
    
    return FINAL_SUMMARY_CACHE[cache_key]


def get_or_create_synthetic_data(num_examples, num_synthetic_needed, low_confidence_examples, entity_types, skip_analysis=False, final_summary=None):
    """Get synthetic data from cache or create incrementally.
    
    FIXED: Now properly handles num_examples=0 case
    """
    
    cache_key = num_examples
    
    # Initialize cache for this number of examples if not exists
    if cache_key not in SYNTHETIC_CACHE:
        SYNTHETIC_CACHE[cache_key] = []
    
    current_count = len(SYNTHETIC_CACHE[cache_key])
    
    if current_count >= num_synthetic_needed:
        logger.info(f"📦 Using cached synthetic data: {num_synthetic_needed}/{current_count} examples")
        return SYNTHETIC_CACHE[cache_key][:num_synthetic_needed]
    
    # Need to generate more synthetic data
    additional_needed = num_synthetic_needed - current_count
    logger.info(f"🔄 Need {additional_needed} more synthetic examples (have {current_count}, need {num_synthetic_needed})")
    
    # NEW: Handle zero corrected examples case
    if num_examples == 0:
        logger.info(f"⚠️ Generating synthetic data with ZERO corrected examples (baseline mode)")
        final_summary_to_use = None  # Never use analysis when no corrected examples
    elif skip_analysis:
        logger.info(f"⏭️ Generating synthetic data without analysis (skip_analysis=True)")
        final_summary_to_use = None
    else:
        logger.info(f"🧠 Generating synthetic data WITH domain analysis integration")
        final_summary_to_use = final_summary
    
    # Generate additional synthetic data WITH ANALYSIS INTEGRATION AND VALIDATION
    synthetic_json, synthetic_ner = generate_synthetic_data_incremental(
        low_confidence_examples, entity_types, num_examples, additional_needed, final_summary_to_use
    )
    
    # Check if we got valid data after cleaning
    if len(synthetic_ner) == 0:
        logger.error("❌ No valid synthetic data generated after cleaning")
        return []
    
    # Add to cache
    SYNTHETIC_CACHE[cache_key].extend(synthetic_ner)
    
    logger.info(f"💾 Updated synthetic cache: {len(SYNTHETIC_CACHE[cache_key])} total examples for {num_examples} corrected examples")
    
    return SYNTHETIC_CACHE[cache_key][:num_synthetic_needed]

# ===============================================================================
# TRAINING COMPONENTS
# ===============================================================================

class SimpleTrainingMonitor(TrainerCallback):
    """Simple training monitor with resource tracking"""
    
    def __init__(self, patience=10):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.steps = []
        self.eval_steps = []
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Resource tracking
        self.gpu_memory = []
        self.cpu_memory = []
        self.timestamps = []
        self.start_time = time.time()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                
                # Track resources
                current_time = (time.time() - self.start_time) / 60  # minutes
                self.timestamps.append(current_time)
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                    self.gpu_memory.append(gpu_mem)
                
                cpu_mem = psutil.virtual_memory().percent
                self.cpu_memory.append(cpu_mem)
                
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Every 50 steps
            torch.cuda.empty_cache()
            gc.collect()


def train_with_synthetic_data(base_model, synthetic_train_data, synthetic_val_data, device, num_steps=200):
    """Train model with synthetic data using LoRA"""
    
    logger.info("="*80)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("="*80)
    logger.info(f"Training steps: {num_steps}, Batch size: {BATCH_SIZE}")
    logger.info(f"Training examples: {len(synthetic_train_data)}, Validation examples: {len(synthetic_val_data)}")
    
    # Clean memory before loading model
    cleanup_memory()
    
    logger.info("Loading base model for training...")
    
    # Load fresh model instance
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192
    
    if hasattr(model.data_processor, 'transformer_tokenizer'):    
        model.data_processor.transformer_tokenizer.model_max_length = 8192
    
    # Get base parameter count
    base_total = sum(p.numel() for p in model.model.parameters())
    logger.info(f"Base Parameters: {base_total:,}")
    
    logger.info("Applying LoRA Configuration...")
    
    # LoRA config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "dense", "projection", "Wqkv", "Wo", "Wi",
            "query", "key", "value",
            "intermediate.dense", "output.dense",
            "span_rep_layer.span_rep_layer.project_start.3",
            "span_rep_layer.span_rep_layer.project_start.0",
            "span_rep_layer.span_rep_layer.project_end.3",
            "span_rep_layer.span_rep_layer.project_end.0",
            "span_rep_layer.span_rep_layer.out_project.3",
            "span_rep_layer.span_rep_layer.out_project.0",
            'prompt_rep_layer.3','prompt_rep_layer.0',
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS
    )
    
    # Apply LoRA
    model.model = get_peft_model(model.model, lora_config)
    
    # Get LoRA parameter count
    lora_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {lora_trainable:,} ({100*lora_trainable/base_total:.1f}% of original)")
    
    model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Training configuration
    training_config = {
        'num_steps': num_steps,
        'train_batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-4,
        'others_lr': 5e-4,
        'warmup_ratio': 0.1,
        'eval_steps': 50,
        'save_steps': 100,
        'logging_steps': 10,
        'patience': 10,
        'max_grad_norm': 1.0,
    }
    
    logger.info(f"Training configuration: {training_config}")
    
    # Setup data collator
    data_collator = DataCollator(
        model.config, 
        data_processor=model.data_processor, 
        prepare_labels=True
    )
    
    # Initialize training monitor
    monitor = SimpleTrainingMonitor(patience=training_config['patience'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/active_learning_lora",
        learning_rate=training_config['learning_rate'],
        weight_decay=0.01,
        others_lr=training_config['others_lr'],
        others_weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=training_config['warmup_ratio'],
        per_device_train_batch_size=training_config['train_batch_size'],
        per_device_eval_batch_size=training_config['train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        max_steps=training_config['num_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        
        focal_loss_alpha=0.75,      
        focal_loss_gamma=1.0,       
        
        eval_strategy="steps",
        eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=2,
        logging_steps=training_config['logging_steps'],
        seed=42,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
        
        fp16=False,
        bf16=False,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Clear cache before training
    cleanup_memory()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=synthetic_train_data,
        eval_dataset=synthetic_val_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        callbacks=[monitor],
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*80)
    logger.info(f"Training time: {training_time/60:.1f} minutes")
    logger.info(f"Best validation loss: {monitor.best_loss:.4f}")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Save the trained model
    model.model.save_pretrained("./models/active_learning_adapter")
    logger.info("Model saved to ./models/active_learning_adapter")
    
    return model, monitor

# ===============================================================================
# MAIN PIPELINE WITH PROPER TRAIN/TEST SEPARATION
# ===============================================================================

def run_active_learning_experiment_optimized(training_pool_results, test_data, entity_types, device, 
                                           num_corrected_examples=5, num_synthetic=20, training_steps=200, skip_analysis=False):
    """
    Optimized active learning pipeline with proper train/test separation
    
    NEW: training_pool_results comes from enhanced_evaluate on the TRAINING data (not test data)
    test_data is used ONLY for final evaluation, never for selecting examples
    """
    
    logger.info("="*80)
    logger.info("OPTIMIZED ACTIVE LEARNING EXPERIMENT WITH PROPER TRAIN/TEST SEPARATION")
    logger.info("="*80)
    logger.info(f"Parameters:")
    logger.info(f"  Corrected examples: {num_corrected_examples}")
    logger.info(f"  Synthetic examples: {num_synthetic}")
    logger.info(f"  Training steps: {training_steps}")
    logger.info(f"  Entity types: {entity_types}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Skip analysis: {skip_analysis}")
    
    # NEW: Handle zero corrected examples case
    if num_corrected_examples == 0:
        logger.info("⚠️ ZERO CORRECTED EXAMPLES MODE - Will skip analysis and use baseline synthetic generation")
        force_skip_analysis = True
    else:
        force_skip_analysis = skip_analysis
    
    # NEW: Handle (0,0) case - evaluate baseline model on test data for reference
    if num_corrected_examples == 0 and num_synthetic == 0:
        logger.info("🔄 (0,0) CASE: Evaluating baseline model on test data")
        
        # Load base model
        base_model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
        base_model.config.max_len = 8192
        
        if hasattr(base_model.data_processor, 'transformer_tokenizer'):    
            base_model.data_processor.transformer_tokenizer.model_max_length = 8192
            
        base_model.to(device)
        
        # Evaluate baseline model on test data
        baseline_test_results = enhanced_evaluate(
            model=base_model,
            data=test_data,
            entity_types=entity_types,
            threshold=0.5,
            batch_size=BATCH_SIZE
        )
        
        # Extract baseline metrics
        baseline_f1 = baseline_test_results['classification_report_df'][
            baseline_test_results['classification_report_df']['entity_type'] == 'micro_avg'
        ]['f1'].iloc[0]
        baseline_confidence = baseline_test_results['overall_metrics']['overall_confidence']
        
        # Clean up base model
        del base_model
        cleanup_memory()
        
        return {
            'baseline': {
                'f1': baseline_f1,
                'confidence': baseline_confidence
            },
            'after_training': {
                'f1': baseline_f1,  # Same as baseline since no training
                'confidence': baseline_confidence,
                'precision': baseline_test_results['classification_report_df'][
                    baseline_test_results['classification_report_df']['entity_type'] == 'micro_avg'
                ]['precision'].iloc[0],
                'recall': baseline_test_results['classification_report_df'][
                    baseline_test_results['classification_report_df']['entity_type'] == 'micro_avg'
                ]['recall'].iloc[0]
            },
            'improvement': {
                'f1': 0.0,  # No improvement since no training
                'confidence': 0.0
            },
            'experiment_params': {
                'num_corrected_examples': num_corrected_examples,
                'num_synthetic': num_synthetic,
                'training_steps': training_steps,
                'synthetic_generated': 0,
                'cache_used': False,
                'skip_analysis': True,
                'analysis_integrated': False,
                'zero_corrected_mode': True,
                'baseline_only': True
            },
            'low_confidence_examples': [],
            'training_monitor': None
        }
    
    # Step 1: Extract low confidence examples from TRAINING POOL evaluation results
    logger.info("="*60)
    logger.info("STEP 1: EXTRACTING LOW CONFIDENCE EXAMPLES FROM TRAINING POOL")
    logger.info("="*60)
    low_confidence_examples = get_lowest_score_examples_sorted(training_pool_results, n=num_corrected_examples)
    
    if num_corrected_examples == 0:
        logger.info("Zero corrected examples requested - proceeding with synthetic-only approach")
        low_confidence_ex_correct = []
    else:
        logger.info("Lowest confidence examples details:")
        low_confidence_ex_correct = []
        for i, ex in enumerate(low_confidence_examples):
            min_score = min(ex['scores']) if ex['scores'] else 0.0
            text_preview = ' '.join(ex['tokenized_text'][:10])
            logger.info(f"  {i+1}. Min score: {min_score:.3f}")
            logger.info(f"      Text: {text_preview}...")
            logger.info(f"      NER: {ex['ner']}")
            logger.info(f"      Predictions: {ex['predictions']}")
            logger.info(f"      Scores: {ex['scores']}")
            low_confidence_ex_correct.append({'tokenized_text': ex['tokenized_text'], 'ner': ex['ner']})
    
    logger.info(f"Corrected examples prepared: {len(low_confidence_ex_correct)}")

    # Step 2: Handle synthetic data with batch caching optimization (FIXED for zero case)
    synthetic_ner = []
    final_summary = None
    
    if num_synthetic > 0:
        logger.info("="*60)
        logger.info("STEP 2: OPTIMIZED SYNTHETIC DATA PIPELINE WITH BATCH CACHING")
        logger.info("="*60)
        
        if not force_skip_analysis:
            # Get or create analysis WITH BATCH CACHING (FIXED: handles num_corrected_examples=0)
            analysis_result = get_or_create_analysis(num_corrected_examples, low_confidence_examples, entity_types, skip_analysis=force_skip_analysis)
            
            if not analysis_result:
                logger.error("❌❌❌ CRITICAL: Analysis failed after retries - STOPPING EXPERIMENT")
                return None
            
            final_summary = analysis_result.get('final_summary')
            logger.info(f"✅ Analysis completed - final summary available: {'YES' if final_summary else 'NO'}")
        else:
            logger.info("⏭️ Skipping analysis pipeline as requested or due to zero corrected examples")
        
        # Get or create synthetic data with analysis integration and validation (FIXED for zero case)
        synthetic_ner = get_or_create_synthetic_data(
            num_corrected_examples, num_synthetic, low_confidence_examples, entity_types, skip_analysis=force_skip_analysis, final_summary=final_summary
        )
        
        if len(synthetic_ner) == 0:
            logger.error("❌ No valid synthetic data available after cleaning - STOPPING EXPERIMENT")
            return None
        
        logger.info(f"Synthetic data ready: {len(synthetic_ner)} examples")
        if synthetic_ner:
            sample = synthetic_ner[0]
            logger.info(f"Sample synthetic: {' '.join(sample['tokenized_text'][:10])}... | NER: {sample['ner']}")
    else:
        logger.info("="*60)
        logger.info("STEP 2: SKIPPING SYNTHETIC DATA PIPELINE (num_synthetic=0)")
        logger.info("="*60)
        if num_corrected_examples > 0:
            logger.info("Will train only on corrected examples")
        else:
            logger.error("❌ INVALID CONFIGURATION: Both corrected examples and synthetic examples are 0")
            logger.error("❌ Cannot train with zero training data - STOPPING EXPERIMENT")
            return None
    
    # Step 3: Prepare training data with validation (FIXED for zero corrected examples)
    logger.info("="*60)
    logger.info("STEP 3: PREPARING TRAINING DATA")
    logger.info("="*60)
    
    if num_synthetic > 0:
        # Split synthetic data for training
        split_point = int(0.8 * len(synthetic_ner))
        synthetic_train = synthetic_ner[:split_point]
        synthetic_val = synthetic_ner[split_point:]
        logger.info(f"Synthetic data split: {len(synthetic_train)} train, {len(synthetic_val)} val")
        
        # Ensure we have enough validation data
        if len(synthetic_val) == 0:
            logger.warning("No validation data - using last training example as validation")
            synthetic_val = [synthetic_train[-1]]
            synthetic_train = synthetic_train[:-1]
            
    else:
        # No synthetic data, use corrected examples for both train and val
        if num_corrected_examples == 0:
            # This should not happen due to earlier check, but just in case
            logger.error("❌ CRITICAL: No training data available (both corrected and synthetic = 0)")
            return None
            
        split_point = max(1, int(0.8 * len(low_confidence_ex_correct)))
        synthetic_train = low_confidence_ex_correct[:split_point]
        synthetic_val = low_confidence_ex_correct[split_point:] or low_confidence_ex_correct[:1]
        logger.info(f"No synthetic data - using corrected examples: {len(synthetic_train)} train, {len(synthetic_val)} val")

    # Add corrected low confidence examples to training data (FIXED: only if corrected examples exist)
    if num_synthetic > 0 and num_corrected_examples > 0:
        synthetic_train.extend(low_confidence_ex_correct)
        random.shuffle(synthetic_train)
        logger.info(f"After adding corrected examples: {len(synthetic_train)} train")
    elif num_corrected_examples > 0:
        logger.info(f"Training only on corrected examples: {len(synthetic_train)} train")
    else:
        logger.info(f"Training only on synthetic examples: {len(synthetic_train)} train")
    
    logger.info(f"Final training dataset: {len(synthetic_train)} examples")
    logger.info(f"Final validation dataset: {len(synthetic_val)} examples")
    
    # Final validation before training
    if len(synthetic_train) < 1:  # CHANGED: Allow training with just 1 example for testing
        logger.error(f"❌ Insufficient training data: {len(synthetic_train)} < 1 required")
        return None
        
    if len(synthetic_val) < 1:
        logger.error(f"❌ No validation data available")
        return None
    
    # Step 4: Train model
    logger.info("="*60)
    logger.info("STEP 4: TRAINING MODEL")
    logger.info("="*60)
    
    try:
        trained_model, training_monitor = train_with_synthetic_data(
            None, synthetic_train, synthetic_val, device, num_steps=training_steps
        )
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
    # Step 5: Evaluate improvement using enhanced evaluation ON TEST DATA
    logger.info("="*60)
    logger.info("STEP 5: EVALUATING IMPROVEMENT ON TEST DATA")
    logger.info("="*60)
    
    # Get baseline metrics from training pool results (but we need test baseline too)
    # Load base model for baseline test evaluation
    base_model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    base_model.config.max_len = 8192
    
    if hasattr(base_model.data_processor, 'transformer_tokenizer'):    
        base_model.data_processor.transformer_tokenizer.model_max_length = 8192
        
    base_model.to(device)
    
    # Get baseline performance on test data
    baseline_test_results = enhanced_evaluate(
        model=base_model,
        data=test_data,
        entity_types=entity_types,
        threshold=0.5,
        batch_size=BATCH_SIZE
    )
    
    baseline_f1 = baseline_test_results['classification_report_df'][
        baseline_test_results['classification_report_df']['entity_type'] == 'micro_avg'
    ]['f1'].iloc[0]
    baseline_confidence = baseline_test_results['overall_metrics']['overall_confidence']
    
    # Clean up base model
    del base_model
    cleanup_memory()
    
    logger.info("Baseline metrics (on test data):")
    logger.info(f"  F1: {baseline_f1:.3f}")
    logger.info(f"  Confidence: {baseline_confidence:.3f}")
    
    # Get new metrics using enhanced evaluation ON TEST DATA
    try:
        new_evaluation_results = enhanced_evaluate(trained_model, test_data, entity_types, threshold=0.5, batch_size=BATCH_SIZE)
        
        new_f1 = new_evaluation_results['classification_report_df'][
            new_evaluation_results['classification_report_df']['entity_type'] == 'micro_avg'
        ]['f1'].iloc[0]
        new_confidence = new_evaluation_results['overall_metrics']['overall_confidence']
        new_precision = new_evaluation_results['classification_report_df'][
            new_evaluation_results['classification_report_df']['entity_type'] == 'micro_avg'
        ]['precision'].iloc[0]
        new_recall = new_evaluation_results['classification_report_df'][
            new_evaluation_results['classification_report_df']['entity_type'] == 'micro_avg'
        ]['recall'].iloc[0]
        
    except Exception as e:
        logger.error(f"❌ Enhanced evaluation failed: {e}")
        return None
    
    logger.info("After training metrics (on test data):")
    logger.info(f"  F1: {new_f1:.3f}")
    logger.info(f"  Precision: {new_precision:.3f}")
    logger.info(f"  Recall: {new_recall:.3f}")
    logger.info(f"  Confidence: {new_confidence:.3f}")
    
    # Calculate improvement
    f1_improvement = new_f1 - baseline_f1
    confidence_improvement = new_confidence - baseline_confidence
    
    logger.info("Improvement summary:")
    logger.info(f"  F1: {f1_improvement:+.3f} ({f1_improvement*100:+.1f}%)")
    logger.info(f"  Confidence: {confidence_improvement:+.3f} ({confidence_improvement*100:+.1f}%)")
    
    # Clean up memory
    del trained_model
    cleanup_memory()
    
    logger.info("="*80)
    logger.info("OPTIMIZED EXPERIMENT WITH PROPER TRAIN/TEST SEPARATION COMPLETED!")
    logger.info("="*80)
    
    return {
        'baseline': {
            'f1': baseline_f1,
            'confidence': baseline_confidence
        },
        'after_training': {
            'f1': new_f1,
            'confidence': new_confidence,
            'precision': new_precision,
            'recall': new_recall
        },
        'improvement': {
            'f1': f1_improvement,
            'confidence': confidence_improvement
        },
        'experiment_params': {
            'num_corrected_examples': num_corrected_examples,
            'num_synthetic': num_synthetic,
            'training_steps': training_steps,
            'synthetic_generated': len(synthetic_ner),
            'cache_used': num_corrected_examples in FINAL_SUMMARY_CACHE,
            'skip_analysis': force_skip_analysis,
            'analysis_integrated': final_summary is not None,
            'zero_corrected_mode': num_corrected_examples == 0,
            'baseline_only': False
        },
        'low_confidence_examples': low_confidence_examples,
        'training_monitor': training_monitor
    }


def run_full_experiment_loop_optimized(training_pool_results, test_data, entity_types, device, skip_analysis=False):
    """Run the full experiment loop with proper train/test separation
    
    NEW: training_pool_results comes from enhanced_evaluate on the TRAINING data
    test_data is used ONLY for final evaluation
    """
    
    logger.info("="*100)
    logger.info("OPTIMIZED EXPERIMENT LOOP WITH PROPER TRAIN/TEST SEPARATION")
    logger.info("="*100)
    
    # FIXED: Experiment parameters - small test values for debugging, INCLUDING 0
    no_correct_examples = [0, 5, 10,25,50,100, 250,500 ]  # ADDED 0 for zero corrected examples
    no_generated_examples = [0, 5, 10,25,50,100, 250,500 ]  # SMALLER test values
    
    # Results storage
    final_f1 = []
    final_confidence = []
    iter_correct = []
    iter_generated = []
    cache_hits = []
    analysis_integrated = []
    zero_corrected_mode = []
    baseline_only = []
    
    total_experiments = len(no_correct_examples) * len(no_generated_examples)
    experiment_count = 0
    
    logger.info(f"Running {total_experiments} optimized experiments:")
    logger.info(f"Corrected examples: {no_correct_examples}")
    logger.info(f"Generated examples: {no_generated_examples}")
    logger.info(f"Entity types: {entity_types}")
    logger.info(f"Consistent batch size: {BATCH_SIZE}")
    logger.info(f"Skip analysis: {skip_analysis}")
    logger.info("Features enabled:")
    logger.info("  - PROPER TRAIN/TEST SEPARATION (fixed data leakage)")
    logger.info("  - Low confidence examples from TRAINING pool only")
    logger.info("  - Test data used ONLY for final evaluation")
    logger.info("  - BATCH-LEVEL CACHING (major optimization)")
    logger.info("  - ZERO CORRECTED EXAMPLES SUPPORT (baseline mode)")
    logger.info("  - Retry logic (3 attempts for analysis stages)")
    logger.info("  - Robust data validation and cleaning")
    logger.info("  - Analysis integration in synthetic generation")
    logger.info("  - Enhanced evaluation for all metrics")
    
    for i in no_correct_examples:
        logger.info(f"\n🔄 Starting experiment series for {i} corrected examples")
        
        for j in no_generated_examples:
            experiment_count += 1
            logger.info("="*60)
            logger.info(f"EXPERIMENT {experiment_count}/{total_experiments}")
            logger.info(f"Corrected examples: {i}, Generated examples: {j}")
            logger.info("="*60)
            
            # Log cache status before experiment
            log_cache_status()
            
            # Clean memory before each experiment
            cleanup_memory()
            
            try:
                experiment_results = run_active_learning_experiment_optimized(
                    training_pool_results=training_pool_results,
                    test_data=test_data,
                    entity_types=entity_types,
                    device=device,
                    num_corrected_examples=i,
                    num_synthetic=j,
                    training_steps=200,
                    skip_analysis=skip_analysis
                )
                
                if experiment_results:
                    iter_correct.append(i)
                    iter_generated.append(j)
                    final_f1.append(experiment_results['after_training']['f1'])
                    final_confidence.append(experiment_results['after_training']['confidence'])
                    cache_hits.append(experiment_results['experiment_params']['cache_used'])
                    analysis_integrated.append(experiment_results['experiment_params']['analysis_integrated'])
                    zero_corrected_mode.append(experiment_results['experiment_params']['zero_corrected_mode'])
                    baseline_only.append(experiment_results['experiment_params'].get('baseline_only', False))
                    
                    logger.info(f"✅ EXPERIMENT {experiment_count} COMPLETED SUCCESSFULLY!")
                    logger.info(f"   F1: {experiment_results['after_training']['f1']:.3f}")
                    logger.info(f"   Confidence: {experiment_results['after_training']['confidence']:.3f}")
                    logger.info(f"   F1 improvement: {experiment_results['improvement']['f1']:+.3f}")
                    logger.info(f"   Confidence improvement: {experiment_results['improvement']['confidence']:+.3f}")
                    logger.info(f"   Cache used: {experiment_results['experiment_params']['cache_used']}")
                    logger.info(f"   Analysis integrated: {experiment_results['experiment_params']['analysis_integrated']}")
                    logger.info(f"   Zero corrected mode: {experiment_results['experiment_params']['zero_corrected_mode']}")
                    logger.info(f"   Baseline only: {experiment_results['experiment_params'].get('baseline_only', False)}")
                else:
                    logger.error(f"❌ EXPERIMENT {experiment_count} FAILED!")
                    iter_correct.append(i)
                    iter_generated.append(j)
                    final_f1.append(np.nan)
                    final_confidence.append(np.nan)
                    cache_hits.append(False)
                    analysis_integrated.append(False)
                    zero_corrected_mode.append(i == 0)
                    baseline_only.append(False)
                    
            except Exception as e:
                logger.error(f"❌ EXPERIMENT {experiment_count} ERROR: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                iter_correct.append(i)
                iter_generated.append(j)
                final_f1.append(np.nan)
                final_confidence.append(np.nan)
                cache_hits.append(False)
                analysis_integrated.append(False)
                zero_corrected_mode.append(i == 0)
                baseline_only.append(False)
            
            # Clean memory after each experiment
            cleanup_memory()
    
    # Create final results dataframe
    exp_results = pd.DataFrame({
        'no_corrected_examples': iter_correct,
        'no_generated_examples': iter_generated,
        'final_f1': final_f1,
        'final_confidence': final_confidence,
        'cache_used': cache_hits,
        'analysis_integrated': analysis_integrated,
        'zero_corrected_mode': zero_corrected_mode,
        'baseline_only': baseline_only
    })
    
    logger.info("="*100)
    logger.info("ALL OPTIMIZED EXPERIMENTS COMPLETED!")
    logger.info("="*100)
    logger.info("Results summary:")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful experiments: {exp_results['final_f1'].notna().sum()}")
    logger.info(f"Failed experiments: {exp_results['final_f1'].isna().sum()}")
    logger.info(f"Cache hits: {exp_results['cache_used'].sum()}")
    logger.info(f"Analysis integrated: {exp_results['analysis_integrated'].sum()}")
    logger.info(f"Zero corrected mode experiments: {exp_results['zero_corrected_mode'].sum()}")
    logger.info(f"Baseline only experiments: {exp_results['baseline_only'].sum()}")
    
    # Final cache efficiency report
    logger.info("="*60)
    logger.info("CACHE EFFICIENCY REPORT")
    logger.info("="*60)
    logger.info(f"Batch analysis cache: {len(BATCH_ANALYSIS_CACHE)} unique batches")
    logger.info(f"Final summary cache: {len(FINAL_SUMMARY_CACHE)} summaries")
    logger.info(f"Synthetic data cache: {len(SYNTHETIC_CACHE)} datasets")
    
    # Log cache efficiency
    for key, data in SYNTHETIC_CACHE.items():
        logger.info(f"  {key} examples → {len(data)} synthetic examples cached")
    
    # Fix dataframe logging - log full results table properly
    logger.info("="*80)
    logger.info("DETAILED RESULTS TABLE:")
    logger.info("="*80)
    
    # Set pandas display options temporarily for full display
    with pd.option_context('display.max_rows', None, 
                          'display.max_columns', None, 
                          'display.width', None, 
                          'display.max_colwidth', None):
        
        # Convert dataframe to string and log line by line for better readability
        df_string = str(exp_results)
        for line in df_string.split('\n'):
            logger.info(line)
    
    logger.info("="*80)
    logger.info("END OF RESULTS TABLE")
    logger.info("="*80)
    
    return exp_results

# ===============================================================================
# MAIN EXECUTION WITH PROPER TRAIN/TEST SEPARATION
# ===============================================================================

def main():
    """Main execution function with proper train/test separation to avoid data leakage"""
    logger.info("="*100)
    logger.info("COMPLETE ACTIVE LEARNING PIPELINE WITH PROPER TRAIN/TEST SEPARATION")
    logger.info("="*100)
    
    # Load MIT movie data
    logger.info("Loading MIT movie Dataset...")
    data_path = "../data/raw/mit-movie"
    
    train_data, entity_types = load_mit_dataset(
        os.path.join(data_path, "train.json"),
        os.path.join(data_path, "labels.json"),
        "train"
    )
    
    test_data, _ = load_mit_dataset(
        os.path.join(data_path, "test.json"),
        os.path.join(data_path, "labels.json"),
        "test"
    )
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"   • Training samples: {len(train_data)}")
    logger.info(f"   • Test samples: {len(test_data)}")
    logger.info(f"   • Entity types: {entity_types}")
    
    # FIXED: Run evaluation on TRAINING POOL to get low confidence examples
    logger.info("="*80)
    logger.info("RUNNING EVALUATION ON TRAINING POOL FOR ACTIVE LEARNING")
    logger.info("="*80)
    logger.info("CRITICAL: This avoids data leakage by using training data for example selection")
    
    try:
        # Load base model
        logger.info("Loading base GLiNER model for training pool evaluation...")
        base_model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
        base_model.config.max_len = 8192
        
        if hasattr(base_model.data_processor, 'transformer_tokenizer'):    
            base_model.data_processor.transformer_tokenizer.model_max_length = 8192
            
        base_model.to(device)
        logger.info("Base model loaded and moved to device")
        
        # Run enhanced evaluation on TRAINING DATA to get low confidence examples
        training_pool_results = enhanced_evaluate(
            model=base_model,
            data=train_data,  # FIXED: Use training data, not test data
            entity_types=entity_types,
            threshold=0.5,
            batch_size=BATCH_SIZE
        )
        
        # Log training pool evaluation metrics
        training_f1 = training_pool_results['overall_metrics']['overall_f1']
        training_confidence = training_pool_results['overall_metrics']['overall_confidence']
        
        logger.info("="*60)
        logger.info("TRAINING POOL EVALUATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Training Pool F1: {training_f1:.3f}")
        logger.info(f"Training Pool Confidence: {training_confidence:.3f}")
        logger.info(f"Total predictions: {training_pool_results['overall_metrics']['total_predictions']}")
        logger.info(f"Available low confidence examples: {len(training_pool_results['all_predictions'])}")
        
        # Clean up base model memory
        del base_model
        cleanup_memory()
        
    except Exception as e:
        logger.error(f"❌ Training pool evaluation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    logger.info("Starting experiments with proper train/test separation...")
    logger.info("Mode 1: WITH domain analysis integration")
    exp_results_with_analysis = run_full_experiment_loop_optimized(
        training_pool_results, test_data, entity_types, device, skip_analysis=False
    )
    
    logger.info("Mode 2: WITHOUT domain analysis (baseline)")
    exp_results_without_analysis = run_full_experiment_loop_optimized(
        training_pool_results, test_data, entity_types, device, skip_analysis=True
    )
    
    logger.info("="*100)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info("="*100)
    
    # Save results
    with open('experiment_results_with_analysis_fixed.json', 'w') as f:
        json.dump(exp_results_with_analysis.to_dict(), f, indent=2)
    
    with open('experiment_results_without_analysis_fixed.json', 'w') as f:
        json.dump(exp_results_without_analysis.to_dict(), f, indent=2)
    
    logger.info("Results saved to:")
    logger.info("  - experiment_results_with_analysis_fixed.json")
    logger.info("  - experiment_results_without_analysis_fixed.json")
    
    # Final optimization report
    logger.info("="*80)
    logger.info("OPTIMIZATION IMPACT SUMMARY")
    logger.info("="*80)
    logger.info("Fixed data leakage issue:")
    logger.info(f"  - Training pool F1: {training_f1:.3f}")
    logger.info(f"  - Training pool Confidence: {training_confidence:.3f}")
    logger.info(f"  - Low confidence examples extracted from TRAINING data only")
    logger.info(f"  - Test data used ONLY for final evaluation")
    
    logger.info("Batch caching eliminated redundant LLM analysis calls:")
    logger.info(f"  - Unique batches analyzed: {len(BATCH_ANALYSIS_CACHE)}")
    unique_corrected_examples = [0, 5, 10]
    total_possible_batches = sum(len(BATCH_ANALYSIS_CACHE) for _ in unique_corrected_examples if _ > 0)
    if total_possible_batches > 0:
        cache_efficiency = 1 - (len(BATCH_ANALYSIS_CACHE) / total_possible_batches)
        logger.info(f"  - Estimated LLM call reduction: {cache_efficiency*100:.1f}%")
    
    logger.info("Zero corrected examples support allows testing baseline synthetic generation:")
    logger.info(f"  - Experiments with 0 corrected examples: {exp_results_with_analysis['zero_corrected_mode'].sum()}")
    logger.info(f"  - Baseline-only experiments (0,0): {exp_results_with_analysis['baseline_only'].sum()}")

if __name__ == "__main__":
    main()