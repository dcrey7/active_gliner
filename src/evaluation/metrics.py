"""
Evaluation metrics - extracted from your original evaluation functions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import defaultdict


def compare_entities(ground_truth: List, predictions: List) -> Dict:
    """
    Compare ground truth and predictions exactly like your original function
    
    Args:
        ground_truth: List of ground truth entities
        predictions: List of predicted entities
        
    Returns:
        Dictionary with comparison results
    """
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


def calculate_overall_metrics(raw_predictions: List, ground_truths: List = None) -> Dict:
    """
    Calculate overall accuracy and confidence metrics exactly like your original function
    
    Args:
        raw_predictions: Raw predictions from model.run()
        ground_truths: Optional ground truth data
        
    Returns:
        Dictionary with overall metrics
    """
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


def generate_classification_report(entity_stats: Dict, entity_prediction_scores: Dict) -> pd.DataFrame:
    """
    Generate enhanced classification report exactly like your original function
    
    Args:
        entity_stats: Dictionary of entity statistics (tp, fp, fn counts)
        entity_prediction_scores: Dictionary of prediction scores by entity type
        
    Returns:
        DataFrame with classification report
    """
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


def extract_f1_from_report(classification_report_df: pd.DataFrame) -> float:
    """
    Extract micro-average F1 score from classification report
    
    Args:
        classification_report_df: DataFrame with classification report
        
    Returns:
        Micro-average F1 score
    """
    micro_row = classification_report_df[classification_report_df['entity_type'] == 'micro_avg']
    if len(micro_row) > 0:
        return micro_row['f1'].iloc[0]
    return 0.0


def extract_precision_recall_from_report(classification_report_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Extract micro-average precision and recall from classification report
    
    Args:
        classification_report_df: DataFrame with classification report
        
    Returns:
        Tuple of (precision, recall)
    """
    micro_row = classification_report_df[classification_report_df['entity_type'] == 'micro_avg']
    if len(micro_row) > 0:
        precision = micro_row['precision'].iloc[0]
        recall = micro_row['recall'].iloc[0]
        return precision, recall
    return 0.0, 0.0
