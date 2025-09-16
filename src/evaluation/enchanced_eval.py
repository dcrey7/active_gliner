import torch
import numpy as np
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from .helper import apply_gradient_styling, display_results




def create_confidence_bins(predictions, entity_types):
    """Create confidence binning analysis for all predictions"""
    
    # Define confidence bins
    bins = [(0, 25), (26, 50), (51, 75), (76, 100)]
    bin_labels = ['0-25%', '26-50%', '51-75%', '76-100%']
    
    # Initialize data structure
    bin_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}
    
    # Process all predictions - expecting dict format from model.run()
    for pred_batch in predictions:
        for pred in pred_batch:
            entity_type = pred['label'].lower()
            confidence = pred['score'] * 100  # Convert to percentage
            
            # Find appropriate bin
            for i, (min_conf, max_conf) in enumerate(bins):
                if min_conf <= confidence <= max_conf:
                    if entity_type in bin_data[bin_labels[i]]:
                        bin_data[bin_labels[i]][entity_type] += 1
                    break
    
    # Convert to DataFrame
    df = pd.DataFrame(bin_data).T
    df.index.name = 'Confidence Range'
    
    return df


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

def analyze_tp_fp_confidence(tp_scores_by_entity, fp_scores_by_entity, entity_types):
    """Separate confidence analysis for True Positives and False Positives"""
    
    bins = [(0, 25), (26, 50), (51, 75), (76, 100)]
    bin_labels = ['0-25%', '26-50%', '51-75%', '76-100%']
    
    # Initialize data structures
    tp_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}
    fp_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}
    batch_size
    # Process TP scores
    for entity_type, scores in tp_scores_by_entity.items():
        for score in scores:
            confidence = score * 100  # Convert to percentage
            # Find appropriate bin
            for i, (min_conf, max_conf) in enumerate(bins):
                if min_conf <= confidence <= max_conf:
                    tp_data[bin_labels[i]][entity_type] += 1
                    break
    
    # Process FP scores  
    for entity_type, scores in fp_scores_by_entity.items():
        for score in scores:
            confidence = score * 100  # Convert to percentage
            # Find appropriate bin
            for i, (min_conf, max_conf) in enumerate(bins):
                if min_conf <= confidence <= max_conf:
                    fp_data[bin_labels[i]][entity_type] += 1
                    break
    
    # Convert to DataFrames
    tp_df = pd.DataFrame(tp_data).T
    fp_df = pd.DataFrame(fp_data).T
    
    tp_df.index.name = 'Confidence Range'
    fp_df.index.name = 'Confidence Range'
    
    return tp_df, fp_df

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

def separate_examples_by_confidence(predictions, texts, overall_confidence, ground_truths=None):
    """Separate examples into high/low confidence groups"""
    
    high_confidence_examples = []
    low_confidence_examples = []
    
    for i, (pred_batch, text) in enumerate(zip(predictions, texts)):
        # Calculate example-level confidence (average of all predictions in this example)
        if pred_batch:
            scores = []
            for pred in pred_batch:
                # Raw predictions are in dictionary format
                if 'score' in pred:
                    scores.append(pred['score'])
            
            example_confidence = np.mean(scores) if scores else 0.0
        else:
            example_confidence = 0.0
        
        example_data = {
            'text': text,
            'predictions': pred_batch,
            'example_confidence': example_confidence
        }
        
        if ground_truths:
            example_data['ground_truth'] = ground_truths[i]
        
        if example_confidence >= overall_confidence:
            high_confidence_examples.append(example_data)
        else:
            low_confidence_examples.append(example_data)
    
    return high_confidence_examples, low_confidence_examples

def enhanced_evaluate(model, data, entity_types, threshold=0.5, batch_size=16, has_ground_truth=True):
    """
    Enhanced evaluation with modular analysis
    
    Args:
        model: GLiNER model
        data: List of examples with tokenized_text and optionally ner
        entity_types: List of entity types to predict
        threshold: Prediction threshold
        batch_size: Batch size for inference
        has_ground_truth: Boolean - True if ground truth available, False otherwise
    
    Returns:
        Dictionary containing analysis results and styled dataframes
    """
    
    print("Running enhanced evaluation...")
    
    # Prepare data for model inference
    texts = []
    ground_truths = [] if has_ground_truth else None
    
    for example in data:
        text = " ".join(example["tokenized_text"])
        texts.append(text)
        if has_ground_truth and "ner" in example:
            ground_truths.append(example["ner"])
    
    print(f"Processing {len(texts)} examples...")
    
    # Run model predictions
    with torch.no_grad():
        all_predictions = model.run(
            texts, entity_types, 
            flat_ner=True, threshold=threshold, batch_size=batch_size
        )
    
    # Process predictions to match ground truth format (if available)
    processed_predictions = []
    for i, predictions in enumerate(all_predictions):
        if has_ground_truth:
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
        else:
            processed_predictions.append(predictions)
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(all_predictions, ground_truths)
    
    # Create confidence binning analysis
    confidence_bins_df = create_confidence_bins(all_predictions, entity_types)
    confidence_bins_styled = apply_gradient_styling(
        confidence_bins_df, 
        "Confidence Distribution by Entity Type"
    )

    # FIXED: Handle different prediction formats for all_predictions creation
    full_predictions_gt = []
    for i, pred in enumerate(processed_predictions):
        if has_ground_truth:
            # When has_ground_truth=True, predictions are in list format: [start, end, label, text, score]
            full_predictions_gt.append({
                "tokenized_text": data[i]["tokenized_text"],
                "predictions": [[p[0], p[1], p[2]] for p in pred],
                "scores": [p[4] for p in pred]
            })
        else:
            # When has_ground_truth=False, predictions are in dict format: {'start': ..., 'end': ..., 'label': ..., 'score': ...}
            full_predictions_gt.append({
                "tokenized_text": data[i]["tokenized_text"],
                "predictions": [[p['start'], p['end'], p['label']] for p in pred],
                "scores": [p['score'] for p in pred]
            })
    
    results = {
        'overall_metrics': overall_metrics,
        'confidence_bins': confidence_bins_styled,
        'all_predictions': full_predictions_gt
    }
    
    if has_ground_truth and ground_truths:
        # Full analysis with ground truth
        entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        entity_prediction_scores = defaultdict(list)
        tp_scores_by_entity = defaultdict(list)  # NEW: Track TP scores separately
        fp_scores_by_entity = defaultdict(list)  # NEW: Track FP scores separately
        incorrect_examples = []
        corrected_examples = []
        
        print("Analyzing errors with ground truth...")
        
        for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
            comparison = compare_entities(gt, pred)

            # Update entity statistics and collect scores separately
            for tp in comparison['true_positives']:
                entity_type = tp[2]
                entity_stats[entity_type]['tp'] += 1
                score = comparison['pred_scores'].get((tp[0], tp[1], tp[2]), 1.0)
                entity_prediction_scores[entity_type].append(score)
                tp_scores_by_entity[entity_type].append(score)  # NEW: Track TP separately
                
            for fp in comparison['false_positives']:
                entity_type = fp[2]
                entity_stats[entity_type]['fp'] += 1
                score = comparison['pred_scores'].get((fp[0], fp[1], fp[2]), 1.0)
                entity_prediction_scores[entity_type].append(score)
                fp_scores_by_entity[entity_type].append(score)  # NEW: Track FP separately
                
            for fn in comparison['false_negatives']:
                entity_stats[fn[2]]['fn'] += 1
            
            # Collect incorrect examples
            has_errors = len(comparison['false_positives']) > 0 or len(comparison['false_negatives']) > 0
            if has_errors:
                incorrect_examples.append({
                    "tokenized_text": data[i]["tokenized_text"],
                    "ner": gt,
                    "predictions": [[p[0], p[1], p[2]] for p in pred],
                    "scores": [p[4] for p in pred],
                    "errors": {
                        "false_negatives": [[fn[0], fn[1], fn[2]] for fn in comparison['false_negatives']],
                        "false_positives": [[fp[0], fp[1], fp[2]] for fp in comparison['false_positives']]
                    }
                })
                
                corrected_examples.append({
                    "tokenized_text": data[i]["tokenized_text"],
                    "ner": gt
                })
        
        # Update all_predictions with ground truth info
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
        classification_report_styled = apply_gradient_styling(
            classification_report, 
            "Enhanced Classification Report with Confidence"
        )
        
        # TP/FP confidence analysis
        tp_confidence_df, fp_confidence_df = analyze_tp_fp_confidence(
            tp_scores_by_entity, fp_scores_by_entity, entity_types
        )
        tp_styled = apply_gradient_styling(tp_confidence_df, "True Positives Confidence Distribution")
        fp_styled = apply_gradient_styling(fp_confidence_df, "False Positives Confidence Distribution")
        
        results.update({
            'classification_report': classification_report_styled,
            'classification_report_df': classification_report,  # Store raw dataframe for F1 extraction
            'all_predictions': full_predictions_gt,
            'tp_confidence_analysis': tp_styled,
            'fp_confidence_analysis': fp_styled,
            'incorrect_examples': incorrect_examples,
            'corrected_examples': corrected_examples,
        })
        
        # Update overall metrics with accuracy calculations
        total_examples = len(data)
        correct_examples = total_examples - len(incorrect_examples)
        total_tp = classification_report[classification_report['entity_type'] == 'micro_avg']['tp'].iloc[0]
        total_entities_should_exist = classification_report[classification_report['entity_type'] == 'micro_avg']['support'].iloc[0]
        overall_f1=results['classification_report_df'][results['classification_report_df']['entity_type'] == 'micro_avg']['f1'].iloc[0]
        
        results['overall_metrics'].update({
            'overall_f1': overall_f1,
            'overall_f1_pct': overall_f1 * 100,
            'total_examples': total_examples,
            'incorrect_examples': incorrect_examples,
            'correct_examples': correct_examples,
            'example_level_accuracy': correct_examples / total_examples,
            'example_level_accuracy_pct': (correct_examples / total_examples) * 100,
            'entity_level_accuracy': total_tp / total_entities_should_exist,
            'entity_level_accuracy_pct': (total_tp / total_entities_should_exist) * 100
        })
    
    else:
        # No ground truth analysis - separate by confidence threshold
        high_conf_examples, low_conf_examples = separate_examples_by_confidence(
            all_predictions, texts, overall_metrics['overall_confidence']
        )
        
        results.update({
            'high_confidence_examples': high_conf_examples,
            'low_confidence_examples': low_conf_examples,
        })
    
    return results
