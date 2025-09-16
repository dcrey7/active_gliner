"""
Main evaluation orchestrator - simplified version of your enhanced_evaluate function
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from .metrics import compare_entities, calculate_overall_metrics, generate_classification_report
import pandas as pd
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


def analyze_tp_fp_confidence(tp_scores_by_entity, fp_scores_by_entity, entity_types):
    """Separate confidence analysis for True Positives and False Positives"""

    bins = [(0, 25), (26, 50), (51, 75), (76, 100)]
    bin_labels = ['0-25%', '26-50%', '51-75%', '76-100%']

    # Initialize data structures
    tp_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}
    fp_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}

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

            example_confidence = float(sum(scores) / len(scores)) if scores else 0.0
        else:
            example_confidence = 0.0

        example_data = {
            'text': text,
            'predictions': pred_batch,
            'example_confidence': example_confidence
        }

        if ground_truths is not None:
            example_data['ground_truth'] = ground_truths[i]

        if example_confidence >= overall_confidence:
            high_confidence_examples.append(example_data)
        else:
            low_confidence_examples.append(example_data)

    return high_confidence_examples, low_confidence_examples



def enhanced_evaluate(model, data: List[Dict], entity_types: List[str], 
                     threshold: float = 0.5, batch_size: int = 16,
                     has_ground_truth: bool = True,
                     logger: Optional[logging.Logger] = None) -> Dict:
    """
    Enhanced evaluation exactly like your original function
    
    Args:
        model: GLiNER model
        data: List of examples with tokenized_text and ner (ground truth)
        entity_types: List of entity types to predict
        threshold: Prediction threshold
        batch_size: Batch size for inference
        logger: Optional logger
    
    Returns:
        Dictionary containing analysis results
    """
    if logger:
        logger.info("Running enhanced evaluation...")
    
    # Prepare data for model inference
    texts = []
    ground_truths = [] if has_ground_truth else None

    for example in data:
        text = " ".join(example["tokenized_text"])
        texts.append(text)
        if has_ground_truth and "ner" in example:
            ground_truths.append(example["ner"])
    
    if logger:
        logger.info(f"Processing {len(texts)} examples...")
    
    # Run model predictions
    with torch.no_grad():
        all_predictions = model.run(
            texts, entity_types, 
            flat_ner=True, threshold=threshold, batch_size=batch_size
        )
    
    # Process predictions to match ground truth format (if available)
    if has_ground_truth:
        processed_predictions = _process_predictions(all_predictions, data)
    else:
        processed_predictions = all_predictions

    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(all_predictions, ground_truths)

    # Confidence binning analysis (always available)
    confidence_bins_df = create_confidence_bins(all_predictions, entity_types)
    confidence_bins_styled = apply_gradient_styling(
        confidence_bins_df,
        "Confidence Distribution by Entity Type"
    )
    
    results = {
        'overall_metrics': overall_metrics,
        'confidence_bins': confidence_bins_styled,
        'all_predictions': []  # fill below depending on ground truth
    }

    if has_ground_truth and ground_truths:
        # Full analysis with ground truth
        entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        entity_prediction_scores = defaultdict(list)
        tp_scores_by_entity = defaultdict(list)
        fp_scores_by_entity = defaultdict(list)
        incorrect_examples = []
        corrected_examples = []

        if logger:
            logger.info("Analyzing errors with ground truth...")

        for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
            comparison = compare_entities(gt, pred)

            # Update entity statistics and collect scores
            for tp in comparison['true_positives']:
                entity_type = tp[2]
                entity_stats[entity_type]['tp'] += 1
                score = comparison['pred_scores'].get((tp[0], tp[1], tp[2]), 1.0)
                entity_prediction_scores[entity_type].append(score)
                tp_scores_by_entity[entity_type].append(score)

            for fp in comparison['false_positives']:
                entity_type = fp[2]
                entity_stats[entity_type]['fp'] += 1
                score = comparison['pred_scores'].get((fp[0], fp[1], fp[2]), 1.0)
                entity_prediction_scores[entity_type].append(score)
                fp_scores_by_entity[entity_type].append(score)

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

        # Build all_predictions payload with ground truth info
        full_predictions_gt = []
        for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
            full_predictions_gt.append({
                "tokenized_text": data[i]["tokenized_text"],
                "ner": gt,
                "predictions": [[p[0], p[1], p[2]] for p in pred],
                "scores": [p[4] for p in pred]
            })

        # Generate classification report (raw + styled)
        classification_report = generate_classification_report(entity_stats, entity_prediction_scores)
        classification_report_styled = apply_gradient_styling(
            classification_report,
            "Enhanced Classification Report with Confidence"
        )

        # TP/FP confidence analysis
        tp_conf_df, fp_conf_df = analyze_tp_fp_confidence(
            tp_scores_by_entity, fp_scores_by_entity, entity_types
        )
        tp_styled = apply_gradient_styling(tp_conf_df, "True Positives Confidence Distribution")
        fp_styled = apply_gradient_styling(fp_conf_df, "False Positives Confidence Distribution")

        # Update overall metrics with accuracy calculations
        total_examples = len(data)
        correct_examples = total_examples - len(incorrect_examples)
        total_tp = classification_report[classification_report['entity_type'] == 'micro_avg']['tp'].iloc[0]
        total_entities_should_exist = classification_report[classification_report['entity_type'] == 'micro_avg']['support'].iloc[0]
        overall_f1 = classification_report[classification_report['entity_type'] == 'micro_avg']['f1'].iloc[0]

        overall_metrics.update({
            'overall_f1': overall_f1,
            'overall_f1_pct': overall_f1 * 100,
            'total_examples': total_examples,
            'incorrect_examples': incorrect_examples,  # stored for convenience
            'correct_examples': correct_examples,
            'example_level_accuracy': correct_examples / total_examples if total_examples else 0.0,
            'example_level_accuracy_pct': (correct_examples / total_examples) * 100 if total_examples else 0.0,
            'entity_level_accuracy': (total_tp / total_entities_should_exist) if total_entities_should_exist else 0.0,
            'entity_level_accuracy_pct': ((total_tp / total_entities_should_exist) * 100) if total_entities_should_exist else 0.0,
        })

        results.update({
            'overall_metrics': overall_metrics,
            'classification_report': classification_report_styled,
            'classification_report_df': classification_report,
            'all_predictions': full_predictions_gt,
            'tp_confidence_analysis': tp_styled,
            'fp_confidence_analysis': fp_styled,
            'incorrect_examples': incorrect_examples,
            'corrected_examples': corrected_examples,
        })
    else:
        # No ground truth: build all_predictions from raw predictions
        full_predictions = []
        for i, pred in enumerate(processed_predictions):
            full_predictions.append({
                "tokenized_text": data[i]["tokenized_text"],
                "predictions": [[p['start'], p['end'], p['label']] for p in pred],
                "scores": [p['score'] for p in pred]
            })

        # Separate examples into high/low confidence groups
        high_conf_examples, low_conf_examples = separate_examples_by_confidence(
            all_predictions, texts, overall_metrics['overall_confidence']
        )

        results.update({
            'overall_metrics': overall_metrics,
            'all_predictions': full_predictions,
            'high_confidence_examples': high_conf_examples,
            'low_confidence_examples': low_conf_examples,
        })

    return results


def _process_predictions(all_predictions: List, data: List[Dict]) -> List[List]:
    """
    Process predictions to match ground truth format (internal helper)
    
    Args:
        all_predictions: Raw predictions from model.run()
        data: Original data with tokenized text
        
    Returns:
        List of processed predictions
    """
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
    
    return processed_predictions


def evaluate_and_extract_metrics(model, data: List[Dict], entity_types: List[str],
                                threshold: float = 0.5, batch_size: int = 16,
                                has_ground_truth: bool = True,
                                logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Simplified evaluation that just returns key metrics (for quick evaluation)
    
    Args:
        model: GLiNER model
        data: List of examples
        entity_types: List of entity types  
        threshold: Prediction threshold
        batch_size: Batch size
        logger: Optional logger
        
    Returns:
        Dictionary with key metrics (f1, precision, recall, confidence)
    """
    results = enhanced_evaluate(
        model, data, entity_types,
        threshold=threshold, batch_size=batch_size,
        has_ground_truth=has_ground_truth, logger=logger
    )

    confidence = results['overall_metrics']['overall_confidence']

    if not has_ground_truth:
        # Without ground truth, only confidence is meaningful
        return {
            'confidence': confidence
        }

    # Extract key metrics (requires ground truth)
    f1 = results['classification_report_df'][
        results['classification_report_df']['entity_type'] == 'micro_avg'
    ]['f1'].iloc[0]

    precision = results['classification_report_df'][
        results['classification_report_df']['entity_type'] == 'micro_avg'
    ]['precision'].iloc[0]

    recall = results['classification_report_df'][
        results['classification_report_df']['entity_type'] == 'micro_avg'
    ]['recall'].iloc[0]

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confidence': confidence
    }
