"""
Shared metric helpers for NER evaluation

Contains:
- Core NER metrics (F1, Precision, Recall) - works for both GLiNER and LLM
- Confidence analysis - GLiNER only
- Entity comparison and statistics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
from .helper import apply_gradient_styling


def compare_entities(ground_truth: List, predictions: List) -> Dict:
    """
    Compare ground truth and predictions to identify TP, FP, FN
    Works for both GLiNER and LLM predictions

    Args:
        ground_truth: List of ground truth spans [(start, end, type), ...]
        predictions: List of predicted spans [(start, end, type, text, score), ...]

    Returns:
        Dictionary with:
        - 'true_positives': List of matching spans
        - 'false_positives': List of incorrect predictions
        - 'false_negatives': List of missed ground truth spans
        - 'pred_scores': Dict mapping spans to confidence scores
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
        # Extract score if available (GLiNER has it, LLM might not)
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


def calculate_ner_metrics(processed_predictions: List[List], ground_truths: List[List],
                         data: List[Dict], entity_types: List[str]) -> Dict:
    """
    Calculate core NER metrics (F1, Precision, Recall, Confusion Matrix)
    Works for BOTH GLiNER and LLM - NO confidence analysis

    Args:
        processed_predictions: List of predictions per example (token-level spans)
        ground_truths: List of ground truth spans per example
        data: Original data (for creating output)
        entity_types: List of entity types

    Returns:
        Dictionary with:
        - 'classification_report': Styled DataFrame with F1/P/R per entity
        - 'classification_report_df': Raw DataFrame
        - 'entity_stats': Dict with TP/FP/FN counts per entity
        - 'entity_prediction_scores': Dict with scores per entity (for confidence if available)
        - 'incorrect_examples': List of examples with errors
        - 'corrected_examples': List of corrected examples
        - 'overall_f1': Overall F1 score
        - 'total_examples': Total number of examples
        - 'correct_examples': Number of examples without errors
    """
    entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    entity_prediction_scores = defaultdict(list)
    tp_scores_by_entity = defaultdict(list)
    fp_scores_by_entity = defaultdict(list)
    incorrect_examples = []
    corrected_examples = []

    # Analyze each example
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
                "scores": [p[4] if len(p) > 4 else 1.0 for p in pred],
                "errors": {
                    "false_negatives": [[fn[0], fn[1], fn[2]] for fn in comparison['false_negatives']],
                    "false_positives": [[fp[0], fp[1], fp[2]] for fp in comparison['false_positives']]
                }
            })

            corrected_examples.append({
                "tokenized_text": data[i]["tokenized_text"],
                "ner": gt
            })

    # Generate classification report
    classification_report = _generate_classification_report(entity_stats, entity_prediction_scores)
    classification_report_styled = apply_gradient_styling(
        classification_report,
        "Enhanced Classification Report with Confidence"
    )

    # Calculate summary metrics
    total_examples = len(data)
    correct_examples = total_examples - len(incorrect_examples)
    total_tp = classification_report[classification_report['entity_type'] == 'micro_avg']['tp'].iloc[0]
    total_entities_should_exist = classification_report[classification_report['entity_type'] == 'micro_avg']['support'].iloc[0]
    overall_f1 = classification_report[classification_report['entity_type'] == 'micro_avg']['f1'].iloc[0]

    return {
        'classification_report': classification_report_styled,
        'classification_report_df': classification_report,
        'entity_stats': entity_stats,
        'entity_prediction_scores': entity_prediction_scores,
        'tp_scores_by_entity': tp_scores_by_entity,
        'fp_scores_by_entity': fp_scores_by_entity,
        'incorrect_examples': incorrect_examples,
        'corrected_examples': corrected_examples,
        'overall_f1': overall_f1,
        'overall_f1_pct': overall_f1 * 100,
        'total_examples': total_examples,
        'correct_examples': correct_examples,
        'example_level_accuracy': correct_examples / total_examples if total_examples > 0 else 0.0,
        'example_level_accuracy_pct': (correct_examples / total_examples) * 100 if total_examples > 0 else 0.0,
        'entity_level_accuracy': total_tp / total_entities_should_exist if total_entities_should_exist > 0 else 0.0,
        'entity_level_accuracy_pct': (total_tp / total_entities_should_exist) * 100 if total_entities_should_exist > 0 else 0.0,
    }


def _generate_classification_report(entity_stats: Dict, entity_prediction_scores: Dict) -> pd.DataFrame:
    """
    Generate classification report with F1, Precision, Recall per entity
    Internal helper function

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
    macro_precision = np.mean([row['precision'] for row in df_data]) if df_data else 0.0
    macro_recall = np.mean([row['recall'] for row in df_data]) if df_data else 0.0
    macro_f1 = np.mean([row['f1'] for row in df_data]) if df_data else 0.0

    # Confidence averages
    micro_avg_confidence = np.mean(all_prediction_scores) if all_prediction_scores else 0.0
    macro_avg_confidence = np.mean([row['avg_prediction_confidence'] for row in df_data if row['avg_prediction_confidence'] > 0]) if df_data else 0.0

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


def analyze_confidence_metrics(raw_predictions: List, entity_types: List[str],
                               tp_scores_by_entity: Dict = None,
                               fp_scores_by_entity: Dict = None) -> Dict:
    """
    Analyze confidence metrics for GLiNER predictions
    ONLY works with predictions that have 'score' field

    Args:
        raw_predictions: Raw predictions from model (list of lists of dicts with 'score' field)
        entity_types: List of entity types
        tp_scores_by_entity: Optional dict of TP scores by entity type
        fp_scores_by_entity: Optional dict of FP scores by entity type

    Returns:
        Dictionary with:
        - 'overall_confidence': Average confidence (0.0-1.0)
        - 'overall_confidence_pct': Average confidence percentage
        - 'total_predictions': Total number of predictions
        - 'confidence_bins': Styled DataFrame with confidence distribution
        - 'tp_confidence_analysis': Styled DataFrame (if TP scores provided)
        - 'fp_confidence_analysis': Styled DataFrame (if FP scores provided)
    """
    # Calculate overall confidence
    all_scores = []
    total_predictions = 0

    for pred_batch in raw_predictions:
        for pred in pred_batch:
            if 'score' in pred:
                all_scores.append(pred['score'])
            total_predictions += 1

    overall_confidence = np.mean(all_scores) if all_scores else 0.0

    # Create confidence bins
    confidence_bins_df = _create_confidence_bins(raw_predictions, entity_types)
    confidence_bins_styled = apply_gradient_styling(
        confidence_bins_df,
        "Confidence Distribution by Entity Type"
    )

    result = {
        'overall_confidence': overall_confidence,
        'overall_confidence_pct': overall_confidence * 100,
        'total_predictions': total_predictions,
        'confidence_bins': confidence_bins_styled,
    }

    # Add TP/FP confidence analysis if provided
    if tp_scores_by_entity is not None and fp_scores_by_entity is not None:
        tp_conf_df, fp_conf_df = _analyze_tp_fp_confidence(
            tp_scores_by_entity, fp_scores_by_entity, entity_types
        )
        result['tp_confidence_analysis'] = apply_gradient_styling(
            tp_conf_df, "True Positives Confidence Distribution"
        )
        result['fp_confidence_analysis'] = apply_gradient_styling(
            fp_conf_df, "False Positives Confidence Distribution"
        )

    return result


def _create_confidence_bins(predictions: List, entity_types: List[str]) -> pd.DataFrame:
    """
    Create confidence binning analysis
    Internal helper function

    Args:
        predictions: List of prediction batches (dict format with 'score' field)
        entity_types: List of entity types

    Returns:
        DataFrame with confidence bins
    """
    # Define confidence bins
    bins = [(0, 25), (26, 50), (51, 75), (76, 100)]
    bin_labels = ['0-25%', '26-50%', '51-75%', '76-100%']

    # Initialize data structure
    bin_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}

    # Process all predictions
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


def _analyze_tp_fp_confidence(tp_scores_by_entity: Dict, fp_scores_by_entity: Dict,
                               entity_types: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate confidence analysis for True Positives and False Positives
    Internal helper function

    Args:
        tp_scores_by_entity: Dict of TP scores by entity type
        fp_scores_by_entity: Dict of FP scores by entity type
        entity_types: List of entity types

    Returns:
        Tuple of (tp_dataframe, fp_dataframe)
    """
    bins = [(0, 25), (26, 50), (51, 75), (76, 100)]
    bin_labels = ['0-25%', '26-50%', '51-75%', '76-100%']

    # Initialize data structures
    tp_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}
    fp_data = {label: {entity: 0 for entity in entity_types} for label in bin_labels}

    # Process TP scores
    for entity_type, scores in tp_scores_by_entity.items():
        for score in scores:
            confidence = score * 100  # Convert to percentage
            for i, (min_conf, max_conf) in enumerate(bins):
                if min_conf <= confidence <= max_conf:
                    tp_data[bin_labels[i]][entity_type] += 1
                    break

    # Process FP scores
    for entity_type, scores in fp_scores_by_entity.items():
        for score in scores:
            confidence = score * 100  # Convert to percentage
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


def separate_examples_by_confidence(predictions: List, texts: List[str],
                                   overall_confidence: float,
                                   ground_truths: List = None) -> Tuple[List, List]:
    """
    Separate examples into high/low confidence groups
    Used when no ground truth available

    Args:
        predictions: Raw predictions (list of lists of dicts)
        texts: List of text strings
        overall_confidence: Threshold for separating (overall average confidence)
        ground_truths: Optional ground truth data

    Returns:
        Tuple of (high_confidence_examples, low_confidence_examples)
    """
    high_confidence_examples = []
    low_confidence_examples = []

    for i, (pred_batch, text) in enumerate(zip(predictions, texts)):
        # Calculate example-level confidence (average of all predictions in this example)
        if pred_batch:
            scores = []
            for pred in pred_batch:
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


def process_char_to_token_spans(raw_predictions: List, data: List[Dict]) -> List[List]:
    """
    Convert character-level predictions to token-level predictions
    Matches ground truth format

    Args:
        raw_predictions: Raw predictions from GLiNER (character spans)
        data: Original data with tokenized_text

    Returns:
        List of processed predictions (token spans)
    """
    processed_predictions = []

    for i, predictions in enumerate(raw_predictions):
        tokenized_text = data[i]["tokenized_text"]
        pred_entities = []

        for pred in predictions:
            # Convert character positions back to word positions
            text = " ".join(tokenized_text)
            char_start = pred['start']
            char_end = pred['end']

            # Find word positions - improved logic to match GLiNER's evaluation
            word_start = None
            word_end = None
            char_pos = 0

            for word_idx, word in enumerate(tokenized_text):
                word_len = len(word)
                word_char_end = char_pos + word_len

                # Check if this word contains the start position
                if word_start is None and char_pos <= char_start < word_char_end:
                    word_start = word_idx

                # Check if this word contains or ends at the end position
                if char_pos < char_end <= word_char_end:
                    word_end = word_idx
                    break

                char_pos = word_char_end + 1  # +1 for space

            if word_start is not None and word_end is not None:
                pred_entities.append([
                    word_start, word_end, pred['label'].lower(),
                    pred['text'], pred['score']
                ])

        processed_predictions.append(pred_entities)

    return processed_predictions
