import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from .utils import apply_gradient_styling


def analyze_confidence_distribution(predictions: List[List[Dict]], entity_types: List[str],
                                   tp_scores_by_entity: Optional[Dict] = None,
                                   fp_scores_by_entity: Optional[Dict] = None) -> Dict:
    """
    Analyze confidence distribution for predictions.

    Args:
        predictions: GLiNER predictions (character-level with scores)
        entity_types: List of entity types
        tp_scores_by_entity: {entity_type: [scores]} for true positives (optional)
        fp_scores_by_entity: {entity_type: [scores]} for false positives (optional)

    Returns:
        {
            'overall_confidence': float,
            'overall_confidence_pct': float,
            'total_predictions': int,
            'confidence_bins': styled_df,
            'tp_confidence_analysis': styled_df (if tp_scores provided),
            'fp_confidence_analysis': styled_df (if fp_scores provided)
        }
    """
    # Calculate overall confidence and create bins
    all_scores = []
    for example_preds in predictions:
        for pred in example_preds:
            if 'score' in pred:
                all_scores.append(pred['score'])

    overall_confidence = sum(all_scores) / len(all_scores) if all_scores else 0
    total_predictions = len(all_scores)

    # Create confidence bins
    confidence_bins_df = create_confidence_bins(predictions, entity_types)

    result = {
        'overall_confidence': overall_confidence,
        'overall_confidence_pct': overall_confidence * 100,
        'total_predictions': total_predictions,
        'confidence_bins': apply_gradient_styling(confidence_bins_df, title="Confidence Distribution")
    }

    # Add TP/FP confidence analysis if provided
    if tp_scores_by_entity is not None and fp_scores_by_entity is not None:
        tp_bins_df, fp_bins_df = create_tp_fp_confidence_bins(
            tp_scores_by_entity, fp_scores_by_entity, entity_types
        )
        result['tp_confidence_analysis'] = apply_gradient_styling(tp_bins_df, title="True Positives Confidence")
        result['fp_confidence_analysis'] = apply_gradient_styling(fp_bins_df, title="False Positives Confidence")

    return result


def create_confidence_bins(predictions: List[List[Dict]], entity_types: List[str]) -> pd.DataFrame:
    """
    Create confidence bins: 0-25%, 26-50%, 51-75%, 76-100%.

    Args:
        predictions: GLiNER predictions (character-level)
        entity_types: List of entity types

    Returns:
        DataFrame with confidence bins
    """
    bins = {
        '0-25%': defaultdict(int),
        '26-50%': defaultdict(int),
        '51-75%': defaultdict(int),
        '76-100%': defaultdict(int)
    }

    for example_preds in predictions:
        for pred in example_preds:
            if 'score' not in pred:
                continue

            score = pred['score']
            label = pred['label']

            if score <= 0.25:
                bins['0-25%'][label] += 1
            elif score <= 0.50:
                bins['26-50%'][label] += 1
            elif score <= 0.75:
                bins['51-75%'][label] += 1
            else:
                bins['76-100%'][label] += 1

    # Build dataframe
    rows = []
    for entity in entity_types:
        row = {'entity_type': entity}
        for bin_name in ['0-25%', '26-50%', '51-75%', '76-100%']:
            row[bin_name] = bins[bin_name][entity]
        rows.append(row)

    # Add totals row
    total_row = {'entity_type': 'total'}
    for bin_name in ['0-25%', '26-50%', '51-75%', '76-100%']:
        total_row[bin_name] = sum(bins[bin_name].values())
    rows.append(total_row)

    return pd.DataFrame(rows)


def create_tp_fp_confidence_bins(tp_scores_by_entity: Dict, fp_scores_by_entity: Dict,
                                 entity_types: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate confidence bins for TP and FP.

    Args:
        tp_scores_by_entity: {entity_type: [scores]} for true positives
        fp_scores_by_entity: {entity_type: [scores]} for false positives
        entity_types: List of entity types

    Returns:
        (tp_bins_df, fp_bins_df)
    """
    def create_bins_from_scores(scores_dict, entity_types):
        bins = {
            '0-25%': defaultdict(int),
            '26-50%': defaultdict(int),
            '51-75%': defaultdict(int),
            '76-100%': defaultdict(int)
        }

        for entity, scores in scores_dict.items():
            for score in scores:
                if score <= 0.25:
                    bins['0-25%'][entity] += 1
                elif score <= 0.50:
                    bins['26-50%'][entity] += 1
                elif score <= 0.75:
                    bins['51-75%'][entity] += 1
                else:
                    bins['76-100%'][entity] += 1

        # Build dataframe
        rows = []
        for entity in entity_types:
            row = {'entity_type': entity}
            for bin_name in ['0-25%', '26-50%', '51-75%', '76-100%']:
                row[bin_name] = bins[bin_name][entity]
            rows.append(row)

        # Add totals row
        total_row = {'entity_type': 'total'}
        for bin_name in ['0-25%', '26-50%', '51-75%', '76-100%']:
            total_row[bin_name] = sum(bins[bin_name].values())
        rows.append(total_row)

        return pd.DataFrame(rows)

    tp_bins_df = create_bins_from_scores(tp_scores_by_entity, entity_types)
    fp_bins_df = create_bins_from_scores(fp_scores_by_entity, entity_types)

    return tp_bins_df, fp_bins_df


def separate_examples_by_confidence(predictions: List[List[Dict]], texts: List[str],
                                   threshold: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate examples into high and low confidence based on average score.

    Args:
        predictions: GLiNER predictions (character-level)
        texts: Original texts
        threshold: Confidence threshold (usually overall_confidence)

    Returns:
        (high_confidence_examples, low_confidence_examples)
    """
    high_conf = []
    low_conf = []

    for text, example_preds in zip(texts, predictions):
        if not example_preds:
            continue

        # Calculate average confidence for this example
        scores = [pred['score'] for pred in example_preds if 'score' in pred]
        avg_score = sum(scores) / len(scores) if scores else 0

        example_dict = {
            'text': text,
            'predictions': example_preds,
            'avg_confidence': avg_score
        }

        if avg_score >= threshold:
            high_conf.append(example_dict)
        else:
            low_conf.append(example_dict)

    return high_conf, low_conf
