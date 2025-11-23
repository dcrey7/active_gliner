import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
from .utils import apply_gradient_styling


def calculate_entity_metrics(predictions: List[List[List]], ground_truths: List[List[List]],
                            data: List[Dict], entity_types: List[str]) -> Dict:
    """
    Calculate TP/FP/FN per entity type at token level.

    Args:
        predictions: Token-level predictions [[[start, end, label, text, score], ...], ...]
        ground_truths: Ground truth labels [[[start, end, label], ...], ...]
        data: Original data with tokenized_text
        entity_types: List of entity types

    Returns:
        {
            'entity_stats': {entity_type: {'tp': int, 'fp': int, 'fn': int}},
            'tp_scores_by_entity': {entity_type: [scores]},
            'fp_scores_by_entity': {entity_type: [scores]},
            'total_examples': int,
            'correct_examples': int,
            'incorrect_examples': int,
            'incorrect_examples_list': [example indices],
            'corrected_examples': [{'tokenized_text': ..., 'ner': ..., 'predictions': ...}]
        }
    """
    entity_stats = {entity: {'tp': 0, 'fp': 0, 'fn': 0} for entity in entity_types}
    tp_scores_by_entity = defaultdict(list)
    fp_scores_by_entity = defaultdict(list)

    total_examples = len(predictions)
    incorrect_examples_list = []
    corrected_examples = []

    total_correct_entities = 0
    total_entities = 0

    for example_idx, (preds, gt) in enumerate(zip(predictions, ground_truths)):
        # Convert to sets for matching (start, end, label)
        pred_set = {(p[0], p[1], p[2]) for p in preds}
        gt_set = {(g[0], g[1], g[2]) for g in gt}

        # Calculate TP, FP, FN for this example
        example_has_error = False

        # True positives: predictions that match ground truth
        for pred in preds:
            pred_tuple = (pred[0], pred[1], pred[2])
            entity_type = pred[2]
            score = pred[4]

            if pred_tuple in gt_set:
                entity_stats[entity_type]['tp'] += 1
                tp_scores_by_entity[entity_type].append(score)
                total_correct_entities += 1
            else:
                entity_stats[entity_type]['fp'] += 1
                fp_scores_by_entity[entity_type].append(score)
                example_has_error = True

        # False negatives: ground truth entities not predicted
        for g in gt:
            gt_tuple = (g[0], g[1], g[2])
            entity_type = g[2]

            if gt_tuple not in pred_set:
                entity_stats[entity_type]['fn'] += 1
                example_has_error = True

        total_entities += len(gt)

        # Track incorrect examples
        if example_has_error:
            incorrect_examples_list.append(example_idx)
            corrected_examples.append({
                'tokenized_text': data[example_idx]['tokenized_text'],
                'ner': gt,
                'predictions': [[p[0], p[1], p[2]] for p in preds],
                'scores': [p[4] for p in preds]
            })

    correct_examples = total_examples - len(incorrect_examples_list)

    return {
        'entity_stats': entity_stats,
        'tp_scores_by_entity': dict(tp_scores_by_entity),
        'fp_scores_by_entity': dict(fp_scores_by_entity),
        'total_examples': total_examples,
        'correct_examples': correct_examples,
        'incorrect_examples': len(incorrect_examples_list),
        'incorrect_examples_list': incorrect_examples_list,
        'corrected_examples': corrected_examples,
        'example_level_accuracy': correct_examples / total_examples if total_examples > 0 else 0,
        'example_level_accuracy_pct': (correct_examples / total_examples * 100) if total_examples > 0 else 0,
        'entity_level_accuracy': total_correct_entities / total_entities if total_entities > 0 else 0,
        'entity_level_accuracy_pct': (total_correct_entities / total_entities * 100) if total_entities > 0 else 0,
    }


def create_classification_report(entity_stats: Dict, entity_types: List[str],
                                has_confidence: bool = True,
                                tp_scores_by_entity: Dict = None):
    """
    Generate styled classification report with F1, Precision, Recall.

    Args:
        entity_stats: {entity_type: {'tp': int, 'fp': int, 'fn': int}}
        entity_types: List of entity types
        has_confidence: Whether to include confidence columns
        tp_scores_by_entity: {entity_type: [scores]} for calculating avg confidence

    Returns:
        (styled_dataframe, raw_dataframe)
    """
    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_scores = []  # Collect all scores for micro avg

    for entity in entity_types:
        stats = entity_stats[entity]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn

        # Calculate avg confidence for this entity type
        avg_confidence = 0.0
        if has_confidence and tp_scores_by_entity:
            entity_scores = tp_scores_by_entity.get(entity, [])
            if entity_scores:
                avg_confidence = sum(entity_scores) / len(entity_scores)
                all_scores.extend(entity_scores)

        row = {
            'entity_type': entity,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': support
        }

        # Add avg_confidence at the end if has_confidence
        if has_confidence:
            row['avg_confidence'] = avg_confidence

        rows.append(row)

    # Add micro average row
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    micro_support = total_tp + total_fn

    # Calculate overall avg confidence
    micro_avg_confidence = 0.0
    if has_confidence and all_scores:
        micro_avg_confidence = sum(all_scores) / len(all_scores)

    micro_row = {
        'entity_type': 'micro_avg',
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'support': micro_support
    }

    # Add avg_confidence at the end if has_confidence
    if has_confidence:
        micro_row['avg_confidence'] = micro_avg_confidence

    rows.append(micro_row)

    df = pd.DataFrame(rows)
    styled_df = apply_gradient_styling(df, title="Classification Report")

    return styled_df, df
