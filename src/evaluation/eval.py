"""
Main evaluation orchestrators for NER systems

Two main functions:
- evaluate_gloner(): For GLiNER predictions (with confidence scores)
- evaluate_llm(): For LLM predictions (no confidence scores)

Both functions accept pre-computed predictions (not models).
"""

import torch
from typing import List, Dict, Optional
from .eval_metrics import (
    calculate_ner_metrics,
    analyze_confidence_metrics,
    separate_examples_by_confidence,
    process_char_to_token_spans
)


def evaluate_gloner(predictions: List, data: List[Dict], entity_types: List[str],
                   has_ground_truth: bool = True) -> Dict:
    """
    Evaluate GLiNER predictions

    GLiNER predictions always have confidence scores.
    Can evaluate with or without ground truth labels.

    Args:
        predictions: GLiNER format predictions (from gloner.predict())
                    List of lists of dicts with 'start', 'end', 'label', 'text', 'score'
        data: Original data with 'tokenized_text' (and optionally 'ner' for ground truth)
        entity_types: List of entity types
        has_ground_truth: Whether data contains ground truth labels

    Returns:
        When has_ground_truth=True:
            {
                'overall_metrics': {...},           # F1, confidence, accuracy
                'confidence_bins': styled_df,       # Confidence distribution
                'all_predictions': [...],           # All predictions with GT
                'classification_report': styled_df, # F1/P/R per entity
                'classification_report_df': df,     # Raw classification report
                'tp_confidence_analysis': styled_df,# TP confidence distribution
                'fp_confidence_analysis': styled_df,# FP confidence distribution
                'incorrect_examples': [...],        # Examples with errors
                'corrected_examples': [...]         # Corrected versions
            }

        When has_ground_truth=False:
            {
                'overall_metrics': {...},           # Total predictions + confidence
                'confidence_bins': styled_df,       # Confidence distribution
                'all_predictions': [...],           # All predictions
                'high_confidence_examples': [...],  # High confidence examples
                'low_confidence_examples': [...]    # Low confidence examples
            }

    Example:
        # With ground truth
        gloner = GLONER.for_training()
        predictions = gloner.predict(test_data, entity_types, device='cuda')
        results = evaluate_gloner(predictions, test_data, entity_types, has_ground_truth=True)
        print(results['overall_metrics']['overall_f1_pct'])

        # Without ground truth (exploration)
        predictions = gloner.predict(unlabeled_data, entity_types, device='cuda')
        results = evaluate_gloner(predictions, unlabeled_data, entity_types, has_ground_truth=False)
        print(results['high_confidence_examples'])
    """
    # Prepare texts for analysis
    texts = []
    ground_truths = [] if has_ground_truth else None

    for example in data:
        text = " ".join(example["tokenized_text"])
        texts.append(text)
        if has_ground_truth and "ner" in example:
            ground_truths.append(example["ner"])

    # Process predictions to match ground truth format (char spans -> token spans)
    processed_predictions = process_char_to_token_spans(predictions, data)

    # Analyze confidence metrics (always available for GLiNER)
    if has_ground_truth:
        # We'll get TP/FP scores from NER metrics calculation
        # For now, pass None - will be added after NER metrics
        confidence_result = analyze_confidence_metrics(
            predictions, entity_types,
            tp_scores_by_entity=None,
            fp_scores_by_entity=None
        )
    else:
        confidence_result = analyze_confidence_metrics(
            predictions, entity_types
        )

    overall_metrics = {
        'total_predictions': confidence_result['total_predictions'],
        'overall_confidence': confidence_result['overall_confidence'],
        'overall_confidence_pct': confidence_result['overall_confidence_pct']
    }

    results = {
        'overall_metrics': overall_metrics,
        'confidence_bins': confidence_result['confidence_bins'],
    }

    if has_ground_truth and ground_truths:
        # Full analysis with ground truth
        ner_result = calculate_ner_metrics(
            processed_predictions, ground_truths, data, entity_types
        )

        # Now add TP/FP confidence analysis
        tp_fp_confidence = analyze_confidence_metrics(
            predictions, entity_types,
            tp_scores_by_entity=ner_result['tp_scores_by_entity'],
            fp_scores_by_entity=ner_result['fp_scores_by_entity']
        )

        # Build all_predictions with ground truth
        full_predictions_gt = []
        for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
            full_predictions_gt.append({
                "tokenized_text": data[i]["tokenized_text"],
                "ner": gt,
                "predictions": [[p[0], p[1], p[2]] for p in pred],
                "scores": [p[4] for p in pred]
            })

        # Update overall metrics with NER metrics
        overall_metrics.update({
            'overall_f1': ner_result['overall_f1'],
            'overall_f1_pct': ner_result['overall_f1_pct'],
            'total_examples': ner_result['total_examples'],
            'incorrect_examples': ner_result['incorrect_examples'],
            'correct_examples': ner_result['correct_examples'],
            'example_level_accuracy': ner_result['example_level_accuracy'],
            'example_level_accuracy_pct': ner_result['example_level_accuracy_pct'],
            'entity_level_accuracy': ner_result['entity_level_accuracy'],
            'entity_level_accuracy_pct': ner_result['entity_level_accuracy_pct'],
        })

        # Combine results
        results.update({
            'overall_metrics': overall_metrics,
            'classification_report': ner_result['classification_report'],
            'classification_report_df': ner_result['classification_report_df'],
            'all_predictions': full_predictions_gt,
            'tp_confidence_analysis': tp_fp_confidence['tp_confidence_analysis'],
            'fp_confidence_analysis': tp_fp_confidence['fp_confidence_analysis'],
            'incorrect_examples': ner_result['incorrect_examples'],
            'corrected_examples': ner_result['corrected_examples'],
        })

    else:
        # No ground truth - only confidence analysis
        # Build all_predictions without ground truth
        full_predictions = []
        for i, pred in enumerate(processed_predictions):
            full_predictions.append({
                "tokenized_text": data[i]["tokenized_text"],
                "predictions": [[p[0], p[1], p[2]] for p in pred],
                "scores": [p[4] for p in pred]
            })

        # Separate examples by confidence
        high_conf_examples, low_conf_examples = separate_examples_by_confidence(
            predictions, texts, overall_metrics['overall_confidence']
        )

        results.update({
            'all_predictions': full_predictions,
            'high_confidence_examples': high_conf_examples,
            'low_confidence_examples': low_conf_examples,
        })

    return results


def evaluate_llm(predictions: List[Dict], data: List[Dict], entity_types: List[str]) -> Dict:
    """
    Evaluate LLM predictions against ground truth

    LLM predictions have NO confidence scores.
    ALWAYS requires ground truth (no confidence to analyze without it).

    Args:
        predictions: NER format predictions (from llm_inference.generate())
                    List of dicts with 'tokenized_text' and 'ner' fields
                    Format: [{"tokenized_text": [...], "ner": [[start, end, type], ...]}, ...]
        data: Original data with 'tokenized_text' and 'ner' (ground truth required)
        entity_types: List of entity types

    Returns:
        {
            'overall_metrics': {...},           # F1, accuracy (NO confidence)
            'all_predictions': [...],           # All predictions with GT
            'classification_report': styled_df, # F1/P/R per entity
            'classification_report_df': df,     # Raw classification report
            'incorrect_examples': [...],        # Examples with errors
            'corrected_examples': [...]         # Corrected versions
        }

    Note:
        - NO confidence_bins (LLM has no scores)
        - NO tp_confidence_analysis (LLM has no scores)
        - NO fp_confidence_analysis (LLM has no scores)

    Raises:
        ValueError: If data doesn't contain ground truth ('ner' field)

    Example:
        llm_predictor = create_predictor(backend="ollama", model="gemma3:12b", ...)
        llm_results = llm_predictor.generate(examples=test_data, entity_types=entity_types)
        llm_predictions = llm_results['all_labels']  # NER format

        results = evaluate_llm(llm_predictions, test_data, entity_types)
        print(results['overall_metrics']['overall_f1_pct'])
    """
    # Validate that ground truth exists
    if not all("ner" in example for example in data):
        raise ValueError(
            "LLM evaluation requires ground truth labels. "
            "All examples in 'data' must have 'ner' field. "
            "Without ground truth, LLM evaluation is meaningless (no confidence scores to analyze)."
        )

    # Extract ground truth
    ground_truths = [example["ner"] for example in data]

    # Convert LLM predictions to processed format
    # LLM predictions are in NER format: [{"tokenized_text": [...], "ner": [[start, end, type], ...]}, ...]
    # Need to convert to processed format: [[[start, end, type, text, score], ...], ...]
    processed_predictions = []
    for pred_example in predictions:
        pred_entities = []
        tokenized_text = pred_example['tokenized_text']

        for ner_span in pred_example['ner']:
            start, end, entity_type = ner_span[0], ner_span[1], ner_span[2]
            # Extract entity text from tokenized text
            entity_text = " ".join(tokenized_text[start:end+1])
            # LLM has no confidence scores - use 1.0 as placeholder for format compatibility
            pred_entities.append([start, end, entity_type, entity_text, 1.0])

        processed_predictions.append(pred_entities)

    # Calculate NER metrics (no confidence analysis)
    ner_result = calculate_ner_metrics(
        processed_predictions, ground_truths, data, entity_types
    )

    # Build all_predictions with ground truth
    full_predictions_gt = []
    for i, (gt, pred) in enumerate(zip(ground_truths, processed_predictions)):
        full_predictions_gt.append({
            "tokenized_text": data[i]["tokenized_text"],
            "ner": gt,
            "predictions": [[p[0], p[1], p[2]] for p in pred],
            "scores": [p[4] for p in pred]  # Will be all 1.0
        })

    # Overall metrics (NO confidence)
    overall_metrics = {
        'overall_f1': ner_result['overall_f1'],
        'overall_f1_pct': ner_result['overall_f1_pct'],
        'total_examples': ner_result['total_examples'],
        'incorrect_examples': ner_result['incorrect_examples'],
        'correct_examples': ner_result['correct_examples'],
        'example_level_accuracy': ner_result['example_level_accuracy'],
        'example_level_accuracy_pct': ner_result['example_level_accuracy_pct'],
        'entity_level_accuracy': ner_result['entity_level_accuracy'],
        'entity_level_accuracy_pct': ner_result['entity_level_accuracy_pct'],
    }

    return {
        'overall_metrics': overall_metrics,
        'classification_report': ner_result['classification_report'],
        'classification_report_df': ner_result['classification_report_df'],
        'all_predictions': full_predictions_gt,
        'incorrect_examples': ner_result['incorrect_examples'],
        'corrected_examples': ner_result['corrected_examples'],
    }
