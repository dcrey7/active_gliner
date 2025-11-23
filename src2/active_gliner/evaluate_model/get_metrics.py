from typing import List, Dict, Optional
from .utils import convert_char_to_token_format
from .get_entity_level_report import calculate_entity_metrics, create_classification_report
from .get_confidence_report import analyze_confidence_distribution, separate_examples_by_confidence


def evaluate_with_ground_truth(predictions: List[List[Dict]], data: List[Dict],
                              entity_types: List[str], has_confidence: bool = True) -> Dict:
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: GLiNER predictions (character-level)
                    [[{'start': 17, 'end': 29, 'label': 'actor', 'text': '...', 'score': 0.95}, ...], ...]
        data: Ground truth in training format (token-level)
              [{'tokenized_text': [...], 'ner': [[start, end, label], ...], 'text': 'original sentence'}, ...]
        entity_types: List of entity types ['actor', 'genre', ...]
        has_confidence: Whether predictions have scores (GLiNER=True, LLM=False)

    Returns:
        {
            'overall_metrics': {
                'total_predictions': int,
                'overall_confidence': float (if has_confidence),
                'overall_confidence_pct': float (if has_confidence),
                'overall_f1': float,
                'overall_f1_pct': float,
                'total_examples': int,
                'correct_examples': int,
                'incorrect_examples': int,
                'example_level_accuracy': float,
                'example_level_accuracy_pct': float,
                'entity_level_accuracy': float,
                'entity_level_accuracy_pct': float
            },
            'classification_report': styled_df,
            'classification_report_df': df,
            'all_predictions': [{'tokenized_text': ..., 'ner': ..., 'predictions': ..., 'scores': ...}],
            'incorrect_examples_list': [example indices],
            'corrected_examples': [{'tokenized_text': ..., 'ner': ..., 'predictions': ...}],
            'confidence_bins': styled_df (if has_confidence),
            'tp_confidence_analysis': styled_df (if has_confidence),
            'fp_confidence_analysis': styled_df (if has_confidence)
        }

    Example:
        # GLiNER evaluation (with confidence)
        model = DefaultModel(device='cuda')
        model.load_for_inference(adapter_path='/app/models/adapter')
        predictions = model.predict_entities(texts, entity_types)
        results = evaluate_with_ground_truth(predictions, test_data, entity_types, has_confidence=True)

        # LLM evaluation (no confidence)
        llm_predictor = create_predictor(backend="ollama", model="gemma3:12b", ...)
        llm_results = llm_predictor.generate(examples=test_data, entity_types=entity_types)
        llm_predictions = llm_results['all_labels']  # In NER format
        # Convert NER format to character format for this function
        results = evaluate_with_ground_truth(llm_predictions, test_data, entity_types, has_confidence=False)
    """
    # Validate ground truth exists
    if not all('ner' in example for example in data):
        raise ValueError("Ground truth required. All examples must have 'ner' field.")

    # Convert character-level predictions to token-level
    token_predictions = convert_char_to_token_format(predictions, data)

    # Extract ground truths
    ground_truths = [example['ner'] for example in data]

    # Calculate entity-level metrics
    entity_result = calculate_entity_metrics(
        token_predictions, ground_truths, data, entity_types
    )

    # Create classification report
    classification_report_styled, classification_report_df = create_classification_report(
        entity_result['entity_stats'], entity_types, has_confidence,
        tp_scores_by_entity=entity_result['tp_scores_by_entity'] if has_confidence else None
    )

    # Get overall F1 from classification report
    overall_f1 = classification_report_df[classification_report_df['entity_type'] == 'micro_avg']['f1'].iloc[0]

    # Build overall metrics
    overall_metrics = {
        'overall_f1': overall_f1,
        'overall_f1_pct': overall_f1 * 100,
        'total_examples': entity_result['total_examples'],
        'correct_examples': entity_result['correct_examples'],
        'incorrect_examples': entity_result['incorrect_examples'],
        'example_level_accuracy': entity_result['example_level_accuracy'],
        'example_level_accuracy_pct': entity_result['example_level_accuracy_pct'],
        'entity_level_accuracy': entity_result['entity_level_accuracy'],
        'entity_level_accuracy_pct': entity_result['entity_level_accuracy_pct']
    }

    # Build all_predictions with ground truth
    all_predictions = []
    for i, (gt, pred) in enumerate(zip(ground_truths, token_predictions)):
        all_predictions.append({
            'tokenized_text': data[i]['tokenized_text'],
            'ner': gt,
            'predictions': [[p[0], p[1], p[2]] for p in pred],
            'scores': [p[4] for p in pred]
        })

    result = {
        'overall_metrics': overall_metrics,
        'classification_report': classification_report_styled,
        'classification_report_df': classification_report_df,
        'all_predictions': all_predictions,
        'incorrect_examples_list': entity_result['incorrect_examples_list'],
        'corrected_examples': entity_result['corrected_examples']
    }

    # Add confidence analysis if available
    if has_confidence:
        confidence_result = analyze_confidence_distribution(
            predictions, entity_types,
            tp_scores_by_entity=entity_result['tp_scores_by_entity'],
            fp_scores_by_entity=entity_result['fp_scores_by_entity']
        )

        # Add to overall metrics
        overall_metrics['total_predictions'] = confidence_result['total_predictions']
        overall_metrics['overall_confidence'] = confidence_result['overall_confidence']
        overall_metrics['overall_confidence_pct'] = confidence_result['overall_confidence_pct']

        # Add confidence reports
        result['confidence_bins'] = confidence_result['confidence_bins']
        result['tp_confidence_analysis'] = confidence_result['tp_confidence_analysis']
        result['fp_confidence_analysis'] = confidence_result['fp_confidence_analysis']

    return result


def evaluate_without_ground_truth(predictions: List[List[Dict]], entity_types: List[str]) -> Dict:
    """
    Evaluate predictions without ground truth (confidence analysis only).

    Args:
        predictions: GLiNER predictions (character-level with scores)
                    [[{'start': 17, 'end': 29, 'label': 'actor', 'text': '...', 'score': 0.95}, ...], ...]
        entity_types: List of entity types ['actor', 'genre', ...]

    Returns:
        {
            'overall_metrics': {
                'total_predictions': int,
                'overall_confidence': float,
                'overall_confidence_pct': float
            },
            'confidence_bins': styled_df,
            'high_confidence_examples': [{'text': ..., 'predictions': ..., 'avg_confidence': ...}],
            'low_confidence_examples': [{'text': ..., 'predictions': ..., 'avg_confidence': ...}]
        }

    Raises:
        ValueError: If predictions don't have confidence scores

    Example:
        model = DefaultModel(device='cuda')
        model.load_for_inference(adapter_path='/app/models/adapter')
        predictions = model.predict_entities(unlabeled_texts, entity_types)
        results = evaluate_without_ground_truth(predictions, entity_types)
        print(f"High confidence examples: {len(results['high_confidence_examples'])}")
    """
    # Validate predictions have scores
    has_scores = any(
        'score' in pred
        for example_preds in predictions
        for pred in example_preds
    )

    if not has_scores:
        raise ValueError(
            "Predictions must have confidence scores for evaluation without ground truth. "
            "LLM predictions don't have scores, so this function cannot be used with LLM predictions."
        )

    # Analyze confidence distribution
    confidence_result = analyze_confidence_distribution(predictions, entity_types)

    # Separate examples by confidence
    texts = [" ".join([pred['text'] for pred in example_preds]) for example_preds in predictions]
    high_conf_examples, low_conf_examples = separate_examples_by_confidence(
        predictions, texts, confidence_result['overall_confidence']
    )

    return {
        'overall_metrics': {
            'total_predictions': confidence_result['total_predictions'],
            'overall_confidence': confidence_result['overall_confidence'],
            'overall_confidence_pct': confidence_result['overall_confidence_pct']
        },
        'confidence_bins': confidence_result['confidence_bins'],
        'high_confidence_examples': high_conf_examples,
        'low_confidence_examples': low_conf_examples
    }
