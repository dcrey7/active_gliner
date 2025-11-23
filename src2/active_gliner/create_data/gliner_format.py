from .gliner_utils import tokenize_text, char_to_word_positions, find_all_entity_occurrences
from typing import List, Tuple, Optional, Dict





def convert_raw_json_to_gliner_training(data_list: List[dict]) -> List[dict]:
    """
    Convert raw JSON data to GLiNER format.

    Args:
        data_list: List of dicts with 'sentence' and 'entities' fields

    Returns:
        List of dicts with 'tokenized_text' and 'ner' fields

    Example:
        Input: [{'sentence': 'Hello world', 'entities': [{'pos': [0, 5], 'type': 'greeting'}]}]
        Output: [{'tokenized_text': ['Hello', 'world'], 'ner': [(0, 0, 'greeting')]}]
    """
    training_data = []

    for data in data_list:
        tokens = tokenize_text(data['sentence'])
        entities = []

        for entity in data['entities']:
            char_start, char_end = entity['pos']
            word_start, word_end = char_to_word_positions(data['sentence'], char_start, char_end)

            if word_start is not None and word_end is not None:
                entities.append((word_start, word_end, entity['type'].lower()))

        training_data.append({
            'tokenized_text': tokens,
            'ner': entities,
            'text': data['sentence']  # Preserve original text for evaluation
        })

    return training_data


def convert_prediction_tasks_json_to_gliner_training(
    filtered_tasks: List[dict],
    prediction_index: Optional[int] = None,
    model_keywords: Optional[List[str]] = None
) -> List[dict]:
    """
    Convert Label Studio prediction tasks to GLiNER training format.

    Select prediction by EITHER index OR model keywords (not both).

    Args:
        filtered_tasks: List of Label Studio tasks
            Format: [{'task': task_dict}, ...]
        prediction_index: Which prediction to use by index (optional)
            - 0: First prediction
            - -1: Last prediction
            - None: Use model_keywords instead (default)
        model_keywords: Keywords to search in model_version (optional)
            - Used only if prediction_index is None
            - Finds first prediction matching any keyword
            - Default: ['cerebras'] (backward compatible)

    Returns:
        List of GLiNER format dicts

    Example:
        >>> # By index (first prediction, any model)
        >>> data = convert_prediction_tasks_json_to_gliner_training(
        ...     tasks,
        ...     prediction_index=0
        ... )

        >>> # By model keywords (Cerebras or Ollama)
        >>> data = convert_prediction_tasks_json_to_gliner_training(
        ...     tasks,
        ...     model_keywords=['cerebras', 'ollama']
        ... )
    """
    # Default to Cerebras for backward compatibility
    if prediction_index is None and model_keywords is None:
        model_keywords = ['cerebras']

    training_data = []

    for item in filtered_tasks:
        task = item['task']
        predictions = task.get('predictions', [])

        if not predictions:
            continue

        # Select prediction by index OR keywords
        selected_pred = None

        if prediction_index is not None:
            # Use index
            try:
                selected_pred = predictions[prediction_index]
            except IndexError:
                # Fallback to first if index out of range
                selected_pred = predictions[0] if predictions else None
        else:
            # Use model keywords
            for pred in predictions:
                model_version = pred.get('model_version', '')
                if any(keyword.lower() in model_version.lower() for keyword in model_keywords):
                    selected_pred = pred
                    break

        if not selected_pred:
            continue

        text = task['data']['text']
        tokens = tokenize_text(text)

        ner_tags = []
        for entity_result in selected_pred.get('result', []):
            value = entity_result.get('value', {})
            start_idx, end_idx = char_to_word_positions(text, value['start'], value['end'])

            if start_idx is not None and end_idx is not None:
                label = value['labels'][0].replace('->', '<>')
                ner_tags.append((start_idx, end_idx, label))

        training_data.append({
            'tokenized_text': tokens,
            'ner': ner_tags,
            'text': text
        })

    return training_data


def convert_annotated_tasks_json_to_gliner_training(
    filtered_tasks: List[dict],
    annotation_index: int = 0
) -> List[dict]:
    """
    Convert Label Studio annotated tasks to GLiNER training format.

    Args:
        filtered_tasks: List of Label Studio tasks
            Format: [{'task': task_dict}, ...]
        annotation_index: Which annotation to use if multiple exist (default: 0)
            - 0: First annotation (default - usually ground truth)
            - -1: Last annotation (most recent)
            - Any int: Specific annotation index

    Returns:
        List of GLiNER format dicts

    Example:
        >>> # Use first annotation (default)
        >>> data = convert_annotated_tasks_json_to_gliner_training(tasks)

        >>> # Use last annotation (most recent)
        >>> data = convert_annotated_tasks_json_to_gliner_training(
        ...     tasks,
        ...     annotation_index=-1
        ... )
    """
    training_data = []

    for item in filtered_tasks:
        task = item['task']
        annotations = task.get('annotations', [])

        if not annotations:
            continue

        # Get annotation by index
        try:
            ann = annotations[annotation_index]
        except IndexError:
            # Fallback to first if index out of range
            ann = annotations[0]

        text = task['data']['text']
        tokens = tokenize_text(text)

        ner_tags = []
        for entity_result in ann.get('result', []):
            value = entity_result.get('value', {})
            start_idx, end_idx = char_to_word_positions(text, value['start'], value['end'])

            if start_idx is not None and end_idx is not None:
                label = value['labels'][0].replace('->', '<>')
                ner_tags.append((start_idx, end_idx, label))

        training_data.append({
            'tokenized_text': tokens,
            'ner': ner_tags,
            'text': text
        })

    return training_data


def convert_llm_entities_to_gliner_predictions(entities: List[Dict], text: str) -> List[Dict]:
    """
    Convert LLM entity format to GLiNER prediction format (character-level)

    Input: entities=[{'entity': 'Brad Pitt', 'types': ['actor']}], text="Brad Pitt stars..."
    Output: [{'start': 0, 'end': 9, 'text': 'Brad Pitt', 'label': 'actor'}]
    """
    predictions = []

    for entity in entities:
        entity_text = entity['entity']
        entity_types = entity['types']

        occurrences = find_all_entity_occurrences(entity_text, text)

        for start, end in occurrences:
            for entity_type in entity_types:
                predictions.append({
                    'start': start,
                    'end': end,
                    'text': entity_text,
                    'label': entity_type
                })

    return predictions


def convert_llm_entities_to_gliner_training(entities: List[Dict], text: str) -> Dict:
    """
    Convert LLM entity format to GLiNER training format (token-level)

    Input: entities=[{'entity': 'Brad Pitt', 'types': ['actor']}], text="Brad Pitt stars in..."
    Output: {'tokenized_text': ['Brad', 'Pitt', 'stars', 'in', '...'], 'ner': [(0, 1, 'actor')], 'text': 'Brad Pitt stars in...'}
    """
    tokens = tokenize_text(text)
    ner_tags = []

    for entity in entities:
        entity_text = entity['entity']
        entity_types = entity['types']

        occurrences = find_all_entity_occurrences(entity_text, text)

        for char_start, char_end in occurrences:
            token_start, token_end = char_to_word_positions(text, char_start, char_end)

            if token_start is not None and token_end is not None:
                for entity_type in entity_types:
                    ner_tags.append((token_start, token_end, entity_type))

    return {
        'tokenized_text': tokens,
        'ner': ner_tags,
        'text': text
    }
