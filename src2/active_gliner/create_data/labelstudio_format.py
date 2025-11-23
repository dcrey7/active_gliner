from typing import List, Dict, Any, Optional, Tuple
from .gliner_format import tokenize_text, char_to_word_positions
from .labelstudio_utils import *
import logging

def mit_to_labelstudio_tasks_format(
    mit_data: List[Dict],
    entity_types: List[str],
    include_annotations: bool = True,
    include_metadata: bool = True,
    limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    Convert MIT movies format to Label Studio task format with metadata

    Args:
        mit_data: List of MIT format examples
            [{"sentence": "...", "entities": [{"name": "...", "type": "...", "pos": [start, end]}]}]
        entity_types: List of entity type labels (e.g., ["actor", "genre", "year"])
        include_annotations: If True, include ground truth as annotations (default: True)
        include_metadata: If True, include tokens, ner_tags, ner_tags_index in data field (default: True)
        limit: Optional limit on number of examples to convert
        logger: Optional logger instance

    Returns:
        List of Label Studio tasks with metadata and optional annotations
            [{"data": {"text": "...", "tokens": [...], "ner_tags": [...], ...}, "annotations": [...]}]

    Example:
        Output (Label Studio format with metadata):
        {
            "data": {
                "text": "what movies star bruce willis",
                "id": 0,
                "tokens": ["what", "movies", "star", "bruce", "willis"],
                "ner_tags": ["O", "O", "O", "B-actor", "I-actor"],
                "ner_tags_index": [0, 0, 0, 1, 2]
            },
            "annotations": [{
                "result": [{
                    "value": {
                        "start": 17,
                        "end": 29,
                        "text": "bruce willis",
                        "labels": ["actor"]
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                }]
            }]
        }
    """
    if logger:
        logger.info(f"Converting MIT movies data to Label Studio format...")
        logger.info(f"Total examples: {len(mit_data)}, Limit: {limit}")
        logger.info(f"Include annotations: {include_annotations}, Include metadata: {include_metadata}")

    # Create label-to-ID mapping for BIO tags (if metadata is needed)
    label_to_id = create_label_to_id_mapping(entity_types) if include_metadata else None

    # Apply limit if specified
    data_to_convert = mit_data[:limit] if limit else mit_data

    tasks = []
    conversion_errors = 0

    for i, example in enumerate(data_to_convert):
        try:
            sentence = example["sentence"]
            entities = example.get("entities", [])

            # Tokenize sentence
            tokens = tokenize_text(sentence)

            # Create basic task data
            task_data = {
                "text": sentence,
                "mit_dataset_id": i  # Original MIT Movies dataset index for tracking
            }

            # Add metadata if requested (tokens, ner_tags, ner_tags_index)
            if include_metadata:
                # Generate BIO tags from entities
                bio_tags = create_bio_tags(tokens, entities, sentence)

                # Convert BIO tags to numeric indices
                bio_indices = convert_bio_to_ids(bio_tags, label_to_id)

                # Add metadata to task data
                task_data["tokens"] = tokens
                task_data["ner_tags"] = bio_tags
                task_data["ner_tags_index"] = bio_indices

            # Note: Removed prediction_score from task data
            # Prediction scores are stored in prediction objects, not task data

            # Create task structure
            task = {"data": task_data}

            # Add annotations if requested (ground truth)
            if include_annotations and "entities" in example and example["entities"]:
                annotations_result = []

                for entity in example["entities"]:
                    # Convert entity to Label Studio annotation format
                    annotation = {
                        "value": {
                            "start": entity["pos"][0],  # character position start
                            "end": entity["pos"][1],    # character position end
                            "text": entity["name"],     # entity text
                            "labels": [entity["type"]]  # entity type as list
                        },
                        "from_name": "label",  # matches label config
                        "to_name": "text",     # matches label config
                        "type": "labels"       # annotation type
                    }
                    annotations_result.append(annotation)

                # Wrap in annotations structure
                task["annotations"] = [{
                    "result": annotations_result
                }]

            tasks.append(task)

        except Exception as e:
            if logger:
                logger.warning(f"Error converting example {i}: {e}")
            conversion_errors += 1
            continue

    if logger:
        logger.info(f"Conversion complete: {len(tasks)} tasks created, {conversion_errors} errors")
    else:
        print(f"Converted {len(tasks)} tasks ({conversion_errors} errors)")

    return tasks



def create_labelstudio_label_config(entity_types: List[str]) -> str:
    """
    Generate Label Studio label config XML from entity types

    Args:
        entity_types: List of entity type labels (e.g., ["actor", "genre", "year"])

    Returns:
        XML string for Label Studio label config

    Example:
        Input: ["actor", "genre"]
        Output: XML with <Label> tags for each entity type (NO keyboard shortcuts)

    Note:
        Keyboard shortcuts are NOT included to avoid corrupting label names
        when passed to ML backends like GLiNER
    """
    # Color palette for labels (cycling if more than 12 labels)
    colors = [
        "red", "blue", "orange", "green", "purple", "cyan",
        "pink", "brown", "yellow", "lime", "magenta", "teal"
    ]

    # Generate label tags with numeric hotkeys (1-9, 0, then letters)
    # Label Studio auto-assigns letters (q,w,e,r) if not specified, which corrupts labels
    # We explicitly assign numbers to prevent this
    label_tags = []
    hotkey_sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f']

    for i, entity_type in enumerate(entity_types):
        color = colors[i % len(colors)]
        hotkey = hotkey_sequence[i % len(hotkey_sequence)]
        label_tag = f'    <Label value="{entity_type}" background="{color}" hotkey="{hotkey}"/>'
        label_tags.append(label_tag)

    # Combine into full config
    label_config = f"""<View>
  <Labels name="label" toName="text">
{chr(10).join(label_tags)}
  </Labels>
  <Text name="text" value="$text"/>
</View>"""

    return label_config


def create_annotation_from_metadata(
    text: str,
    tokens: List[str],
    ner_tags: List[str],
    from_name: str = "label",
    to_name: str = "text"
) -> List[Dict]:
    """
    Convert task metadata (tokens + BIO tags) to Label Studio annotation format
    
    This is used when you want to create ground truth annotations from MIT dataset metadata
    
    Args:
        text: Original text string
        tokens: List of word tokens
        ner_tags: List of BIO tags (e.g., ['O', 'O', 'B-actor', 'I-actor'])
        from_name: Label Studio from_name (default: "label")
        to_name: Label Studio to_name (default: "text")
    
    Returns:
        List of Label Studio annotation result objects
        
    Example:
        >>> text = "what movies star bruce willis"
        >>> tokens = ["what", "movies", "star", "bruce", "willis"]
        >>> ner_tags = ["O", "O", "O", "B-actor", "I-actor"]
        >>> result = create_annotation_from_metadata(text, tokens, ner_tags)
        >>> print(result)
        [{
            "value": {
                "start": 17,
                "end": 29,
                "text": "bruce willis",
                "labels": ["actor"]
            },
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
        }]
    """
    result = []
    current_entity = None
    current_start = 0
    char_position = 0
    
    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        # Find token position in text (accounting for spaces)
        token_start = text.find(token, char_position)
        
        if token_start == -1:
            # Token not found, skip
            char_position += len(token) + 1
            continue
            
        token_end = token_start + len(token)
        
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                result.append(current_entity)
            
            # Start new entity
            entity_label = tag[2:]  # Remove 'B-' prefix
            current_entity = {
                "value": {
                    "start": token_start,
                    "end": token_end,
                    "text": token,
                    "labels": [entity_label]
                },
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels"
            }
            current_start = token_start
            
        elif tag.startswith('I-') and current_entity:
            # Continue current entity
            current_entity["value"]["end"] = token_end
            current_entity["value"]["text"] = text[current_start:token_end]
            
        else:  # 'O' tag or end of entity
            if current_entity:
                result.append(current_entity)
                current_entity = None
        
        char_position = token_end
    
    # Don't forget last entity
    if current_entity:
        result.append(current_entity)
    
    return result


def batch_create_annotations_from_metadata(
    client,
    tasks: List[Dict],
    ground_truth: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Batch create annotations from task metadata (tokens, ner_tags)
    
    This is useful for converting MIT dataset ground truth to Label Studio annotations
    
    Args:
        client: Label Studio SDK client
        tasks: List of tasks (with 'data' containing 'text', 'tokens', 'ner_tags')
        ground_truth: Mark annotations as ground truth (default: True)
        verbose: Print progress (default: True)
    
    Returns:
        Dictionary with statistics: {'successful': N, 'failed': M, 'skipped': K}
        
    Example:
        >>> client = LabelStudio(base_url="...", api_key="...")
        >>> tasks = [task.model_dump() for task in client.tasks.list(project=29)]
        >>> stats = batch_create_annotations_from_metadata(client, tasks)
        >>> print(f"Created {stats['successful']} annotations")
    """
    stats = {'successful': 0, 'failed': 0, 'skipped': 0}
    
    for task in tasks:
        task_id = task.get('id')
        task_data = task.get('data', {})
        
        # Skip if already has annotations
        if task.get('annotations'):
            if verbose:
                print(f"Task {task_id}: Already has annotations, skipping")
            stats['skipped'] += 1
            continue
        
        # Extract metadata
        text = task_data.get('text')
        tokens = task_data.get('tokens')
        ner_tags = task_data.get('ner_tags')
        
        # Skip if missing metadata
        if not text or not tokens or not ner_tags:
            if verbose:
                print(f"Task {task_id}: Missing metadata, skipping")
            stats['skipped'] += 1
            continue
        
        # Convert to annotation format
        try:
            annotation_result = create_annotation_from_metadata(
                text=text,
                tokens=tokens,
                ner_tags=ner_tags
            )
            
            # Create annotation in Label Studio
            client.annotations.create(
                id=task_id,
                result=annotation_result,
                ground_truth=ground_truth
            )
            
            if verbose:
                print(f"Task {task_id}: Created annotation with {len(annotation_result)} entities")
            
            stats['successful'] += 1
            
        except Exception as e:
            if verbose:
                print(f"Task {task_id}: Error - {e}")
            stats['failed'] += 1
    
    return stats

