from typing import List, Dict, Any, Optional, Tuple
from .gliner_utils import tokenize_text, char_to_word_positions
import logging



def create_bio_tags(
    tokens: List[str],
    entities: List[Dict],
    sentence: str
) -> List[str]:
    """
    Create BIO tags from entities with character positions

    Args:
        tokens: List of word tokens
        entities: List of entities with character positions
            [{"name": "...", "type": "...", "pos": [char_start, char_end]}]
        sentence: Original sentence (needed for char→word conversion)

    Returns:
        List of BIO tags (e.g., ["O", "O", "B-actor", "I-actor"])

    Example:
        >>> tokens = ["what", "movies", "star", "bruce", "willis"]
        >>> entities = [{"name": "bruce willis", "type": "actor", "pos": [17, 29]}]
        >>> create_bio_tags(tokens, entities, "what movies star bruce willis")
        ["O", "O", "O", "B-actor", "I-actor"]
    """
    bio_tags = ["O"] * len(tokens)

    for entity in entities:
        entity_type = entity["type"]
        char_start, char_end = entity["pos"]

        # Convert character positions to word positions
        word_start, word_end = char_to_word_positions(sentence, char_start, char_end)

        if word_start is not None and word_end is not None:
            # Assign BIO tags
            bio_tags[word_start] = f"B-{entity_type}"
            for i in range(word_start + 1, word_end + 1):
                if i < len(bio_tags):  # Safety check
                    bio_tags[i] = f"I-{entity_type}"

    return bio_tags


def create_label_to_id_mapping(entity_types: List[str]) -> Dict[str, int]:
    """
    Create numeric ID mapping for BIO tags

    Args:
        entity_types: List of entity type labels (e.g., ["actor", "genre", "year"])

    Returns:
        Dictionary mapping BIO tags to numeric IDs

    Example:
        >>> create_label_to_id_mapping(["actor", "genre"])
        {"O": 0, "B-actor": 1, "I-actor": 2, "B-genre": 3, "I-genre": 4}
    """
    label_to_id = {"O": 0}
    current_id = 1

    for entity_type in entity_types:
        label_to_id[f"B-{entity_type}"] = current_id
        current_id += 1
        label_to_id[f"I-{entity_type}"] = current_id
        current_id += 1

    return label_to_id




def convert_bio_to_ids(bio_tags: List[str], label_to_id: Dict[str, int]) -> List[int]:
    """
    Convert BIO tags to numeric indices

    Args:
        bio_tags: List of BIO tags (e.g., ["O", "B-actor", "I-actor"])
        label_to_id: Mapping from BIO tags to IDs

    Returns:
        List of numeric IDs

    Example:
        >>> bio_tags = ["O", "O", "B-actor", "I-actor"]
        >>> label_to_id = {"O": 0, "B-actor": 1, "I-actor": 2}
        >>> convert_bio_to_ids(bio_tags, label_to_id)
        [0, 0, 1, 2]
    """
    return [label_to_id.get(tag, 0) for tag in bio_tags]

