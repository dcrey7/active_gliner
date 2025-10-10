"""
Data transformation utilities - simplified versions of your original functions
"""

import re
import json
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging
import random


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text exactly like your original function
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def convert_synthetic_to_ner_format(synthetic_data: List[Dict]) -> List[Dict]:
    """
    Convert synthetic JSON data to NER format exactly like your original function
    
    Args:
        synthetic_data: List of synthetic examples in JSON format
        
    Returns:
        List of examples in NER format
    """
    print(f"Converting {len(synthetic_data)} synthetic examples to NER format...")
    
    all_examples = []
    conversion_errors = 0

    for i, dt in enumerate(synthetic_data):
        try:
            tokens = tokenize_text(dt['text'])
            ents = [(k["entity"], k["types"]) for k in dt['entities']]
        except Exception as e:
            print(f"Error processing synthetic example {i}: {e}")
            conversion_errors += 1
            continue

        spans = []
        for entity in ents:
            entity_tokens = tokenize_text(str(entity[0]))

            # Find the start and end indices of each entity in the tokenized text
            for j in range(len(tokens) - len(entity_tokens) + 1):
                if " ".join(tokens[j:j + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                    for el in entity[1]:
                        spans.append((j, j + len(entity_tokens) - 1, el.lower().replace('_', ' ')))

        # Append the tokenized text and its corresponding named entity recognition data
        all_examples.append({"tokenized_text": tokens, "ner": spans})

    print(f"Conversion completed: {len(all_examples)} examples, {conversion_errors} errors")
    return all_examples


def validate_and_clean_ner_data(ner_data: List[Dict], valid_entity_types: List[str],
                                logger: logging.Logger = None) -> List[Dict]:
    """
    Validate and clean NER data exactly like your original function
    
    Args:
        ner_data: List of NER examples
        valid_entity_types: List of valid entity types
        logger: Optional logger
        
    Returns:
        List of cleaned NER examples
    """
    if logger:
        logger.info(f"Validating and cleaning {len(ner_data)} NER examples...")
    
    cleaned_data = []
    stats = {
        'examples_removed': 0,
        'entities_removed': 0,
        'out_of_bounds': 0,
        'invalid_order': 0,
        'invalid_types': 0,
        'empty_after_cleaning': 0
    }
    
    invalid_types_found = set()
    
    for i, example in enumerate(ner_data):
        try:
            tokenized_text = example['tokenized_text'] 
            ner = example['ner']
            text_len = len(tokenized_text)
            
            # Skip very short texts (likely errors)
            if text_len < 2:
                stats['examples_removed'] += 1
                continue
            
            cleaned_entities = []
            
            for entity in ner:
                # Check if entity has correct format [start, end, type]
                if not isinstance(entity, (list, tuple)) or len(entity) != 3:
                    stats['entities_removed'] += 1
                    continue
                    
                start, end, entity_type = entity
                
                # Check index types
                if not isinstance(start, int) or not isinstance(end, int):
                    stats['entities_removed'] += 1
                    continue
                
                # Check index order (start should not be greater than end)
                if start > end:
                    stats['invalid_order'] += 1
                    stats['entities_removed'] += 1
                    continue
                
                # Check index bounds (most critical - this was causing the crash)
                if start < 0 or end >= text_len:
                    stats['out_of_bounds'] += 1
                    stats['entities_removed'] += 1
                    continue
                
                # Check for extremely long spans (likely errors)
                if (end - start) > 15:  # More than 15 tokens is suspicious
                    stats['entities_removed'] += 1
                    continue
                
                # Check entity type validity
                if entity_type not in valid_entity_types:
                    invalid_types_found.add(entity_type)
                    stats['invalid_types'] += 1
                    stats['entities_removed'] += 1
                    continue
                
                # If we get here, entity is valid
                cleaned_entities.append([start, end, entity_type])
            
            # Only keep examples with at least one valid entity
            if len(cleaned_entities) > 0:
                cleaned_data.append({
                    "tokenized_text": tokenized_text,
                    "ner": cleaned_entities
                })
            else:
                stats['empty_after_cleaning'] += 1
                stats['examples_removed'] += 1
                
        except Exception as e:
            if logger:
                logger.warning(f"Error validating example {i}: {e}")
            stats['examples_removed'] += 1
            continue
    
    # Log cleaning results
    if logger:
        logger.info(f"Validation completed:")
        logger.info(f"  Original examples: {len(ner_data)}")
        logger.info(f"  Cleaned examples: {len(cleaned_data)}")
        logger.info(f"  Examples removed: {stats['examples_removed']}")
        logger.info(f"  Entities removed: {stats['entities_removed']}")
        
        if stats['out_of_bounds'] > 0:
            logger.info(f"  - Out of bounds indices: {stats['out_of_bounds']}")
        if stats['invalid_order'] > 0:
            logger.info(f"  - Invalid index order: {stats['invalid_order']}")
        if stats['invalid_types'] > 0:
            logger.info(f"  - Invalid entity types: {stats['invalid_types']}")
            
        if invalid_types_found:
            logger.info(f"  Invalid types found: {sorted(invalid_types_found)}")
    
    return cleaned_data


def get_ner_statistics(ner_data: List[Dict], entity_types: List[str] = None) -> Dict:
    """
    Get comprehensive statistics for NER data (works for any NER format data)
    Replaces both get_dataset_stats and get_ner_statistics from original code
    
    Args:
        ner_data: List of NER examples in {"tokenized_text": [...], "ner": [...]} format
        entity_types: Optional list of entity types for comprehensive stats
        
    Returns:
        Dictionary with statistics
    """
    if not ner_data:
        return {}
    
    # Basic statistics
    lengths = [len(d["tokenized_text"]) for d in ner_data]
    len_ner = [len(d["ner"]) for d in ner_data]
    unique_entities = [str(n[2]).lower() for d in ner_data for n in d["ner"]]
    
    entity_counts = Counter(unique_entities)
    
    stats = {
        'total_examples': len(ner_data),
        'avg_num_tokens': sum(lengths) / len(lengths),
        'avg_num_entities': sum(len_ner) / len(len_ner),
        'total_entities': len(unique_entities),
        'unique_entity_types': len(set(unique_entities)),
        'entity_type_counts': entity_counts
    }
    
    # If entity_types provided, add comprehensive type coverage
    if entity_types:
        type_coverage = {}
        for etype in entity_types:
            type_coverage[etype] = entity_counts.get(etype, 0)
        stats['entity_type_coverage'] = type_coverage
    
    return stats


def log_ner_statistics(ner_data: List[Dict], dataset_name: str, logger: logging.Logger,
                      entity_types: List[str] = None):
    """
    Log NER statistics (unified function for all NER data)
    Replaces both log_dataset_stats and log_ner_statistics from original code

    Args:
        ner_data: List of NER examples
        dataset_name: Name for logging
        logger: Logger instance
        entity_types: Optional list of entity types
    """
    if not ner_data:
        logger.info(f"{dataset_name}: No data to analyze")
        return

    stats = get_ner_statistics(ner_data, entity_types)

    logger.info(f"{dataset_name} Dataset Statistics:")
    logger.info(f"  Total examples: {stats['total_examples']}")
    logger.info(f"  Avg num tokens: {stats['avg_num_tokens']:.2f}")
    logger.info(f"  Avg num entities: {stats['avg_num_entities']:.2f}")
    logger.info(f"  Total entities: {stats['total_entities']}")
    logger.info(f"  Unique entity types: {stats['unique_entity_types']}")

    # Top entity types
    top_types = stats['entity_type_counts'].most_common(5)
    logger.info(f"  Top entity types: {top_types}")

    # Coverage info if entity_types provided
    if entity_types and 'entity_type_coverage' in stats:
        missing_types = [t for t in entity_types if stats['entity_type_coverage'][t] == 0]
        if missing_types:
            logger.info(f"  Missing entity types: {missing_types}")


def prepare_texts_for_inference(data: List[Dict]) -> Tuple[List[str], List[List] | None]:
    """
    Prepare texts from NER format data for model inference.
    Extracts tokenized text and optional ground truth labels.

    Args:
        data: List of NER examples with "tokenized_text" and optionally "ner" fields

    Returns:
        Tuple of (texts, ground_truths) where:
        - texts: List of joined text strings ready for inference
        - ground_truths: List of NER spans if available, None otherwise
    """
    texts = []
    has_ground_truth = all("ner" in example for example in data)
    ground_truths = [] if has_ground_truth else None

    for example in data:
        # Join tokenized text into single string
        text = " ".join(example["tokenized_text"])
        texts.append(text)

        # Extract ground truth if available
        if has_ground_truth and "ner" in example:
            ground_truths.append(example["ner"])

    return texts, ground_truths



def create_mixed_training_data(examples, llm_labels, gt_ratio):
    """
    Create training data with specified GT/LLM ratio

    Args:
        examples: Original examples with GT labels
        llm_labels: LLM-generated labels for same examples
        gt_ratio: Percentage of examples to use GT labels (0-100)

    Returns:
        List of training examples with mixed labels
    """
    n_examples = len(examples)
    n_gt = int(n_examples * gt_ratio / 100)

    # Randomly select which examples get GT labels
    gt_indices = random.sample(range(n_examples), n_gt)

    mixed_data = []
    for i, (example, llm_example) in enumerate(zip(examples, llm_labels)):
        if i in gt_indices:
            # Use GT labels
            mixed_data.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]
            })
        else:
            # Use LLM labels
            mixed_data.append({
                "tokenized_text": llm_example["tokenized_text"],
                "ner": llm_example["ner"]
            })

    return mixed_data

