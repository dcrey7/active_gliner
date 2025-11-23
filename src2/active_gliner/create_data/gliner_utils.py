import re
from typing import List, Tuple, Optional
import time
from collections import Counter


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words and punctuation.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens

    Example:
        >>> tokenize_text("Hello, world!")
        ['Hello', ',', 'world', '!']
    """
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def char_to_word_positions(sentence: str, char_start: int, char_end: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert character positions to word token positions.

    Args:
        sentence: Original sentence
        char_start: Character start position
        char_end: Character end position

    Returns:
        (word_start_idx, word_end_idx): Tuple of word indices, or (None, None) if not found

    Example:
        >>> char_to_word_positions("what movies star bruce willis", 17, 29)
        (3, 4)  # "bruce willis" spans tokens 3-4
    """
    tokens = tokenize_text(sentence)

    word_start_idx = None
    word_end_idx = None
    char_pos = 0

    for i, token in enumerate(tokens):
        # Find the next occurrence of this token starting from char_pos
        token_start = sentence.find(token, char_pos)

        if token_start == -1:
            continue

        token_end = token_start + len(token)

        # Check if this token overlaps with the entity
        if token_start <= char_start < token_end:
            word_start_idx = i

        if token_start < char_end <= token_end:
            word_end_idx = i

        # Move past this token for next search
        char_pos = token_end

    return word_start_idx, word_end_idx


def find_all_entity_occurrences(entity_text: str, text: str) -> List[Tuple[int, int]]:
    """
    Find all start/end positions of entity_text in text

    Input: entity_text="Brad Pitt", text="Brad Pitt stars with Brad Pitt in..."
    Output: [(0, 9), (22, 31)]
    """
    occurrences = []
    start_idx = 0

    while True:
        start = text.find(entity_text, start_idx)
        if start == -1:
            break
        end = start + len(entity_text)
        occurrences.append((start, end))
        start_idx = end

    return occurrences



def analyze_entity_distribution(dataset, dataset_name="Dataset"):
    """
    Analyze entity distribution in a dataset
    
    Args:
        dataset: Formatted dataset with 'tokenized_text' and 'ner' fields
        dataset_name: Name of the dataset for display purposes
    
    Returns:
        Dict with entity statistics
    """

    
    start_time = time.time()
    
    # Count entities
    entity_counter = Counter()
    total_entities = 0
    total_tokens = 0
    
    for example in dataset:
        # Count tokens
        total_tokens += len(example['tokenized_text'])
        
        # Count entities
        for start, end, label in example['ner']:
            entity_counter[label] += 1
            total_entities += 1
    
    # Calculate percentages
    entity_stats = []
    for entity, count in entity_counter.most_common():
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        entity_stats.append({
            'entity': entity,
            f'count_{dataset_name}': count,
            f'percentage_{dataset_name}': percentage
        })
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Entity Distribution Analysis: {dataset_name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(dataset):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total entities: {total_entities:,}")
    print(f"Unique entity types: {len(entity_counter)}")
    print(f"Average entities per example: {total_entities/len(dataset):.2f}")
    print(f"\n{'Entity':<25} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*45}")
    
    # for stat in entity_stats:
    #     print(f"{stat['entity']:<25} {stat['count']:<10} {stat['percentage']:<10.2f}%")
    
    # Additional insights
    print(f"\nEntity density: {(total_entities/total_tokens)*100:.2f}% of tokens are entities")
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.2f} seconds")
    
    return entity_stats

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split



def multilabel_stratified_split(formatted_data, entity_types, test_size=0.3, val_size=0.5):
    """
    Perform multi-label stratified split for NER data
    
    Args:
        formatted_data: List of dicts with 'tokenized_text' and 'ner' fields
        entity_types: List of all possible entity types
        test_size: Proportion for test+val (default 0.3 for 70/30 split)
        val_size: Proportion of test_size for validation (default 0.5 for 15/15)
    
    Returns:
        train_data, val_data, test_data in original format
    """
    start_time = time.time()
    print(f"Starting multi-label stratified split for {len(formatted_data)} examples...")
    
    # Step 1: Create multi-label representation
    print("Creating multi-label representation...")
    
    # Extract entity types for each example
    entity_sets = []
    for example in formatted_data:
        # Get unique entity types in this example
        entities_in_example = set([label for _, _, label in example['ner']])
        entity_sets.append(list(entities_in_example))
    
    # Create binary matrix using MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=entity_types)
    y = mlb.fit_transform(entity_sets)
    
    print(f"Binary matrix shape: {y.shape}")
    print(f"Sparsity: {(1 - np.mean(y)) * 100:.1f}% zeros")
    
    # Create indices array for X
    X = np.arange(len(formatted_data)).reshape(-1, 1)
    
    # Step 2: First split - train vs (val+test)
    print(f"Performing first split: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% temp...")
    
    # Set random seed using numpy
    SEED = 42
    np.random.seed(SEED)
    
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(
        X, y, 
        test_size=test_size
    )
    
    # Step 3: Second split - val vs test
    print(f"Performing second split: {100*val_size:.0f}% val, {100*(1-val_size):.0f}% test...")
    
    X_val, y_val, X_test, y_test = iterative_train_test_split(
        X_temp, y_temp,
        test_size=(1-val_size)
    )
    
    # Step 4: Reconstruct original format
    print("Reconstructing original data format...")
    
    train_indices = X_train.flatten()
    val_indices = X_val.flatten()
    test_indices = X_test.flatten()
    
    train_data = [formatted_data[i] for i in train_indices]
    val_data = [formatted_data[i] for i in val_indices]
    test_data = [formatted_data[i] for i in test_indices]
    
    # Log results
    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    # Verify distribution
    print("\nVerifying entity distributions:")
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        entity_counts = {}
        total = 0
        for example in split_data:
            for _, _, label in example['ner']:
                entity_counts[label] = entity_counts.get(label, 0) + 1
                total += 1
        
        print(f"\n{split_name} set:")
        for entity in sorted(entity_types):
            count = entity_counts.get(entity, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {entity}: {count} ({pct:.1f}%)")
    
    return train_data, val_data, test_data

