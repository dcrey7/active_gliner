"""
Data loading utilities - simplified version of your load_mit_dataset function
"""

import json
import os
from typing import List, Tuple, Dict, Any
import logging


def load_mit_dataset(data_path: str, labels_path: str, split_name: str = "train"):
    """
    Load and process MIT dataset exactly like your original function
    
    Args:
        data_path: Path to the data JSON file
        labels_path: Path to the labels JSON file  
        split_name: Name of the split (for logging)
        
    Returns:
        Tuple of (processed_data, entity_types)
    """
    print(f"Loading {split_name} data from: {data_path}")
    
    # Load data and labels
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    processed_data = []
    
    # Process each item exactly like your original code
    for item in data:
        words = item['sentence'].split()
        entities = []
        
        for entity in item['entities']:
            start_char, end_char = entity['pos']
            char_count = 0
            start_word = None
            end_word = None
            
            # Find word positions from character positions
            for i, word in enumerate(words):
                word_length = len(word)
                if char_count == start_char:
                    start_word = i
                if char_count + word_length == end_char:
                    end_word = i
                    break
                char_count += word_length + 1
            
            if start_word is not None and end_word is not None:
                entities.append((start_word, end_word, entity['type'].lower()))
        
        processed_data.append({
            "tokenized_text": words,
            "ner": entities
        })
    
    # Process entity types
    entity_types = [label.lower() for label in labels]
    
    print(f"Processed {len(processed_data)} examples")
    print(f"Entity types: {entity_types}")
    
    return processed_data, entity_types


def load_json_file(file_path: str) :
    """
    Load JSON file from given path
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)



def load_dataset_from_config(config_settings, split="train"):
    """
    Load dataset using settings object
    
    Args:
        config_settings: Settings object with data paths
        split: "train" or "test"
        
    Returns:
        Tuple of (processed_data, entity_types)
    """
    if split == "train":
        data_file = config_settings.train_file
    elif split == "test":
        data_file = config_settings.test_file
    else:
        raise ValueError(f"Unknown split: {split}")
    
    data_path = config_settings.data_path / data_file
    labels_path = config_settings.data_path / config_settings.labels_file
    
    return load_mit_dataset(str(data_path), str(labels_path), split)


# Stats functions moved to data/transforms.py to avoid duplication

def save_json_file(data: Any, file_path: str):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to the output JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {file_path}")