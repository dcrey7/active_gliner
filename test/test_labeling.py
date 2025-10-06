#!/usr/bin/env python3
"""
Confidence Analysis Script 2: Fine-tuning Performance Analysis
Tests GLiNER fine-tuned on LLM labels vs GT labels of worst confidence examples
Evaluates fine-tuned models on FULL MIT test set
WITH CACHING to avoid re-labeling same examples

Similar to test8_gemma.py but using worst confidence examples for training
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)

# Load low confidence examples for training
print("📂 Loading pre-saved low confidence examples...")
with open('../results/high_mse_2500_examples.json', 'r') as file:
    low_n = json.load(file)
print(f"📊 Loaded {len(low_n)} low confidence examples for training")

# Define entity types
entity_types =["genre", "year", "plot", "average ratings", "actor", "title", "song", "character", "rating", "review", "director", "trailer"] 

# Import the label generator after setting up paths
from generation import create_label_generator

# Initialize labeler and generate labels
generator = create_label_generator('ollama', model_name='gemma3:4b')
test_labels = generator.generate(
    low_n,
    num_samples=5,
    entity_types=entity_types
)

for i in test_labels:
    print(f"Generated labels number {i}")  

    # Load FULL test data for evaluation

    

    


