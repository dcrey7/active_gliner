"""
Simple Label Generator for Existing Text
Sweet, Simple, and Solid - Generate Labels for Low Confidence Examples

Usage:
    generator = LabelGenerator()
    labeled_data = generator.generate(
        low_n_examples=low_confidence_examples,
        num_samples=100,
        entity_types=["PERSON", "ORG", "LOCATION"],
        label_cache=cache_list
    )
"""

import ollama
import json
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from utils.logging import get_logger


class LabelGenerator:
    """Simple label generator for existing text using Ollama"""
    
    def __init__(self, model_name: str = "gemma3:12b"):
        """
        Initialize the label generator
        
        Args:
            model_name: Ollama model to use for labeling
        """
        self.model_name = model_name
        self.logger = get_logger("ActiveLearning")
        self.logger.info(f"Label Generator model: {self.model_name}")

    def _create_prompt(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """
        Create simple labeling prompt
        
        Args:
            tokenized_text: Text tokens to label
            entity_types: Entity types to identify
            
        Returns:
            Formatted prompt string
        """
        text = " ".join(tokenized_text)
        
        prompt = f"""CRITICAL: You are an expert at Name Entity Reconginition information extractor. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled with the specified entity types
- Use ONLY the provided entity types

**Entity Types to Use (ONLY these types):**
"""
        
        # Add entity types dynamically
        for entity_type in entity_types:
            prompt += f"- {entity_type}: Entities of type {entity_type}\n"
        
        prompt += f"""
**Text to Label:**
{text}

**CRITICAL Requirements:**
- MUST use entities from these types ONLY: {', '.join(entity_types)}
- Identify ALL relevant entities in the text
- Use clear, exact entity names as they appear in text
- Do not modify or paraphrase entity names
- Include entities even if you're not 100% certain

**MANDATORY Output Format:**
{{
  "text": "{text}",
  "entities": [
    {{"entity": "exact entity name", "types": ["entity type"]}},
    ...
  ]
}}

CRITICAL: Generate ONLY the JSON format above. Start immediately with the JSON object.
"""
        
        return prompt
    
    def generate(self, low_n_examples: List[Dict], num_samples: int, 
                entity_types: List[str], label_cache: List[Dict], 
                verbose: bool = True) -> List[Dict]:
        """
        Generate labels for low confidence examples
        
        Args:
            low_n_examples: Low confidence examples with tokenized_text
            num_samples: Number of examples to label
            entity_types: Entity types to identify
            label_cache: Cache list that persists labeled data
            verbose: Whether to show progress
            
        Returns:
            List of cleaned NER formatted examples
        """
        if verbose:
            self.logger.info("="*60)
            self.logger.info("LABEL GENERATION")
            self.logger.info("="*60)
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Low confidence examples available: {len(low_n_examples)}")
            self.logger.info(f"Target labels: {num_samples}")
            self.logger.info(f"Cached labels: {len(label_cache)}")
            self.logger.info("="*60)
        
        # Calculate how many new labels we actually need
        if len(label_cache) >= num_samples:
            if verbose:
                self.logger.info(f"Using {num_samples} labels from cache (no generation needed)")
            return label_cache[:num_samples]
        
        no_new_labels_needed = num_samples - len(label_cache)
        if verbose:
            self.logger.info(f"Need to generate {no_new_labels_needed} new labels ({len(label_cache)} already cached)")
        
        # Check if we have enough examples to label
        available_examples = len(low_n_examples) - len(label_cache)
        if available_examples < no_new_labels_needed:
            self.logger.warning(f"Not enough examples! Need {no_new_labels_needed}, have {available_examples}")
            no_new_labels_needed = available_examples
        
        synthetic_outputs = []
        
        # Generation loop with immediate retry logic
        for i in tqdm(range(no_new_labels_needed), desc="Labeling", disable=not verbose):
            # Get next example to label (skip already cached ones)
            example_idx = len(label_cache) + i
            example = low_n_examples[example_idx]
            tokenized_text = example['tokenized_text']
            
            # Create labeling prompt
            prompt = self._create_prompt(tokenized_text, entity_types)
            
            # Immediate retry logic
            max_retries = 3
            success = False
            
            for attempt in range(max_retries + 1):
                try:
                    # Generate with Ollama
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={
                            'top_k': 50,
                            'top_p': 0.8,
                            'num_predict': 500,
                            'temperature': 0.3,  # Lower temperature for consistency
                        }
                    )
                    
                    # Parse JSON response
                    response_text = response['response'].strip()
                    
                    # Clean up response (remove any markdown formatting)
                    if response_text.startswith('```json'):
                        response_text = response_text.replace('```json', '').replace('```', '').strip()
                    
                    # Find JSON in response if wrapped with other text
                    if '{' in response_text and '}' in response_text:
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        response_text = response_text[start_idx:end_idx]
                    
                    js = json.loads(response_text)
                    synthetic_outputs.append(js)
                    success = True
                    
                    # if verbose and i % 20 == 0:
                    self.logger.info(f"Labeled example {i+1} {tokenized_text}: {(js.get('entities', []))} ")
                    
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        self.logger.warning(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")
                            
                except Exception as e:
                    error_msg = f"Labeling failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        self.logger.error(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")
        
        if verbose:
            self.logger.info(f"Successfully generated {len(synthetic_outputs)}/{no_new_labels_needed} labels")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
        
        # Clean and validate using existing pipeline
        cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types)
        
        if verbose:
            self.logger.info(f"Final cleaned examples: {len(cleaned_data)}")
        
        # Add new cleaned data to cache
        label_cache.extend(cleaned_data)
        
        if verbose:
            self.logger.info(f"Cache updated: {len(label_cache)} total labeled examples")
            
            # Show some stats for all data (cached + new)
            if len(label_cache) > 0:
                avg_entities = sum(len(ex['ner']) for ex in label_cache) / len(label_cache)
                self.logger.info(f"Average entities per example: {avg_entities:.1f}")
                
                # Entity type distribution
                entity_counts = {}
                for ex in label_cache:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                self.logger.info(f"Entity distribution: {entity_counts}")
            
            self.logger.info("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return label_cache[:num_samples]