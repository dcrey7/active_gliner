"""
LLM Evaluator for Direct NER Evaluation WITH CACHING
Evaluates Mistral/Gemma predictions directly using existing GLiNER evaluation pipeline
FIXED: Custom cleaning that preserves indices for correct cache alignment
"""

import ollama
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import logging

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from data.transforms import convert_synthetic_to_ner_format
from utils.logging import get_logger


class LLMEvaluator:
    """Evaluate LLM NER predictions directly against test set WITH CACHING"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "gemma3:12b", model_path: str = None):
        """
        Initialize LLM evaluator
        
        Args:
            model_type: "ollama" or "mistral" 
            model_name: Model name (for ollama) or path (for mistral)
            model_path: Path to mistral model folder
        """
        self.model_type = model_type
        self.model_name = model_name
        self.logger = get_logger("ActiveLearning")
        
        if model_type == "mistral":
            # Load Mistral inference
            if model_path is None:
                self.model_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
            else:
                self.model_path = Path(model_path)
            
            self.logger.info(f"Loading Mistral model from: {self.model_path}")
            tokenizer_path = self.model_path / "tokenizer.model.v3"
            self.tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
            self.model = Transformer.from_folder(self.model_path)
            self.logger.info("Mistral model loaded successfully")
        else:
            # Ollama setup
            self.logger.info(f"Using Ollama model: {model_name}")
    
    def _create_prompt(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """Create labeling prompt"""
        text = " ".join(tokenized_text)
        
        prompt = f"""CRITICAL: Label the given text with named entities.

**Entity Types to Use (ONLY these types):**
"""
        for entity_type in entity_types:
            prompt += f"- {entity_type}: Entities of type {entity_type}\n"
        
        prompt += f"""
**Text to Label:**
{text}

**MANDATORY Output Format:**
{{
  "text": "{text}",
  "entities": [
    {{"entity": "exact entity name", "types": ["entity type"]}},
    ...
  ]
}}

CRITICAL: Generate ONLY the JSON format above.
"""
        return prompt
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate prediction using Ollama"""
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                'top_k': 50,
                'top_p': 0.8,
                'num_predict': 500,
                'temperature': 0.3,
            }
        )
        return response['response'].strip()
    
    def _generate_with_mistral(self, prompt: str) -> str:
        """Generate prediction using Mistral inference"""
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate(
            [tokens], self.model, max_tokens=500, temperature=0.3,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        return result
    
    def _clean_predictions_preserve_indices(self, ner_data: List[Dict], valid_entity_types: List[str]) -> List[Dict]:
        """
        Clean NER data while preserving ALL examples for correct cache alignment
        This is the critical fix - never removes examples, just cleans their entities
        """
        cleaned_data = []
        stats = {
            'examples_processed': 0,
            'entities_removed': 0,
            'out_of_bounds': 0,
            'invalid_order': 0,
            'invalid_types': 0,
            'examples_with_empty_entities': 0
        }
        
        invalid_types_found = set()
        
        for i, example in enumerate(ner_data):
            try:
                tokenized_text = example.get('tokenized_text', [])
                ner = example.get('ner', [])
                text_len = len(tokenized_text)
                
                cleaned_entities = []
                
                # Skip validation for empty text, but still preserve the example
                if text_len >= 2:
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
                
                # CRITICAL: Always append example, even if it has no valid entities
                cleaned_data.append({
                    "tokenized_text": tokenized_text,
                    "ner": cleaned_entities
                })
                
                if len(cleaned_entities) == 0:
                    stats['examples_with_empty_entities'] += 1
                
                stats['examples_processed'] += 1
                        
            except Exception as e:
                self.logger.warning(f"Error validating example {i}: {e}")
                # Still append the example to preserve indexing
                cleaned_data.append({
                    "tokenized_text": example.get('tokenized_text', []),
                    "ner": []
                })
                stats['examples_processed'] += 1
                stats['examples_with_empty_entities'] += 1
        
        # Log cleaning results
        self.logger.info(f"Index-preserving validation completed:")
        self.logger.info(f"  Examples processed: {stats['examples_processed']}")
        self.logger.info(f"  Examples with empty entities: {stats['examples_with_empty_entities']}")
        self.logger.info(f"  Entities removed: {stats['entities_removed']}")
        
        if stats['out_of_bounds'] > 0:
            self.logger.info(f"  - Out of bounds indices: {stats['out_of_bounds']}")
        if stats['invalid_order'] > 0:
            self.logger.info(f"  - Invalid index order: {stats['invalid_order']}")
        if stats['invalid_types'] > 0:
            self.logger.info(f"  - Invalid entity types: {stats['invalid_types']}")
            
        if invalid_types_found:
            self.logger.info(f"  Invalid types found: {sorted(invalid_types_found)}")
        
        # CRITICAL: Verify index preservation
        assert len(cleaned_data) == len(ner_data), f"Index preservation failed: {len(ner_data)} -> {len(cleaned_data)}"
        
        return cleaned_data
    
    def predict_all(self, test_data: List[Dict], entity_types: List[str], 
                   evaluation_cache: List[Dict], verbose: bool = True) -> List[Dict]:
        """
        Generate predictions for test examples WITH CACHING
        Similar to gemma_labeler.py generate() method
        
        Args:
            test_data: Test dataset with tokenized_text and ner
            entity_types: List of entity types
            evaluation_cache: Cache list that persists evaluation data
            verbose: Whether to show progress
            
        Returns:
            List of cleaned NER predictions (same length as test_data)
        """
        if verbose:
            self.logger.info("="*60)
            self.logger.info("LLM EVALUATION WITH CACHING")
            self.logger.info("="*60)
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Test examples available: {len(test_data)}")
            self.logger.info(f"Target evaluations: {len(test_data)}")
            self.logger.info(f"Cached evaluations: {len(evaluation_cache)}")
            self.logger.info("="*60)
        
        # Calculate how many new evaluations we actually need
        if len(evaluation_cache) >= len(test_data):
            if verbose:
                self.logger.info(f"Using {len(test_data)} evaluations from cache (no evaluation needed)")
            return evaluation_cache[:len(test_data)]
        
        no_new_evals_needed = len(test_data) - len(evaluation_cache)
        if verbose:
            self.logger.info(f"Need to evaluate {no_new_evals_needed} new examples ({len(evaluation_cache)} already cached)")
        
        # Check if we have enough examples to evaluate
        available_examples = len(test_data) - len(evaluation_cache)
        if available_examples < no_new_evals_needed:
            self.logger.warning(f"Not enough examples! Need {no_new_evals_needed}, have {available_examples}")
            no_new_evals_needed = available_examples
        
        synthetic_outputs = []
        
        # Evaluation loop with immediate retry logic
        for i in tqdm(range(no_new_evals_needed), desc="LLM Evaluating", disable=not verbose):
            # Get next example to evaluate (skip already cached ones)
            example_idx = len(evaluation_cache) + i
            example = test_data[example_idx]
            tokenized_text = example['tokenized_text']
            
            # Create evaluation prompt
            prompt = self._create_prompt(tokenized_text, entity_types)
            
            # Immediate retry logic
            max_retries = 3
            success = False
            
            for attempt in range(max_retries + 1):
                try:
                    if self.model_type == "mistral":
                        response_text = self._generate_with_mistral(prompt)
                    else:
                        response_text = self._generate_with_ollama(prompt)
                    
                    # Clean and parse JSON
                    if '```json' in response_text:
                        start_idx = response_text.find('```json') + 7
                        end_idx = response_text.find('```', start_idx)
                        if end_idx != -1:
                            response_text = response_text[start_idx:end_idx].strip()
                    elif '{' in response_text and '}' in response_text:
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        response_text = response_text[start_idx:end_idx]
                    
                    import json
                    js = json.loads(response_text)
                    synthetic_outputs.append(js)
                    success = True
                    
                    if verbose:
                        self.logger.info(f"Evaluated example {i+1}: {tokenized_text} -> {js.get('entities', [])}")
                    
                    break
                    
                except Exception as e:
                    if attempt == max_retries:
                        self.logger.error(f"Failed to evaluate example {i+1} after {max_retries+1} attempts")
                        # Add empty prediction for failed examples
                        synthetic_outputs.append({
                            "text": " ".join(tokenized_text),
                            "entities": []
                        })
                        break
        
        if verbose:
            self.logger.info(f"Successfully evaluated {len(synthetic_outputs)}/{no_new_evals_needed} new examples")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
        
        # CRITICAL FIX: Use custom cleaning that preserves indices
        cleaned_predictions = self._clean_predictions_preserve_indices(ner_formatted_data, entity_types)
        if verbose:
            self.logger.info(f"Final cleaned predictions: {len(cleaned_predictions)} (index-preserved)")
        
        # Add new cleaned data to cache
        evaluation_cache.extend(cleaned_predictions)
        
        if verbose:
            self.logger.info(f"Cache updated: {len(evaluation_cache)} total evaluations")
            self.logger.info("="*60)
        
        # Return exactly len(test_data) (from cache + newly evaluated)
        return evaluation_cache[:len(test_data)]


def convert_ner_to_gliner_format(cleaned_predictions: List[Dict]) -> List[List[Dict]]:
    """
    Convert cleaned NER predictions to GLiNER prediction format
    
    Args:
        cleaned_predictions: List of {"tokenized_text": [...], "ner": [(start, end, type)]}
        
    Returns:
        GLiNER format: List of lists of prediction dictionaries
    """
    gliner_predictions = []
    
    for example in cleaned_predictions:
        tokenized_text = example['tokenized_text']
        ner_spans = example['ner']
        
        example_predictions = []
        
        for start_token, end_token, entity_type in ner_spans:
            # Convert token positions to character positions
            char_start = 0
            for i in range(start_token):
                char_start += len(tokenized_text[i]) + 1  # +1 for space
            
            char_end = char_start
            for i in range(start_token, end_token + 1):
                char_end += len(tokenized_text[i])
                if i < end_token:
                    char_end += 1  # +1 for space
            
            # Get entity text
            entity_text = " ".join(tokenized_text[start_token:end_token+1])
            
            # Create GLiNER format prediction
            example_predictions.append({
                'start': char_start,
                'end': char_end,
                'label': entity_type,
                'text': entity_text,
                'score': 1.0  # LLM doesn't provide confidence scores
            })
        
        gliner_predictions.append(example_predictions)
    
    return gliner_predictions


class LLMModelWrapper:
    """Wrapper to make LLM predictions look like GLiNER model for evaluation"""
    
    def __init__(self, gliner_predictions: List[List[Dict]]):
        self.predictions = gliner_predictions
    
    def run(self, texts: List[str], entity_types: List[str], **kwargs) -> List[List[Dict]]:
        """Return pre-computed predictions"""
        return self.predictions
    
    def evaluate(self, test_data: List[Dict], entity_types: List[str], **kwargs) -> tuple:
        """Dummy GLiNER evaluate method - returns fake results"""
        # Calculate simple F1 for GLiNER compatibility
        total_tp = total_fp = total_fn = 0
        
        for i, example in enumerate(test_data):
            ground_truth = set((start, end, label) for start, end, label in example['ner'])
            
            if i < len(self.predictions):
                predictions = set()
                for pred in self.predictions[i]:
                    # Convert char positions back to token positions for comparison
                    # This is approximate - in real use enhanced_evaluate handles this properly
                    predictions.add((0, 0, pred['label']))  # Dummy positions
                
                tp = len(ground_truth & predictions)
                fp = len(predictions - ground_truth)
                fn = len(ground_truth - predictions)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
        
        # Calculate F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {}, f1


class LLMEvaluationPipeline:
    """Simple evaluation pipeline wrapper for compatibility"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "gemma3:12b"):
        self.evaluator = LLMEvaluator(model_type=model_type, model_name=model_name)
    
    def evaluate_dataset(self, test_data: List[Dict], entity_types: List[str], 
                        evaluation_cache: List[Dict] = None) -> List[List[Dict]]:
        """Evaluate dataset with caching support"""
        if evaluation_cache is None:
            evaluation_cache = []
        
        # Get cleaned predictions with preserved indices
        cleaned_predictions = self.evaluator.predict_all(
            test_data=test_data,
            entity_types=entity_types,
            evaluation_cache=evaluation_cache,
            verbose=True
        )
        
        # Convert to GLiNER format
        gliner_predictions = convert_ner_to_gliner_format(cleaned_predictions)
        
        # Verify final indexing
        assert len(gliner_predictions) == len(test_data), "Final index preservation failed"
        
        return gliner_predictions