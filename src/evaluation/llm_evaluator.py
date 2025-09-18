"""
LLM Evaluator for Direct NER Evaluation
Evaluates Mistral/Gemma predictions directly using existing GLiNER evaluation pipeline
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

from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from utils.logging import get_logger


class LLMEvaluator:
    """Evaluate LLM NER predictions directly against test set"""
    
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
    
    def predict_all(self, test_data: List[Dict], entity_types: List[str]) -> List[Dict]:
        """
        Generate predictions for all test examples
        
        Args:
            test_data: Test dataset with tokenized_text and ner
            entity_types: List of entity types
            
        Returns:
            List of cleaned NER predictions
        """
        self.logger.info(f"Generating {self.model_type.upper()} predictions for {len(test_data)} examples...")
        
        all_synthetic_outputs = []
        
        for i, example in enumerate(tqdm(test_data, desc="LLM Labeling")):
            tokenized_text = example['tokenized_text']
            prompt = self._create_prompt(tokenized_text, entity_types)
            
            # Generate prediction
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
                    all_synthetic_outputs.append(js)
                    success = True
                    break
                    
                except Exception as e:
                    if attempt == max_retries:
                        self.logger.error(f"Failed to parse example {i+1} after {max_retries+1} attempts")
                        # Add empty prediction for failed examples
                        all_synthetic_outputs.append({
                            "text": " ".join(tokenized_text),
                            "entities": []
                        })
                        break
        
        self.logger.info(f"Generated {len(all_synthetic_outputs)} predictions")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(all_synthetic_outputs)
        self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
        
        # Clean and validate using existing pipeline
        cleaned_predictions = validate_and_clean_ner_data(ner_formatted_data, entity_types, self.logger)
        self.logger.info(f"Final cleaned predictions: {len(cleaned_predictions)}")
        
        return cleaned_predictions


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


class FakeModelWrapper:
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