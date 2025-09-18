"""
Simple Label Generator with Mistral Inference
Generate Labels for Low Confidence Examples using Mistral Inference

Usage:
    generator = MistralLabelGenerator()
    labeled_data = generator.generate(
        low_n_examples=low_confidence_examples,
        num_samples=100,
        entity_types=["PERSON", "ORG", "LOCATION"],
        label_cache=cache_list
    )
"""

import json
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from utils.logging import get_logger
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from config.settings import Settings

settings = Settings()
settings.setup()
logger = get_logger("ActiveLearning")
set_all_seeds(seed=settings.global_seed, logger=logger)
device = setup_device(logger=logger)


class LabelGenerator:
    """Simple label generator for existing text using Mistral Inference"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the label generator
        
        Args:
            model_path: Path to the Mistral model folder (default: ~/mistral_models/7B-Instruct-v0.3)
        """
        if model_path is None:
            self.model_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
        else:
            self.model_path = Path(model_path)
        
        logger.info(f"Loading Mistral model from: {self.model_path}")
        
        # Initialize tokenizer and model
        tokenizer_path = self.model_path / "tokenizer.model.v3"
        self.tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
        self.model = Transformer.from_folder(self.model_path)
        self.model_name = "MISTRAL 7B v0.3 (mistrl inf)"
        logger.info(f"Mistral label generator loaded successfully")
        logger.info(f"Label Generator model: {self.model_name}")
    
    def _create_prompt(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """
        Create simple labeling prompt for Mistral
        
        Args:
            tokenized_text: Text tokens to label
            entity_types: Entity types to identify
            
        Returns:
            Formatted prompt string
        """
        text = " ".join(tokenized_text)
        
        prompt = f"""CRITICAL: Label the given text with named entities.

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
    
    def _generate_single_label(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> Tuple[str, int, int]:
        """
        Generate a single label using Mistral inference with exact token tracking
        
        Args:
            prompt: The labeling prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        # Create chat completion request
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        
        # Encode the prompt and count input tokens
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        input_token_count = len(tokens)
        
        # Generate response
        out_tokens, _ = generate(
            [tokens], 
            self.model, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        
        # Count output tokens (out_tokens[0] is already just the generated tokens)
        output_token_count = len(out_tokens[0])
        
        # Decode the response
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        
        return result, input_token_count, output_token_count
    
    def generate(self, low_n_examples: List[Dict], num_samples: int, 
                entity_types: List[str], label_cache: List[Dict], 
                verbose: bool = True) -> List[Dict]:
        """
        Generate labels for low confidence examples using Mistral inference
        
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
            logger.info("="*60)
            logger.info("MISTRAL LABEL GENERATION")
            logger.info("="*60)
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Entity types: {entity_types}")
            logger.info(f"Low confidence examples available: {len(low_n_examples)}")
            logger.info(f"Target labels: {num_samples}")
            logger.info(f"Cached labels: {len(label_cache)}")
            logger.info("="*60)
        
        # Initialize token tracking
        input_tokens_list = []
        output_tokens_list = []
        
        # Model configuration for Mistral 7B
        model_context_limit = 32768  # Mistral 7B context window
        model_output_limit = 500     # Max generation tokens for labeling
        model_limits = (model_context_limit, model_output_limit)
        
        # Calculate how many new labels we actually need
        if len(label_cache) >= num_samples:
            if verbose:
                logger.info(f"Using {num_samples} labels from cache (no generation needed)")
                
            avg_entities = sum(len(ex['ner']) for ex in label_cache[:num_samples]) / len(label_cache[:num_samples])
            
            token_metrics = {
                'avg_input_tokens': 0,  # No generation, so no input
                'model_input_output': model_limits,
                'avg_output_tokens': 0,  # No generation, so no output
            }
            
            if verbose:
                logger.info(f"Token metrics (cached): model_limits={model_limits}, no generation needed")
            
            return label_cache[:num_samples]
        
        no_new_labels_needed = num_samples - len(label_cache)
        if verbose:
            logger.info(f"Need to generate {no_new_labels_needed} new labels ({len(label_cache)} already cached)")
        
        # Check if we have enough examples to label
        available_examples = len(low_n_examples) - len(label_cache)
        if available_examples < no_new_labels_needed:
            logger.warning(f"Not enough examples! Need {no_new_labels_needed}, have {available_examples}")
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
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Generate with Mistral inference and get exact token counts
                    response_text, input_tokens, output_tokens = self._generate_single_label(
                        prompt=prompt,
                        max_tokens=model_output_limit,
                        temperature=0.3
                    )
                    
                    # Store exact token counts
                    input_tokens_list.append(input_tokens)
                    output_tokens_list.append(output_tokens)
                    
                    # Clean up response (remove any markdown formatting and extract JSON)
                    response_text = response_text.strip()
                    
                    # Find the JSON part in the response
                    if '```json' in response_text:
                        start_idx = response_text.find('```json') + 7
                        end_idx = response_text.find('```', start_idx)
                        if end_idx != -1:
                            response_text = response_text[start_idx:end_idx].strip()
                    elif '{' in response_text:
                        # Find the first { and try to extract JSON from there
                        start_idx = response_text.find('{')
                        response_text = response_text[start_idx:].strip()
                        # Try to find the matching closing brace
                        brace_count = 0
                        end_idx = -1
                        for idx, char in enumerate(response_text):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = idx + 1
                                    break
                        if end_idx != -1:
                            response_text = response_text[:end_idx]
                    
                    js = json.loads(response_text)
                    synthetic_outputs.append(js)
                    success = True
                    
                    if verbose:
                        logger.info(f"Labeled example {i+1} {tokenized_text}: {js.get('entities', [])}")
                    
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        logger.warning(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")
                        # Add zeros for failed attempts (no token data available)
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
                except Exception as e:
                    error_msg = f"Generation failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        logger.error(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")
                        # Add zeros for failed attempts (no token data available)
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
                
            if verbose and success and i % 5 == 0:
                # Log token metrics every 5 samples using EXACT counts
                avg_input_so_far = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
                avg_output_so_far = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
                logger.info(f"Generated {i+1}/{no_new_labels_needed} labels...")
                logger.info(f"EXACT Token metrics: avg_input={avg_input_so_far:.0f}, avg_output={avg_output_so_far:.0f}, limits={model_limits}")
                sys.stdout.flush()
        
        if verbose:
            logger.info(f"Successfully generated {len(synthetic_outputs)}/{no_new_labels_needed} labels")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
        
        # Clean and validate using existing pipeline
        cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types)
        
        if verbose:
            logger.info(f"Final cleaned examples: {len(cleaned_data)}")
        
        # Add new cleaned data to cache
        label_cache.extend(cleaned_data)
        
        # Calculate average entities
        samples_for_stats = label_cache[:num_samples]
        avg_entities = sum(len(ex['ner']) for ex in samples_for_stats) / len(samples_for_stats) if samples_for_stats else 0
        
        # Calculate token metrics
        token_metrics = {
            'avg_input_tokens': sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0,
            'model_input_output': model_limits,
            'avg_output_tokens': sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0,
        }
        
        if verbose:
            logger.info(f"Cache updated: {len(label_cache)} total labeled examples")
            logger.info("="*60)
            
            # Show token metrics
            logger.info(f"TOKEN METRICS (EXACT FROM MISTRAL TOKENIZER):")
            logger.info(f"   Average input tokens: {token_metrics['avg_input_tokens']:.0f} (EXACT)")
            logger.info(f"   Model limits (input,output): {token_metrics['model_input_output']}")
            logger.info(f"   Average output tokens: {token_metrics['avg_output_tokens']:.0f} (EXACT)")
            
            # Warning if approaching limits
            if token_metrics['avg_input_tokens'] > model_context_limit * 0.9:
                logger.warning(f"WARNING: Input tokens approaching context limit!")
            if token_metrics['avg_output_tokens'] > model_output_limit * 0.9:
                logger.warning(f"WARNING: Output tokens approaching generation limit!")
            
            # Show some stats for all data (cached + new)
            if samples_for_stats:
                logger.info(f"Average entities per example: {avg_entities:.1f}")
                
                # Entity type distribution
                entity_counts = {}
                for ex in samples_for_stats:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                logger.info(f"Entity distribution: {entity_counts}")
            logger.info("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return label_cache[:num_samples]