"""
Simple Synthetic Data Generator with Mistral Inference and Token Tracking
Sweet, Simple, and Solid - Fully Generic Approach with Exact Token Metrics

Usage:
    generator = SyntheticDataGenerator()
    data, avg_entities, token_metrics = generator.generate(
        corrected_examples=examples,
        num_samples=100,
        entity_types=["PERSON", "ORG", "LOCATION"],
        countries=["USA", "France", "Japan"],
        genres=["news articles", "reports"],
        subject="healthcare",
        syn_cache=cache_list
    )
"""

import json
import random
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from utils.logging import get_logger  # ← Add this import
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from config.settings import Settings

settings = Settings()
settings.setup()
logger = get_logger("ActiveLearning")  # ← Use existing logger, don't create new one
set_all_seeds(seed=settings.global_seed, logger=logger)
device = setup_device(logger=logger)
class SyntheticDataGenerator:
    """Simple, generic synthetic data generator using Mistral Inference with exact token tracking"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the generator
        
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
        logger.info(f"✅ Mistral model loaded successfully")
        logger.info(f"Mistral model loaded from {self.model_path}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count exact tokens using Mistral tokenizer
        
        Args:
            text: Text to tokenize
            
        Returns:
            Exact token count
        """
        # Use the tokenizer to get exact token count
        tokens = self.tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
        return len(tokens)
    
    def _create_prompt(self, corrected_examples: List[Dict], entity_types: List[str], 
                      subject: str, country: str, genre: str) -> str:
        """
        Create fully generic prompt based on user inputs
        
        Args:
            corrected_examples: Template examples showing desired format
            entity_types: Specific entity labels to use
            subject: Domain/topic for generation
            country: Country for variation
            genre: Genre/style for variation
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""CRITICAL: Generate training data for Named Entity Recognition.

**Objective:**
Generate realistic {genre} text about "{subject}" that includes clearly identified named entities.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled with the specified entity types
- Follow the exact format shown in the examples below

**Entity Types to Use (ONLY these types):**
"""
        
        # Add entity types dynamically
        for entity_type in entity_types:
            prompt += f"- {entity_type}: Entities of type {entity_type}\n"
        
        prompt += f"""
**TEMPLATE EXAMPLES:**
Here are examples showing the expected format and style for {subject}:

"""
        
        # Add corrected examples as templates
        for i, example in enumerate(corrected_examples):  # Limit to 3 examples for prompt size
            text = " ".join(example['tokenized_text'])
            entities = []
            
            # Convert NER format to JSON entities
            for start, end, label in example['ner']:
                entity_text = " ".join(example['tokenized_text'][start:end+1])
                entities.append({
                    "entity": entity_text,
                    "types": [label]
                })
            
            prompt += f"""Example {i+1}:
{{
  "text": "{text}",
  "entities": {json.dumps(entities, indent=2)}
}}

"""
        
        prompt += f"""**GENERATION TASK:**
Generate a NEW {genre} text about "{subject}" similar to the examples above but with different content.
Context: country={country}, style={genre}

**CRITICAL Requirements:**
- MUST include entities from these types ONLY: {', '.join(entity_types)}
- Create diverse expressions and formats for each entity type
- Use clear, explicit language for entity identification
- Provide sufficient context for each entity
- Make entities easily distinguishable in the text
- Content should be about: {subject}
- Style should match: {genre}
- Geographic context: {country}

**MANDATORY Output Format:**
{{
  "text": "your generated text here",
  "entities": [
    {{"entity": "entity name", "types": ["entity type"]}},
    ...
  ]
}}

CRITICAL: Generate ONLY ONE example in the specified JSON format. Start immediately with the JSON object.
"""
        
        return prompt
    
    def _generate_single_sample(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> Tuple[str, int, int]:
        """
        Generate a single sample using Mistral inference with exact token tracking
        
        Args:
            prompt: The generation prompt
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
    
    def generate(self, corrected_examples: List[Dict], num_samples: int, 
                entity_types: List[str], countries: List[str], genres: List[str], syn_cache: List[str],
                subject: str, verbose: bool = True) -> Tuple[List[Dict], float, Dict[str, Any]]:
        """
        Generate synthetic data with full user control and exact token tracking from Mistral
        
        Args:
            corrected_examples: Template examples showing desired format/style
            num_samples: Number of samples to generate
            entity_types: Specific entity labels to use (e.g., ["PERSON", "ORG"])
            countries: List of countries for variation (e.g., ["USA", "France"])
            genres: List of genres for variation (e.g., ["news", "reports"])
            syn_cache: Cache list that persists cleaned data for this corrected example set
            subject: Domain/topic (e.g., "healthcare", "finance")
            verbose: Whether to show progress
            
        Returns:
            Tuple of (cleaned NER formatted examples, avg_entities, token_metrics)
            token_metrics contains: avg_input_tokens (exact), model_input_output (limits), avg_output_tokens (exact)
        """
        if verbose:
            logger.info("="*60)
            logger.info("SYNTHETIC DATA GENERATION (MISTRAL)")
            logger.info("="*60)
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Entity types: {entity_types}")
            logger.info(f"Countries: {countries}")
            logger.info(f"Genres: {genres}")
            logger.info(f"Template examples: {len(corrected_examples)}")
            logger.info(f"Target samples: {num_samples}")
            logger.info(f"Cached samples: {len(syn_cache)}")
            logger.info("="*60)
        
        # Initialize token tracking
        input_tokens_list = []
        output_tokens_list = []
        
        # Model configuration for Mistral 7B
        model_context_limit = 32768  # Mistral 7B context window
        model_output_limit = 800     # Max generation tokens
        model_limits = (model_context_limit, model_output_limit)
        
        # Calculate how many new samples we actually need
        if len(syn_cache) >= num_samples:
            if verbose:
                logger.info(f"✅ Using {num_samples} samples from cache (no generation needed)")
            
            avg_entities = sum(len(ex['ner']) for ex in syn_cache[:num_samples]) / len(syn_cache[:num_samples])
            
            token_metrics = {
                'avg_input_tokens': 0,  # No generation, so no input
                'model_input_output': model_limits,
                'avg_output_tokens': 0,  # No generation, so no output
            }
            
            if verbose:
                logger.info(f"📊 Token metrics (cached): model_limits={model_limits}, no generation needed")
            
            return syn_cache[:num_samples], avg_entities, token_metrics
        
        no_new_syn_needed = num_samples - len(syn_cache)
        if verbose:
            logger.info(f"📝 Need to generate {no_new_syn_needed} new samples ({len(syn_cache)} already cached)")
        
        synthetic_outputs = []
        
        # Generation loop with immediate retry logic
        for i in tqdm(range(no_new_syn_needed), desc="Generating", disable=not verbose):
            # Random variation for diversity
            country = random.choice(countries)
            genre = random.choice(genres)
            
            # Create dynamic prompt
            prompt = self._create_prompt(
                corrected_examples=corrected_examples,
                entity_types=entity_types,
                subject=subject,
                country=country,
                genre=genre
            )
            
            # Immediate retry logic
            max_retries = 3
            success = False
            if verbose:
                logger.info(f"generating {i+1} sample")
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Generate with Mistral inference and get exact token counts
                    response_text, input_tokens, output_tokens = self._generate_single_sample(
                        prompt=prompt,
                        max_tokens=model_output_limit,
                        temperature=0.7
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
                    logger.info(f"used [{country,subject,genre} to get sample: {js}] ")
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"⚠️ JSON parsing failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)  # print() can use flush=True
                        sys.stdout.flush()  # Force flush for Jupyter notebooks
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                        # Add zeros for failed attempts (no token data available)
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
                except Exception as e:
                    error_msg = f"❌ Generation failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)  # print() can use flush=True
                        sys.stdout.flush()  # Force flush for Jupyter notebooks
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                        # Add zeros for failed attempts (no token data available)
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
                
            if verbose and success and i % 10 == 0:
                # Log token metrics every 10 samples using EXACT counts
                avg_input_so_far = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
                avg_output_so_far = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
                logger.info(f"\n✅ Generated {i+1}/{no_new_syn_needed} samples...")
                logger.info(f"📊 EXACT Token metrics: avg_input={avg_input_so_far:.0f}, avg_output={avg_output_so_far:.0f}, limits={model_limits}")  # ✅ FIXED: removed flush=True
                sys.stdout.flush()
        
        if verbose:
            logger.info(f"\n✅ Successfully generated {len(synthetic_outputs)}/{no_new_syn_needed} raw samples")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            logger.info(f"📝 Converted to NER format: {len(ner_formatted_data)} examples")
        
        # Clean and validate using existing pipeline
        cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types)
        
        if verbose:
            logger.info(f"🧹 Final cleaned examples: {len(cleaned_data)}")
        
        # Add new cleaned data to cache
        syn_cache.extend(cleaned_data)
        
        # Calculate average entities
        samples_for_stats = syn_cache[:num_samples]
        avg_entities = sum(len(ex['ner']) for ex in samples_for_stats) / len(samples_for_stats) if samples_for_stats else 0
        
        # Calculate token metrics
        token_metrics = {
            'avg_input_tokens': sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0,
            'model_input_output': model_limits,
            'avg_output_tokens': sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0,
        }
        
        if verbose:
            logger.info(f"💾 Cache updated: {len(syn_cache)} total samples")
            logger.info("="*60)
            
            # Show token metrics
            logger.info(f"📊 TOKEN METRICS (EXACT FROM MISTRAL TOKENIZER):")
            logger.info(f"   Average input tokens: {token_metrics['avg_input_tokens']:.0f} (EXACT)")
            logger.info(f"   Model limits (input,output): {token_metrics['model_input_output']}")
            logger.info(f"   Average output tokens: {token_metrics['avg_output_tokens']:.0f} (EXACT)")
            
            # Warning if approaching limits
            if token_metrics['avg_input_tokens'] > model_context_limit * 0.9:
                logger.info(f"⚠️  WARNING: Input tokens approaching context limit!")
            if token_metrics['avg_output_tokens'] > model_output_limit * 0.9:
                logger.info(f"⚠️  WARNING: Output tokens approaching generation limit!")
            
            # Show some stats for all data (cached + new)
            if samples_for_stats:
                logger.info(f"📊 Average entities per example: {avg_entities:.1f}")
                
                # Entity type distribution
                entity_counts = {}
                for ex in samples_for_stats:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                logger.info(f"📈 Entity distribution: {entity_counts}")
            logger.info("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return syn_cache[:num_samples], avg_entities, token_metrics