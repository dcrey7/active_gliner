"""
Simple Label Generator for Existing Text using Cerebras API
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

import json
import sys
import time
from typing import List, Dict, Any
from tqdm import tqdm
import os
from cerebras.cloud.sdk import Cerebras
import cerebras.cloud.sdk

from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from utils.logging import get_logger

class LabelGenerator:
    """Simple label generator for existing text using Cerebras API"""
    
    def __init__(self, model_name: str = "qwen-3-235b-a22b-instruct-2507"):
        """
        Initialize the label generator
        
        Args:
            model_name: Cerebras model to use for labeling
                      Options: "qwen-3-235b-a22b-instruct-2507", "gpt-oss-120b"
        """
        self.model_name = model_name
        self.logger = get_logger("ActiveLearning")
        
        # Initialize Cerebras client
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable not set")
        
        self.client = Cerebras(
            api_key=api_key,
            max_retries=2,  # Let us handle retries manually for better control
            timeout=90.0    # Increase timeout for complex prompts
        )
        
        self.logger.info(f"Label Generator model: {self.model_name}")
        
        # Rate limiting tracking
        self.requests_per_minute = 0
        self.minute_start_time = time.time()
        self.tokens_per_minute = 0
        
        # Model limits (both models have same limits)
        self.max_requests_per_minute = 30
        self.max_tokens_per_minute = 64000 if "gpt-oss" in model_name else 60000
        self.context_limit = 65536

    def _wait_for_rate_limit(self, estimated_tokens: int = 500):
        """
        Intelligent rate limiting based on current usage
        
        Args:
            estimated_tokens: Estimated tokens for next request
        """
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.minute_start_time >= 60:
            self.requests_per_minute = 0
            self.tokens_per_minute = 0
            self.minute_start_time = current_time
        
        # Check if we need to wait for request limit
        if self.requests_per_minute >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time) + 1
            if wait_time > 0:
                self.logger.info(f"Request limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.tokens_per_minute = 0
                self.minute_start_time = time.time()
        
        # Check if we need to wait for token limit
        if self.tokens_per_minute + estimated_tokens >= self.max_tokens_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time) + 1
            if wait_time > 0:
                self.logger.info(f"Token limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.tokens_per_minute = 0
                self.minute_start_time = time.time()
        
        # Add small buffer between requests
        time.sleep(2.1)  # Just over 2 seconds to stay under 30/minute safely

    def _update_rate_limits_from_headers(self, response_headers):
        """Update rate limit tracking from response headers"""
        try:
            remaining_requests = response_headers.get('x-ratelimit-remaining-requests-day')
            remaining_tokens = response_headers.get('x-ratelimit-remaining-tokens-minute')
            
            if remaining_requests:
                self.logger.debug(f"Remaining requests today: {remaining_requests}")
            if remaining_tokens:
                self.logger.debug(f"Remaining tokens this minute: {remaining_tokens}")
                
        except Exception as e:
            self.logger.debug(f"Could not parse rate limit headers: {e}")

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
        
        prompt = f"""CRITICAL: You are an expert at Name Entity Recognition information extractor. Label the given text with named entities.

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
    
    def _make_api_call(self, prompt: str) -> tuple[str, int, int]:
        """
        Make API call with proper error handling and rate limiting
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Estimate tokens (rough approximation: 4 chars per token)
        estimated_tokens = len(prompt) // 4
        
        # Wait for rate limits
        self._wait_for_rate_limit(estimated_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=500,
                top_p=0.8
            )
            
            # Update rate limiting counters
            self.requests_per_minute += 1
            
            # Extract token counts from response
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            self.tokens_per_minute += input_tokens + output_tokens
            
            # Update from response headers if available
            if hasattr(response, 'response') and hasattr(response.response, 'headers'):
                self._update_rate_limits_from_headers(response.response.headers)
            
            return response.choices[0].message.content, input_tokens, output_tokens
            
        except cerebras.cloud.sdk.RateLimitError as e:
            self.logger.warning(f"Rate limit hit: {e}")
            # Exponential backoff for rate limits
            wait_time = min(60, 2 ** (3))  # Max 60 seconds
            time.sleep(wait_time)
            raise  # Re-raise to trigger retry
            
        except cerebras.cloud.sdk.APITimeoutError as e:
            self.logger.warning(f"API timeout: {e}")
            raise  # Re-raise to trigger retry
            
        except cerebras.cloud.sdk.APIConnectionError as e:
            self.logger.warning(f"Connection error: {e}")
            time.sleep(5)  # Brief pause before retry
            raise  # Re-raise to trigger retry
            
        except cerebras.cloud.sdk.APIStatusError as e:
            self.logger.error(f"API status error {e.status_code}: {e}")
            if e.status_code >= 500:
                time.sleep(10)  # Server error, wait longer
            raise  # Re-raise to trigger retry
    
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
            self.logger.info("CEREBRAS API LABEL GENERATION")
            self.logger.info("="*60)
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Context limit: {self.context_limit:,} tokens")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Low confidence examples available: {len(low_n_examples)}")
            self.logger.info(f"Target labels: {num_samples}")
            self.logger.info(f"Cached labels: {len(label_cache)}")
            self.logger.info(f"Rate limits: {self.max_requests_per_minute} req/min, {self.max_tokens_per_minute:,} tokens/min")
            self.logger.info("="*60)
        
        # Calculate how many new labels we actually need
        if len(label_cache) >= num_samples:
            if verbose:
                self.logger.info(f"Using {num_samples} labels from cache (no generation needed)")
            return label_cache[:num_samples]
        
        no_new_labels_needed = num_samples - len(label_cache)
        if verbose:
            estimated_time = (no_new_labels_needed * 2.1) / 60  # 2.1 seconds per request
            self.logger.info(f"Need to generate {no_new_labels_needed} new labels ({len(label_cache)} already cached)")
            self.logger.info(f"Estimated time: {estimated_time:.1f} minutes")
        
        # Check if we have enough examples to label
        available_examples = len(low_n_examples) - len(label_cache)
        if available_examples < no_new_labels_needed:
            self.logger.warning(f"Not enough examples! Need {no_new_labels_needed}, have {available_examples}")
            no_new_labels_needed = available_examples
        
        synthetic_outputs = []
        input_tokens_list = []
        output_tokens_list = []
        
        # Model configuration for token metrics
        model_limits = (self.context_limit, 500)  # (context, max_output)
        
        # Generation loop with immediate retry logic
        for i in tqdm(range(no_new_labels_needed), desc="Labeling", disable=not verbose):
            # Get next example to label (skip already cached ones)
            example_idx = len(label_cache) + i
            example = low_n_examples[example_idx]
            tokenized_text = example['tokenized_text']
            
            # Create labeling prompt
            prompt = self._create_prompt(tokenized_text, entity_types)
            
            # Check prompt length
            estimated_prompt_tokens = len(prompt) // 4
            if estimated_prompt_tokens > self.context_limit * 0.9:
                self.logger.warning(f"Prompt for example {i+1} may be too long ({estimated_prompt_tokens} est. tokens)")
            
            # Immediate retry logic
            max_retries = 3
            success = False
            
            for attempt in range(max_retries + 1):
                try:
                    response_text, input_tokens, output_tokens = self._make_api_call(prompt)
                    
                    # Store token counts
                    input_tokens_list.append(input_tokens)
                    output_tokens_list.append(output_tokens)
                    
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
                    
                    self.logger.info(f"Labeled example {i+1} {tokenized_text}: {js.get('entities', [])}")
                    
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        self.logger.warning(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")
                        # Add zeros for failed attempts
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
                        
                except Exception as e:
                    error_msg = f"Labeling failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        self.logger.error(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")
                        # Add zeros for failed attempts
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
            
            # Progress update every 50 examples
            if verbose and success and (i + 1) % 50 == 0:
                avg_input = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
                avg_output = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
                elapsed = (time.time() - self.minute_start_time) / 60
                remaining = no_new_labels_needed - (i + 1)
                eta = (remaining * 2.1) / 60  # Rough ETA
                self.logger.info(f"Progress: {i+1}/{no_new_labels_needed} | Tokens: in={avg_input:.0f}, out={avg_output:.0f} | ETA: {eta:.1f}min")
        
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
            
            # Show token metrics
            if input_tokens_list:
                avg_input = sum(input_tokens_list) / len(input_tokens_list)
                avg_output = sum(output_tokens_list) / len(output_tokens_list)
                total_tokens = sum(input_tokens_list) + sum(output_tokens_list)
                
                self.logger.info("="*60)
                self.logger.info(f"TOKEN METRICS (CEREBRAS API):")
                self.logger.info(f"   Average input tokens: {avg_input:.0f}")
                self.logger.info(f"   Average output tokens: {avg_output:.0f}")
                self.logger.info(f"   Total tokens used: {total_tokens:,}")
                self.logger.info(f"   Model limits: {model_limits}")
                
                # Show some stats for all data (cached + new)
                if len(label_cache) > 0:
                    avg_entities = sum(len(ex['ner']) for ex in label_cache) / len(label_cache)
                    self.logger.info(f"   Average entities per example: {avg_entities:.1f}")
                    
                    # Entity type distribution
                    entity_counts = {}
                    for ex in label_cache:
                        for _, _, label in ex['ner']:
                            entity_counts[label] = entity_counts.get(label, 0) + 1
                    self.logger.info(f"   Entity distribution: {entity_counts}")
            
            self.logger.info("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return label_cache[:num_samples]