"""
Enhanced Label Generator for Existing Text using Cerebras API
Robust, Persistent, and Resilient - Generate Labels with Graceful Failure Handling

Features:
- Persistent caching to disk (results/data/)
- Graceful handling of API quota limits
- Structured output validation with Pydantic
- Resume capability from disk cache
- Zero data loss on quota exceeded errors

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
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import cerebras.cloud.sdk
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel, ValidationError

from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from utils.logging import get_logger


class EntityAnnotation(BaseModel):
    """Pydantic model for entity annotation"""
    entity: str
    types: List[str]


class LabelResponse(BaseModel):
    """Pydantic model for labeling response"""
    text: str
    entities: List[EntityAnnotation]


class LabelGenerator:
    """Enhanced label generator with persistent caching and graceful failure handling"""
    
    def __init__(self, model_name: str = "qwen-3-235b-a22b-instruct-2507"):
        """
        Initialize the enhanced label generator
        
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
        
        self.logger.info(f"Enhanced Label Generator model: {self.model_name}")
        
        # Rate limiting tracking
        self.requests_per_minute = 0
        self.minute_start_time = time.time()
        self.tokens_per_minute = 0
        
        # Model limits (both models have same limits)
        self.max_requests_per_minute = 30
        self.max_tokens_per_minute = 64000 if "gpt-oss" in model_name else 60000
        self.context_limit = 65536
        
        # Cache directory
        self.cache_dir = Path("results/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_filename(self, num_labels: int) -> Path:
        """Generate cache filename based on model and label count"""
        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_model_name}_{num_labels}_labels.json"

    def _load_cache_from_disk(self, target_labels: int) -> List[Dict]:
        """
        Load existing cache from disk if available
        
        Args:
            target_labels: Target number of labels we want
            
        Returns:
            List of cached labels (empty list if no cache found)
        """
        # Check for exact match first
        cache_file = self._get_cache_filename(target_labels)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                labels = cache_data.get('labels', [])
                self.logger.info(f"Loaded {len(labels)} labels from exact cache: {cache_file}")
                return labels
            except Exception as e:
                self.logger.warning(f"Failed to load cache from {cache_file}: {e}")
        
        # Check for smaller cache files we can build upon
        for num_labels in range(target_labels - 500, target_labels, 50):
            if num_labels > 0:
                cache_file = self._get_cache_filename(num_labels)
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        labels = cache_data.get('labels', [])
                        self.logger.info(f"Loaded {len(labels)} labels from partial cache: {cache_file}")
                        return labels
                    except Exception as e:
                        self.logger.warning(f"Failed to load partial cache from {cache_file}: {e}")
        
        self.logger.info("No existing cache found, starting fresh")
        return []

    def _save_cache_to_disk(self, label_cache: List[Dict], reason: str = "quota_exceeded") -> None:
        """
        Save cache to disk atomically
        
        Args:
            label_cache: List of labeled examples to save
            reason: Reason for saving (for logging)
        """
        if not label_cache:
            self.logger.warning("No labels to save to cache")
            return
        
        num_labels = len(label_cache)
        cache_file = self._get_cache_filename(num_labels)
        temp_file = cache_file.with_suffix('.tmp')
        
        try:
            cache_data = {
                "metadata": {
                    "model_name": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "total_labels": num_labels,
                    "reason": reason,
                    "context_limit": self.context_limit
                },
                "labels": label_cache
            }
            
            # Atomic write: write to temp file first, then rename
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            temp_file.rename(cache_file)
            self.logger.info(f"Successfully saved {num_labels} labels to {cache_file} (reason: {reason})")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_file}: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _is_hard_quota_error(self, error: Exception) -> bool:
        """
        Check if error is a hard quota limit (daily/hourly) vs temporary rate limit
        
        Args:
            error: Exception to check
            
        Returns:
            True if hard quota error that should trigger cache save and graceful exit
        """
        error_str = str(error).lower()
        
        # Hard quota errors that can't be retried today
        hard_quota_indicators = [
            "token_quota_exceeded",
            "tokens per day limit exceeded",
            "daily limit exceeded", 
            "requests per day limit exceeded",
            "quota exceeded",
            "too_many_tokens_error"
        ]
        
        return any(indicator in error_str for indicator in hard_quota_indicators)

    def _wait_for_rate_limit(self, estimated_tokens: int = 500):
        """Intelligent rate limiting based on current usage"""
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
        
        # Add buffer between requests
        time.sleep(2.1)

    def _create_prompt(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """Create labeling prompt with Pydantic schema instructions"""
        text = " ".join(tokenized_text)
        
        prompt = f"""CRITICAL: You are an expert at Named Entity Recognition. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**MANDATORY Format Requirements:**
- Output MUST be valid JSON matching the exact schema below
- Each entity MUST be accurately labeled with the specified entity types
- Use ONLY the provided entity types

**Entity Types to Use (ONLY these types):**
"""
        
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

**MANDATORY JSON Schema:**
{{
  "text": "{text}",
  "entities": [
    {{"entity": "exact entity name", "types": ["entity_type"]}},
    ...
  ]
}}

**Example Format:**
{{
  "text": "John works at Microsoft in Seattle",
  "entities": [
    {{"entity": "John", "types": ["PERSON"]}},
    {{"entity": "Microsoft", "types": ["ORG"]}},
    {{"entity": "Seattle", "types": ["LOCATION"]}}
  ]
}}

CRITICAL: Generate ONLY the JSON format above. Start immediately with the JSON object.
"""
        
        return prompt

    def _make_api_call_with_validation(self, prompt: str) -> tuple[LabelResponse, int, int]:
        """
        Make API call with Pydantic validation
        
        Returns:
            Tuple of (validated_response, input_tokens, output_tokens)
        """
        estimated_tokens = len(prompt) // 4
        self._wait_for_rate_limit(estimated_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=500,
                top_p=0.8,
                # Note: Structured outputs would go here if using JSON schema mode
                # response_format={"type": "json_schema", "json_schema": {...}}
            )
            
            # Update rate limiting counters
            self.requests_per_minute += 1
            
            # Extract token counts
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            self.tokens_per_minute += input_tokens + output_tokens
            
            # Parse and validate response
            response_text = response.choices[0].message.content
            
            # Clean response text
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            if '{' in response_text and '}' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                response_text = response_text[start_idx:end_idx]
            
            # Parse JSON and validate with Pydantic
            json_data = json.loads(response_text)
            validated_response = LabelResponse(**json_data)
            
            return validated_response, input_tokens, output_tokens
            
        except cerebras.cloud.sdk.RateLimitError as e:
            if self._is_hard_quota_error(e):
                self.logger.error(f"Hard quota limit reached: {e}")
                raise  # This will be caught by the quota handler
            else:
                self.logger.warning(f"Temporary rate limit: {e}")
                wait_time = min(60, 2 ** 3)
                time.sleep(wait_time)
                raise  # Re-raise for retry
                
        except (cerebras.cloud.sdk.APITimeoutError, cerebras.cloud.sdk.APIConnectionError) as e:
            self.logger.warning(f"API error (retryable): {e}")
            time.sleep(5)
            raise
            
        except cerebras.cloud.sdk.APIStatusError as e:
            self.logger.error(f"API status error {e.status_code}: {e}")
            if e.status_code >= 500:
                time.sleep(10)
            raise

    def generate(self, low_n_examples: List[Dict], num_samples: int, 
                entity_types: List[str], label_cache: List[Dict], 
                verbose: bool = True) -> List[Dict]:
        """
        Generate labels with persistent caching and graceful failure handling
        
        Args:
            low_n_examples: Low confidence examples with tokenized_text
            num_samples: Number of examples to label
            entity_types: Entity types to identify
            label_cache: Cache list that gets extended (for compatibility)
            verbose: Whether to show progress
            
        Returns:
            List of cleaned NER formatted examples
        """
        if verbose:
            self.logger.info("="*60)
            self.logger.info("ENHANCED CEREBRAS API LABEL GENERATION")
            self.logger.info("="*60)
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Context limit: {self.context_limit:,} tokens")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Low confidence examples available: {len(low_n_examples)}")
            self.logger.info(f"Target labels: {num_samples}")
            self.logger.info(f"Current cache size: {len(label_cache)}")
            
        # Try to load existing cache from disk
        disk_cache = self._load_cache_from_disk(num_samples)
        
        # Merge disk cache with runtime cache (prefer disk cache for completeness)
        if disk_cache:
            if len(disk_cache) > len(label_cache):
                self.logger.info(f"Using disk cache ({len(disk_cache)} labels) over runtime cache ({len(label_cache)} labels)")
                label_cache.clear()
                label_cache.extend(disk_cache)
            else:
                self.logger.info(f"Runtime cache ({len(label_cache)} labels) is larger than disk cache ({len(disk_cache)} labels)")
        
        # Calculate how many new labels we need
        if len(label_cache) >= num_samples:
            if verbose:
                self.logger.info(f"Using {num_samples} labels from cache (no generation needed)")
            return label_cache[:num_samples]
        
        no_new_labels_needed = num_samples - len(label_cache)
        if verbose:
            estimated_time = (no_new_labels_needed * 2.1) / 60
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
        
        # Generation loop with quota-aware error handling
        try:
            for i in tqdm(range(no_new_labels_needed), desc="Enhanced Labeling", disable=not verbose):
                example_idx = len(label_cache) + i
                example = low_n_examples[example_idx]
                tokenized_text = example['tokenized_text']
                
                prompt = self._create_prompt(tokenized_text, entity_types)
                
                # Retry logic for this specific example
                max_retries = 3
                success = False
                
                for attempt in range(max_retries + 1):
                    try:
                        validated_response, input_tokens, output_tokens = self._make_api_call_with_validation(prompt)
                        
                        # Convert Pydantic model to our expected format
                        synthetic_output = {
                            "text": validated_response.text,
                            "entities": [
                                {"entity": ent.entity, "types": ent.types}
                                for ent in validated_response.entities
                            ]
                        }
                        
                        synthetic_outputs.append(synthetic_output)
                        input_tokens_list.append(input_tokens)
                        output_tokens_list.append(output_tokens)
                        success = True
                        
                        if verbose and i % 20 == 0:
                            self.logger.info(f"Labeled example {i+1}/{no_new_labels_needed}: {synthetic_output.get('entities', [])}")
                        
                        break  # Success, exit retry loop
                        
                    except cerebras.cloud.sdk.RateLimitError as e:
                        if self._is_hard_quota_error(e):
                            self.logger.error(f"HARD QUOTA EXCEEDED at example {i+1}! Processing remaining data and saving...")
                            raise  # This will be caught by outer try-catch
                        else:
                            if attempt < max_retries:
                                wait_time = min(60, 2 ** (attempt + 1))
                                self.logger.warning(f"Temporary rate limit, waiting {wait_time}s... (attempt {attempt+1})")
                                time.sleep(wait_time)
                            else:
                                self.logger.error(f"Rate limit retries exhausted for example {i+1}")
                                break
                                
                    except (json.JSONDecodeError, ValidationError) as e:
                        error_msg = f"Validation failed for example {i+1}, attempt {attempt+1}: {str(e)[:100]}"
                        if verbose:
                            self.logger.warning(error_msg)
                        if attempt == max_retries:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed validation after all retries")
                            input_tokens_list.append(0)
                            output_tokens_list.append(0)
                            
                    except Exception as e:
                        error_msg = f"Labeling failed for example {i+1}, attempt {attempt+1}: {str(e)[:100]}"
                        if verbose:
                            self.logger.error(error_msg)
                        if attempt == max_retries:
                            input_tokens_list.append(0)
                            output_tokens_list.append(0)
                
                if not success:
                    self.logger.warning(f"Failed to label example {i+1} after all attempts")
                
        except cerebras.cloud.sdk.RateLimitError as e:
            if self._is_hard_quota_error(e):
                self.logger.error("HARD QUOTA LIMIT REACHED - Entering graceful shutdown mode")
                
                # Process any remaining synthetic outputs through the cleaning pipeline
                if synthetic_outputs:
                    self.logger.info(f"Processing {len(synthetic_outputs)} remaining synthetic outputs...")
                    
                    # Convert to NER format
                    ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
                    self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
                    
                    # Clean and validate
                    cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types, self.logger)
                    self.logger.info(f"Cleaned examples: {len(cleaned_data)}")
                    
                    # Add to cache
                    label_cache.extend(cleaned_data)
                    self.logger.info(f"Added {len(cleaned_data)} labels to cache. Total cache: {len(label_cache)}")
                
                # Save current progress to disk
                self._save_cache_to_disk(label_cache, reason="quota_exceeded")
                
                # Return whatever we have so far
                self.logger.info(f"Graceful exit: returning {len(label_cache)} labels (requested {num_samples})")
                return label_cache
            else:
                # Re-raise if not a hard quota error
                raise
        
        # Normal completion path
        if verbose:
            self.logger.info(f"Successfully generated {len(synthetic_outputs)}/{no_new_labels_needed} labels")
        
        # Process all synthetic outputs
        if synthetic_outputs:
            ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
            if verbose:
                self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
            
            cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types, self.logger)
            if verbose:
                self.logger.info(f"Final cleaned examples: {len(cleaned_data)}")
            
            label_cache.extend(cleaned_data)
            
            # Save successful completion to disk
            self._save_cache_to_disk(label_cache, reason="completed")
        
        if verbose:
            self.logger.info(f"Cache updated: {len(label_cache)} total labeled examples")
            
            # Show token metrics
            if input_tokens_list:
                avg_input = sum(input_tokens_list) / len(input_tokens_list)
                avg_output = sum(output_tokens_list) / len(output_tokens_list)
                total_tokens = sum(input_tokens_list) + sum(output_tokens_list)
                
                self.logger.info("="*60)
                self.logger.info(f"TOKEN METRICS (ENHANCED CEREBRAS API):")
                self.logger.info(f"   Average input tokens: {avg_input:.0f}")
                self.logger.info(f"   Average output tokens: {avg_output:.0f}")
                self.logger.info(f"   Total tokens used: {total_tokens:,}")
                self.logger.info(f"   Model context limit: {self.context_limit:,}")
                
                if len(label_cache) > 0:
                    avg_entities = sum(len(ex['ner']) for ex in label_cache) / len(label_cache)
                    self.logger.info(f"   Average entities per example: {avg_entities:.1f}")
                    
                    entity_counts = {}
                    for ex in label_cache:
                        for _, _, label in ex['ner']:
                            entity_counts[label] = entity_counts.get(label, 0) + 1
                    self.logger.info(f"   Entity distribution: {entity_counts}")
            
            self.logger.info("="*60)

        return label_cache[:num_samples]