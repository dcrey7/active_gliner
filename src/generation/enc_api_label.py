"""
Enhanced Label Generator for Existing Text using Cerebras API
Production-Ready with Disk Caching, Structured Outputs, and Graceful Quota Handling

Features:
- Structured outputs (native JSON schema validation)
- Persistent disk caching to results/data/
- Graceful handling of API quota limits
- Resume capability from disk cache
- Zero data loss on quota exceeded errors
- Atomic file writes for safety
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import cerebras.cloud.sdk
from cerebras.cloud.sdk import Cerebras

from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from utils.logging import get_logger


class QuotaExceededException(Exception):
    """Raised when API quota is exceeded with partial results"""
    
    def __init__(self, partial_labels: List[Dict], requested: int, actual: int, message: str):
        self.partial_labels = partial_labels
        self.requested = requested
        self.actual = actual
        self.message = message
        super().__init__(message)


class LabelGenerator:
    """Enhanced label generator with structured outputs and disk caching"""
    
    LABEL_SCHEMA = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "types": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["entity", "types"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["text", "entities"],
        "additionalProperties": False
    }
    
    def __init__(self, model_name: str = "qwen-3-235b-a22b-thinking-2507"):
        """Initialize the enhanced label generator"""
        self.model_name = model_name
        self.logger = get_logger("ActiveLearning")
        
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable not set")
        
        self.client = Cerebras(
            api_key=api_key,
            max_retries=2,
            timeout=90.0
        )
        
        self.logger.info(f"Enhanced Label Generator initialized: {self.model_name}")
        
        # Rate limiting tracking
        self.requests_per_minute = 0
        self.minute_start_time = time.time()
        self.tokens_per_minute = 0
        
        # Model limits for thinking model
        self.max_requests_per_minute = 30
        self.max_tokens_per_minute = 60000
        self.context_limit = 65536
        
        # Cache directory
        self.cache_dir = Path("../results/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, num_labels: int) -> Path:
        """Generate cache filename based on model and label count"""
        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_model_name}_{num_labels}_labels.json"
    
    def _load_cache_from_disk(self, target_labels: int) -> List[Dict]:
        """Load existing cache from disk if available"""
        cache_file = self._get_cache_filename(target_labels)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                labels = cache_data.get('labels', [])
                self.logger.info(f"📂 Loaded {len(labels)} labels from exact cache: {cache_file.name}")
                return labels
            except Exception as e:
                self.logger.warning(f"Failed to load cache from {cache_file}: {e}")
        
        # Check for smaller cache files
        for num_labels in range(target_labels - 500, 0, -50):
            if num_labels > 0:
                cache_file = self._get_cache_filename(num_labels)
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        labels = cache_data.get('labels', [])
                        self.logger.info(f"📂 Loaded {len(labels)} labels from partial cache: {cache_file.name}")
                        return labels
                    except Exception as e:
                        self.logger.warning(f"Failed to load partial cache from {cache_file}: {e}")
        
        self.logger.info("No existing cache found, starting fresh")
        return []
    
    def _save_cache_to_disk(self, label_cache: List[Dict], reason: str = "completed") -> None:
        """Save cache to disk atomically"""
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
            
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            temp_file.rename(cache_file)
            self.logger.info(f"💾 Saved {num_labels} labels to {cache_file.name} (reason: {reason})")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_file}: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def _is_hard_quota_error(self, error: Exception) -> bool:
        """Check if error is a hard quota limit (daily/hourly)"""
        error_str = str(error).lower()
        hard_quota_indicators = [
            "token_quota_exceeded", "tokens per day limit exceeded",
            "tokens per hour limit exceeded", "daily limit exceeded",
            "hourly limit exceeded", "requests per day limit exceeded",
            "requests per hour limit exceeded", "quota exceeded"
        ]
        return any(indicator in error_str for indicator in hard_quota_indicators)
    
    def _wait_for_rate_limit(self, estimated_tokens: int = 500):
        """Intelligent rate limiting based on current usage"""
        current_time = time.time()
        
        if current_time - self.minute_start_time >= 60:
            self.requests_per_minute = 0
            self.tokens_per_minute = 0
            self.minute_start_time = current_time
        
        if self.requests_per_minute >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time) + 1
            if wait_time > 0:
                self.logger.info(f"Request limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.tokens_per_minute = 0
                self.minute_start_time = time.time()
        
        if self.tokens_per_minute + estimated_tokens >= self.max_tokens_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time) + 1
            if wait_time > 0:
                self.logger.info(f"Token limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.tokens_per_minute = 0
                self.minute_start_time = time.time()
        
        time.sleep(2.1)
    
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
        """Create labeling prompt"""
        text = " ".join(tokenized_text)
        
        prompt = f"""CRITICAL: You are an expert at Named Entity Recognition. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
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

**MANDATORY Output Format:**
{{
  "text": "{text}",
  "entities": [
    {{"entity": "exact entity name", "types": ["entity_type"]}},
    ...
  ]
}}

CRITICAL: Generate ONLY the JSON format above. Start immediately with the JSON object.
"""
        return prompt
    
    def _make_api_call(self, prompt: str) -> Tuple[str, int, int]:
        """Make API call with structured outputs"""
        estimated_tokens = len(prompt) // 4
        self._wait_for_rate_limit(estimated_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=60000,
                top_p=0.8,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ner_label",
                        "strict": True,
                        "schema": self.LABEL_SCHEMA
                    }
                }
            )
            
            self.requests_per_minute += 1
            
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            self.tokens_per_minute += input_tokens + output_tokens
            
            if hasattr(response, 'response') and hasattr(response.response, 'headers'):
                self._update_rate_limits_from_headers(response.response.headers)
            
            return response.choices[0].message.content, input_tokens, output_tokens
            
        except cerebras.cloud.sdk.RateLimitError as e:
            self.logger.warning(f"Rate limit hit: {e}")
            if self._is_hard_quota_error(e):
                raise
            wait_time = min(60, 2 ** 3)
            time.sleep(wait_time)
            raise
            
        except cerebras.cloud.sdk.APITimeoutError as e:
            self.logger.warning(f"API timeout: {e}")
            raise
            
        except cerebras.cloud.sdk.APIConnectionError as e:
            self.logger.warning(f"Connection error: {e}")
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
        """Generate labels with persistent caching and graceful quota handling"""
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
            self.logger.info(f"Rate limits: {self.max_requests_per_minute} req/min, {self.max_tokens_per_minute:,} tokens/min")
        
        disk_cache = self._load_cache_from_disk(num_samples)
        
        if disk_cache and len(disk_cache) > len(label_cache):
            self.logger.info(f"Using disk cache ({len(disk_cache)} labels) over runtime cache ({len(label_cache)} labels)")
            label_cache.clear()
            label_cache.extend(disk_cache)
        elif disk_cache:
            self.logger.info(f"Runtime cache ({len(label_cache)} labels) is larger than disk cache ({len(disk_cache)} labels)")
        
        if len(label_cache) >= num_samples:
            if verbose:
                self.logger.info(f"Using {num_samples} labels from cache (no generation needed)")
                self.logger.info("="*60)
            return label_cache[:num_samples]
        
        no_new_labels_needed = num_samples - len(label_cache)
        if verbose:
            estimated_time = (no_new_labels_needed * 2.1) / 60
            self.logger.info(f"Need to generate {no_new_labels_needed} new labels ({len(label_cache)} already cached)")
            self.logger.info(f"Estimated time: {estimated_time:.1f} minutes")
        
        available_examples = len(low_n_examples) - len(label_cache)
        if available_examples < no_new_labels_needed:
            self.logger.warning(f"Not enough examples! Need {no_new_labels_needed}, have {available_examples}")
            no_new_labels_needed = available_examples
        
        synthetic_outputs = []
        input_tokens_list = []
        output_tokens_list = []
        
        if verbose:
            self.logger.info("="*60)
        
        try:
            for i in tqdm(range(no_new_labels_needed), desc="Enhanced Labeling", disable=not verbose):
                example_idx = len(label_cache) + i
                example = low_n_examples[example_idx]
                tokenized_text = example['tokenized_text']
                
                prompt = self._create_prompt(tokenized_text, entity_types)
                
                estimated_prompt_tokens = len(prompt) // 4
                if estimated_prompt_tokens > self.context_limit * 0.9:
                    self.logger.warning(f"Prompt for example {i+1} may be too long ({estimated_prompt_tokens} est. tokens)")
                
                max_retries = 3
                success = False
                
                for attempt in range(max_retries + 1):
                    try:
                        response_text, input_tokens, output_tokens = self._make_api_call(prompt)
                        
                        js = json.loads(response_text)
                        
                        # Store with token metadata
                        js['_token_input'] = input_tokens
                        js['_token_output'] = output_tokens
                        js['_model'] = self.model_name
                        
                        synthetic_outputs.append(js)
                        input_tokens_list.append(input_tokens)
                        output_tokens_list.append(output_tokens)
                        success = True
                        
                        if verbose:
                            self.logger.info(f"✅ Labeled example {i+1}/{no_new_labels_needed}")
                            self.logger.info(f"📝 Text: {' '.join(tokenized_text)}")
                            self.logger.info(f"🏷️  Full structured output: {json.dumps(js, ensure_ascii=False)}")
                            self.logger.info(f"📊 Tokens: input={input_tokens}, output={output_tokens}")
                        
                        break
                        
                    except cerebras.cloud.sdk.RateLimitError as e:
                        if self._is_hard_quota_error(e):
                            raise
                        else:
                            if attempt < max_retries:
                                wait_time = min(60, 2 ** (attempt + 1))
                                self.logger.warning(f"Temporary rate limit, waiting {wait_time}s... (attempt {attempt+1})")
                                time.sleep(wait_time)
                            else:
                                self.logger.error(f"Rate limit retries exhausted for example {i+1}")
                                break
                                
                    except json.JSONDecodeError as e:
                        error_msg = f"JSON parsing failed for example {i+1}, attempt {attempt+1}: {str(e)}"
                        if verbose:
                            self.logger.warning(error_msg)
                        if attempt == max_retries:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed validation after all retries")
                            input_tokens_list.append(0)
                            output_tokens_list.append(0)
                            
                    except Exception as e:
                        error_msg = f"Labeling failed for example {i+1}, attempt {attempt+1}: {str(e)}"
                        if verbose:
                            self.logger.error(error_msg)
                        if attempt == max_retries:
                            input_tokens_list.append(0)
                            output_tokens_list.append(0)
                
                if not success:
                    self.logger.warning(f"Failed to label example {i+1} after all attempts")
                
                if verbose and success and (i + 1) % 50 == 0:
                    avg_input = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
                    avg_output = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
                    remaining = no_new_labels_needed - (i + 1)
                    eta = (remaining * 2.1) / 60
                    self.logger.info(f"Progress: {i+1}/{no_new_labels_needed} | Tokens: in={avg_input:.0f}, out={avg_output:.0f} | ETA: {eta:.1f}min")
                
        except cerebras.cloud.sdk.RateLimitError as e:
            if self._is_hard_quota_error(e):
                self.logger.error("="*60)
                self.logger.error("🚨 HARD QUOTA LIMIT REACHED")
                self.logger.error("="*60)
                self.logger.error(f"Full error: {str(e)}")
                
                if synthetic_outputs:
                    self.logger.info(f"Processing {len(synthetic_outputs)} pending synthetic outputs...")
                    
                    ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
                    self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
                    
                    # Preserve token metadata during cleaning
                    for orig, ner in zip(synthetic_outputs, ner_formatted_data):
                        if '_token_input' in orig:
                            ner['_token_input'] = orig['_token_input']
                            ner['_token_output'] = orig['_token_output']
                            ner['_model'] = orig['_model']
                    
                    cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types, self.logger)
                    self.logger.info(f"Cleaned examples: {len(cleaned_data)}")
                    
                    label_cache.extend(cleaned_data)
                    self.logger.info(f"Added {len(cleaned_data)} labels to cache. Total cache: {len(label_cache)}")
                
                self._save_cache_to_disk(label_cache, reason="quota_exceeded")
                
                actual_labels = len(label_cache)
                raise QuotaExceededException(
                    partial_labels=label_cache,
                    requested=num_samples,
                    actual=actual_labels,
                    message=f"Daily quota exceeded. Generated {actual_labels}/{num_samples} labels."
                )
            else:
                raise
        
        if verbose:
            self.logger.info(f"Successfully generated {len(synthetic_outputs)}/{no_new_labels_needed} labels")
        
        if synthetic_outputs:
            ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
            if verbose:
                self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")
            
            # Preserve token metadata during cleaning
            for orig, ner in zip(synthetic_outputs, ner_formatted_data):
                if '_token_input' in orig:
                    ner['_token_input'] = orig['_token_input']
                    ner['_token_output'] = orig['_token_output']
                    ner['_model'] = orig['_model']
            
            cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types, self.logger)
            if verbose:
                self.logger.info(f"Final cleaned examples: {len(cleaned_data)}")
            
            label_cache.extend(cleaned_data)
            
            self._save_cache_to_disk(label_cache, reason="completed")
        
        if verbose:
            self.logger.info(f"Cache updated: {len(label_cache)} total labeled examples")
            
            if input_tokens_list:
                avg_input = sum(input_tokens_list) / len(input_tokens_list)
                avg_output = sum(output_tokens_list) / len(output_tokens_list)
                total_tokens = sum(input_tokens_list) + sum(output_tokens_list)
                
                self.logger.info("="*60)
                self.logger.info(f"TOKEN METRICS (ACTUAL FROM CEREBRAS API):")
                self.logger.info(f"   Average input tokens: {avg_input:.0f}")
                self.logger.info(f"   Average output tokens: {avg_output:.0f}")
                self.logger.info(f"   Total tokens used: {total_tokens:,}")
                self.logger.info(f"   Model limits: ({self.context_limit}, 60000)")
                
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