"""
Batch analysis with retry logic - extracted from your monolithic analyze_single_batch_with_retry function
Preserves exact prompt engineering and error handling
"""

import ollama
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional


def analyze_single_batch_with_retry(batch_examples: List[Dict], entity_types: List[str], 
                                  batch_num: int, cache_manager, settings,
                                  max_retries: int = 3, logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    """
    Analyze a single batch with retry logic - exact copy of your original function
    
    Args:
        batch_examples: List of examples to analyze
        entity_types: List of entity types
        batch_num: Batch number for logging
        cache_manager: CacheManager instance
        settings: Settings object
        max_retries: Maximum retry attempts
        logger: Optional logger
        
    Returns:
        Analysis result dictionary or None if failed
    """
    if logger:
        logger.info(f"Analyzing batch {batch_num} ({len(batch_examples)} examples)")
    
    # Build prompt for this batch - exact copy from your original
    prompt = f"""CRITICAL: You are an expert NER analyst. This is a PRODUCTION system.

TARGET ENTITY TYPES: {', '.join(entity_types)}

BATCH EXAMPLES:
"""
    
    # Add batch examples
    for i, example in enumerate(batch_examples):
        text = " ".join(example['tokenized_text'])
        min_score = min(example['scores']) if example['scores'] else 0.0
        
        prompt += f"""
Example {i + 1}:
Text: {text}
Ground Truth: {example['ner']}
Predictions: {example['predictions']}
Scores: {example['scores']}
Min Score: {min_score:.3f}
"""
    
    # Add structured analysis request - exact copy from your original
    prompt += f"""

MANDATORY TASK: Provide domain summary, then analyze each entity type.

CRITICAL: OUTPUT MUST BE VALID JSON ONLY. NO OTHER TEXT ALLOWED.

MANDATORY OUTPUT JSON FORMAT:
{{
  "domain_summary": "brief description of domain patterns in this batch",
  "entity_analysis": {{"""
    
    # Add structure for each entity type
    for i, entity_type in enumerate(entity_types):
        if i > 0:
            prompt += ","
        prompt += f"""
    "{entity_type}": {{
      "position_analysis": "boundary/position issues for {entity_type} with specific text examples",
      "good_predicted_examples": ["actual text from examples that worked well", "another good text example"],
      "bad_predicted_examples": ["actual text from examples that failed", "another bad text example"],
      "variation_analysis": "variations needed for {entity_type} with specific examples"
    }}"""
    
    prompt += f"""
  }}
}}

CRITICAL REQUIREMENTS:
- MUST OUTPUT ONLY VALID JSON
- Focus ONLY on: {', '.join(entity_types)}
- Use ACTUAL TEXT from examples, not "example 1" or "example 2"  
- Show specific text snippets that worked/failed
- If entity not in batch, put "No examples in this batch"
- Keep it concise but use real text examples

MANDATORY: Generate ONLY the JSON structure above, nothing else:
"""
    
    # Retry logic for this batch - exact copy from your original
    for attempt in range(max_retries):
        if logger:
            logger.info(f"Batch {batch_num} - Attempt {attempt + 1}/{max_retries}")
        
        try:
            response = ollama.generate(
                model=settings.ollama_model,
                prompt=prompt,
                options={
                    'top_k': 50,
                    'top_p': settings.ollama_top_p,
                    'num_predict': 2500,
                    'temperature': settings.ollama_temperature,
                }
            )
            
            response_text = response['response'].strip()
            if logger:
                logger.info(f"Received response for batch {batch_num} attempt {attempt + 1} (length: {len(response_text)})")
            
            # Try to extract JSON - exact logic from your original
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                batch_result = json.loads(json_text)
                batch_result['batch_number'] = batch_num
                if logger:
                    logger.info(f"✅ Batch {batch_num} completed successfully on attempt {attempt + 1}")
                return batch_result
            else:
                if logger:
                    logger.warning(f"❌ No JSON found in batch {batch_num} attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        logger.error(f"Response preview: {response_text[:200]}...")
                
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"❌ JSON parsing failed for batch {batch_num} attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Response preview: {response_text[:200]}...")
        except Exception as e:
            if logger:
                logger.warning(f"❌ LLM call failed for batch {batch_num} attempt {attempt + 1}: {e}")
    
    if logger:
        logger.error(f"❌❌❌ CRITICAL: Batch {batch_num} failed after {max_retries} attempts")
    return None


def analyze_domain_with_batch_caching(low_confidence_examples: List[Dict], entity_types: List[str], 
                                    cache_manager, settings, batch_size: int = 10,
                                    logger: Optional[logging.Logger] = None) -> Optional[List[Dict]]:
    """
    Process examples in batches with intelligent batch caching - exact copy of your original function
    
    Args:
        low_confidence_examples: List of examples to analyze
        entity_types: List of entity types
        cache_manager: CacheManager instance
        settings: Settings object
        batch_size: Batch size for processing
        logger: Optional logger
        
    Returns:
        List of batch results or None if failed
    """
    total_examples = len(low_confidence_examples)
    if logger:
        logger.info("="*60)
        logger.info("DOMAIN ANALYSIS WITH BATCH CACHING")
        logger.info("="*60)
        logger.info(f"Analyzing {total_examples} examples in batches of {batch_size}")
        logger.info(f"Entity types: {entity_types}")
    
    all_results = []
    total_batches = (total_examples - 1) // batch_size + 1
    cache_hits = 0
    new_analyses = 0
    
    # Process in batches with caching - exact logic from your original
    for batch_start in range(0, total_examples, batch_size):
        batch_end = min(batch_start + batch_size, total_examples)
        batch_examples = low_confidence_examples[batch_start:batch_end]
        batch_num = batch_start//batch_size + 1
        
        # Check if this batch has been analyzed before
        batch_key = cache_manager.get_batch_cache_key(batch_examples)
        
        cached_result = cache_manager.get_batch_analysis(batch_key)
        if cached_result:
            if logger:
                logger.info(f"📋 Batch {batch_num}/{total_batches}: Using cached analysis (examples {batch_start+1}-{batch_end})")
            cached_result = cached_result.copy()
            cached_result['batch_number'] = batch_num  # Update batch number
            all_results.append(cached_result)
            cache_hits += 1
        else:
            if logger:
                logger.info(f"🔄 Batch {batch_num}/{total_batches}: New analysis needed (examples {batch_start+1}-{batch_end})")
            
            batch_result = analyze_single_batch_with_retry(
                batch_examples, entity_types, batch_num, cache_manager, settings, logger=logger
            )
            
            if batch_result is None:
                if logger:
                    logger.error(f"❌❌❌ CRITICAL: Batch {batch_num} analysis failed")
                return None
            
            # Cache the result for future use
            cache_manager.set_batch_analysis(batch_key, batch_result.copy())
            all_results.append(batch_result)
            new_analyses += 1
    
    if logger:
        logger.info("="*60)
        logger.info("BATCH ANALYSIS COMPLETED")
        logger.info("="*60)
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Cache hits: {cache_hits}")
        logger.info(f"New analyses: {new_analyses}")
        logger.info(f"Cache efficiency: {cache_hits / total_batches * 100:.1f}%")
    
    return all_results
