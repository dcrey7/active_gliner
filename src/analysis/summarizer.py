"""
Result summarization - extracted from your combine_all_batch_results and create_final_summary_with_retry functions
Preserves exact prompt engineering and logic
"""

import ollama
import json
from typing import List, Dict, Any, Optional
import logging


def combine_all_batch_results(all_batch_results: List[Dict], entity_types: List[str],
                            logger: Optional[logging.Logger] = None) -> Dict:
    """
    Combine results from all batches into one structure - exact copy of your original function
    
    Args:
        all_batch_results: List of batch analysis results
        entity_types: List of entity types
        logger: Optional logger
        
    Returns:
        Combined result dictionary
    """
    if logger:
        logger.info("="*60)
        logger.info("COMBINING BATCH RESULTS")
        logger.info("="*60)
        logger.info(f"Combining {len(all_batch_results)} batch results for entity types: {entity_types}")
    
    combined_result = {
        'total_batches': len(all_batch_results),
        'entity_types': entity_types,
        'domain_summaries': [],
        'combined_entity_analysis': {}
    }
    
    # Initialize combined entity analysis
    for entity_type in entity_types:
        combined_result['combined_entity_analysis'][entity_type] = {
            'position_analysis': [],
            'good_predicted_examples': [],
            'bad_predicted_examples': [],
            'variation_analysis': []
        }
    
    # Combine all batch results - exact logic from your original
    for result in all_batch_results:
        domain_summary = result.get('domain_summary', '')
        if domain_summary:
            combined_result['domain_summaries'].append(f"Batch {result.get('batch_number', '?')}: {domain_summary}")
        
        entity_analysis = result.get('entity_analysis', {})
        for entity_type in entity_types:
            if entity_type in entity_analysis:
                entity_data = entity_analysis[entity_type]
                
                pos_analysis = entity_data.get('position_analysis', '')
                if pos_analysis and 'No examples' not in pos_analysis:
                    combined_result['combined_entity_analysis'][entity_type]['position_analysis'].append(pos_analysis)
                
                good_examples = entity_data.get('good_predicted_examples', [])
                if isinstance(good_examples, list):
                    for example in good_examples:
                        if example and 'No examples' not in example:
                            combined_result['combined_entity_analysis'][entity_type]['good_predicted_examples'].append(example)
                
                bad_examples = entity_data.get('bad_predicted_examples', [])
                if isinstance(bad_examples, list):
                    for example in bad_examples:
                        if example and 'No examples' not in example:
                            combined_result['combined_entity_analysis'][entity_type]['bad_predicted_examples'].append(example)
                
                var_analysis = entity_data.get('variation_analysis', '')
                if var_analysis and 'No examples' not in var_analysis:
                    combined_result['combined_entity_analysis'][entity_type]['variation_analysis'].append(var_analysis)
    
    if logger:
        logger.info(f"✅ Combined {len(all_batch_results)} batches successfully")
    return combined_result


def create_final_summary_with_retry(combined_result: Dict, settings, max_retries: int = 3,
                                  logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    """
    Send combined results to LLM for final reasoning and summarization - exact copy of your original function
    
    Args:
        combined_result: Combined batch results
        settings: Settings object
        max_retries: Maximum retry attempts
        logger: Optional logger
        
    Returns:
        Final summary dictionary or None if failed
    """
    if logger:
        logger.info("="*60)
        logger.info("CREATING FINAL SUMMARY (WITH RETRY LOGIC)")
        logger.info("="*60)
        logger.info(f"Max retries: {max_retries}")
    
    # Build prompt - exact copy from your original
    prompt = f"""CRITICAL: You are an expert NER analyst. This is a PRODUCTION system.

I have analyzed multiple batches of NER examples. 

ENTITY TYPES ANALYZED: {', '.join(combined_result['entity_types'])}

DOMAIN SUMMARIES FROM BATCHES:
{chr(10).join(f"- {summary}" for summary in combined_result['domain_summaries'])}

DETAILED ENTITY ANALYSIS:
"""
    
    for entity_type, data in combined_result['combined_entity_analysis'].items():
        if any([data['position_analysis'], data['good_predicted_examples'], 
               data['bad_predicted_examples'], data['variation_analysis']]):
            prompt += f"""
{entity_type.upper()}:
Position Analysis: {'; '.join(data['position_analysis'])}
Good Examples: {'; '.join(data['good_predicted_examples'])}
Bad Examples: {'; '.join(data['bad_predicted_examples'])}  
Variation Analysis: {'; '.join(data['variation_analysis'])}
"""
    
    prompt += f"""

MANDATORY TASK: Based on all this data, provide a final reasoned summary.

CRITICAL: OUTPUT MUST BE VALID JSON ONLY. NO OTHER TEXT ALLOWED.

MANDATORY OUTPUT JSON FORMAT:
{{
  "domain_summary": "Overall domain description and main patterns",
  "entity_summaries": {{"""
    
    for i, entity_type in enumerate(combined_result['entity_types']):
        if i > 0:
            prompt += ","
        prompt += f"""
    "{entity_type}": {{
      "position_summary": "Key position/boundary issues for {entity_type}",
      "good_examples_summary": "What works well for {entity_type} prediction",
      "bad_examples_summary": "What fails for {entity_type} prediction", 
      "variations_summary": "Key variations needed for {entity_type}"
    }}"""
    
    prompt += f"""
  }}
}}

CRITICAL REQUIREMENTS:
- MUST OUTPUT ONLY VALID JSON
- Synthesize findings from all batches
- Focus on patterns across all examples
- Keep summaries concise but informative
- If no data for an entity, put "No significant findings"

MANDATORY: Generate ONLY the JSON structure above, nothing else:
"""
    
    # Retry logic for final summary - exact copy from your original
    for attempt in range(max_retries):
        if logger:
            logger.info(f"Final Summary - Attempt {attempt + 1}/{max_retries}")
        
        try:
            response = ollama.generate(
                model=settings.ollama_model,
                prompt=prompt,
                options={
                    'top_k': 50,
                    'top_p': settings.ollama_top_p,
                    'num_predict': 2000,
                    'temperature': 0.1,
                }
            )
            
            response_text = response['response'].strip()
            if logger:
                logger.info(f"Received final summary response attempt {attempt + 1} (length: {len(response_text)})")
            
            # Extract JSON - exact logic from your original
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                final_summary = json.loads(json_text)
                if logger:
                    logger.info(f"✅ Final summary created successfully on attempt {attempt + 1}!")
                return final_summary
            else:
                if logger:
                    logger.warning(f"❌ No JSON found in final summary attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        logger.error(f"Response preview: {response_text[:200]}...")
                
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"❌ JSON parsing failed for final summary attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Response preview: {response_text[:200]}...")
        except Exception as e:
            if logger:
                logger.warning(f"❌ LLM call failed for final summary attempt {attempt + 1}: {e}")
    
    if logger:
        logger.error("❌❌❌ CRITICAL: Final summary failed after all retry attempts")
    return None
