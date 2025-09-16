"""
Synthetic data generation - extracted from your generate_synthetic_data_incremental function
Preserves exact LLM integration and validation logic
"""

import ollama
import json
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from .prompts import create_baseline_synthetic_prompt, create_targeted_prompt_with_analysis


def generate_synthetic_data_incremental(low_confidence_examples: List[Dict], entity_types: List[str], 
                                       num_examples: int, num_samples_needed: int, settings,
                                       final_summary: Optional[Dict] = None,
                                       logger: Optional[logging.Logger] = None) -> tuple[List[Dict], List[Dict]]:
    """
    Generate synthetic training data incrementally - configuration-driven version
    
    Args:
        low_confidence_examples: List of corrected low confidence examples
        entity_types: List of entity types
        num_examples: Number of corrected examples (for mode determination)
        num_samples_needed: Number of synthetic examples to generate
        settings: Settings object
        final_summary: Optional analysis summary for integration
        logger: Optional logger
        
    Returns:
        Tuple of (synthetic_json_outputs, cleaned_ner_formatted_data)
    """
    # Get configuration values from settings instead of hardcoding
    gen_config = settings.get_generation_config()
    countries = gen_config['countries']
    language = gen_config['language']
    
    # Get domain focus from analysis or fallback to default
    domain_focus = settings.get_domain_focus(final_summary)
    
    if logger:
        logger.info("="*60)
        logger.info("INCREMENTAL SYNTHETIC DATA GENERATION")
        logger.info("="*60)
        logger.info(f"Need {num_samples_needed} new synthetic examples")
        logger.info(f"Using entity types: {entity_types}")
        logger.info(f"Domain focus: {domain_focus}")
        logger.info(f"Language: {language}")
        logger.info(f"Corrected examples available: {len(low_confidence_examples)}")
        logger.info(f"Using domain analysis: {'YES' if final_summary else 'NO'}")
    
    # Handle zero corrected examples case - exact logic from your original
    use_baseline_prompt = len(low_confidence_examples) == 0
    if use_baseline_prompt:
        if logger:
            logger.info("⚠️ ZERO CORRECTED EXAMPLES - Using baseline synthetic data generation")
    else:
        if logger:
            logger.info(f"✅ Using {len(low_confidence_examples)} corrected examples as templates")
    
    synthetic_outputs = []
    
    # Generation loop - using configuration values
    for i in tqdm(range(num_samples_needed), desc="Generating synthetic data"):
        # Random variation using config values
        country = random.choice(countries)
        
        if logger:
            logger.debug(f"Generating example {i+1}/{num_samples_needed} with country: {country}")
        
        # Create appropriate prompt using config values
        if use_baseline_prompt:
            # Use baseline prompt when no corrected examples available
            prompt = create_baseline_synthetic_prompt(
                entity_types=entity_types,
                domain_focus=domain_focus,
                language=language,
                country=country
            )
        else:
            # Use targeted prompt with analysis integration when corrected examples available
            prompt = create_targeted_prompt_with_analysis(
                low_confidence_examples=low_confidence_examples,
                entity_types=entity_types,
                domain_focus=domain_focus,
                language=language,
                country=country,
                final_summary=final_summary
            )
        
        try:
            # LLM call - exact parameters from your original
            response = ollama.generate(
                model=settings.ollama_model,
                prompt=prompt,
                options={
                    'top_k': 100,
                    'top_p': 0.8,
                    'num_predict': 800,
                    'temperature': 0.7,
                    'stop': ['<end>']
                }
            )
            
            # Parse JSON from response
            response_text = response['response'].strip()
            
            js = json.loads(response_text)
            synthetic_outputs.append(js)
            if logger:
                logger.debug(f"✅ Example {i+1} generated successfully")
            
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"❌ Failed to parse JSON for sample {i+1}: {e}")
                logger.debug(f"Response preview: {response_text[:200]}...")
            continue
        except Exception as e:
            if logger:
                logger.error(f"❌ Failed to generate sample {i+1}: {e}")
            continue
    
    if logger:
        logger.info("="*60)
        logger.info("INCREMENTAL GENERATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Successfully generated {len(synthetic_outputs)}/{num_samples_needed} new synthetic examples")
        logger.info(f"Prompt type used: {'BASELINE' if use_baseline_prompt else 'TEMPLATE-BASED'}")
        logger.info(f"Domain focus used: {domain_focus}")
    
    # Convert to NER format - exact process from your original
    ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
    if logger:
        logger.info(f"Initial NER formatted examples: {len(ner_formatted_data)}")
    
    # VALIDATE AND CLEAN THE DATA - exact validation from your original
    cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types, logger)
    
    if len(cleaned_data) == 0:
        if logger:
            logger.error("❌ No valid examples remain after cleaning")
        return synthetic_outputs, []
    
    # Check if we lost too many examples - exact logic from your original
    loss_rate = (len(ner_formatted_data) - len(cleaned_data)) / len(ner_formatted_data)
    if loss_rate > 0.5:  # More than 50% loss
        if logger:
            logger.warning(f"⚠️ High data loss during cleaning: {loss_rate*100:.1f}% of examples removed")
    
    if logger:
        logger.info(f"Final cleaned examples: {len(cleaned_data)}")
    
    return synthetic_outputs, cleaned_data


def get_or_create_synthetic_data(num_examples: int, num_synthetic_needed: int, 
                               low_confidence_examples: List[Dict], entity_types: List[str],
                               cache_manager, settings, skip_analysis: bool = False, 
                               final_summary: Optional[Dict] = None,
                               logger: Optional[logging.Logger] = None) -> List[Dict]:
    """
    Get synthetic data from cache or create incrementally - exact copy of your original function
    
    Args:
        num_examples: Number of corrected examples (cache key)
        num_synthetic_needed: Number of synthetic examples needed
        low_confidence_examples: List of corrected examples
        entity_types: List of entity types
        cache_manager: CacheManager instance
        settings: Settings object
        skip_analysis: Whether to skip analysis integration
        final_summary: Optional analysis summary
        logger: Optional logger
        
    Returns:
        List of synthetic examples in NER format
    """
    cache_key = num_examples
    
    # Get existing cached data
    cached_data = cache_manager.get_synthetic_data(cache_key)
    current_count = len(cached_data)
    
    if current_count >= num_synthetic_needed:
        if logger:
            logger.info(f"📦 Using cached synthetic data: {num_synthetic_needed}/{current_count} examples")
        return cached_data[:num_synthetic_needed]
    
    # Need to generate more synthetic data
    additional_needed = num_synthetic_needed - current_count
    if logger:
        logger.info(f"🔄 Need {additional_needed} more synthetic examples (have {current_count}, need {num_synthetic_needed})")
    
    # Handle zero corrected examples case - exact logic from your original
    if num_examples == 0:
        if logger:
            logger.info(f"⚠️ Generating synthetic data with ZERO corrected examples (baseline mode)")
        final_summary_to_use = None  # Never use analysis when no corrected examples
    elif skip_analysis:
        if logger:
            logger.info(f"⏭️ Generating synthetic data without analysis (skip_analysis=True)")
        final_summary_to_use = None
    else:
        if logger:
            logger.info(f"🧠 Generating synthetic data WITH domain analysis integration")
        final_summary_to_use = final_summary
    
    # Generate additional synthetic data - exact function call from your original
    synthetic_json, synthetic_ner = generate_synthetic_data_incremental(
        low_confidence_examples, entity_types, num_examples, additional_needed, 
        settings, final_summary_to_use, logger
    )
    
    # Check if we got valid data after cleaning
    if len(synthetic_ner) == 0:
        if logger:
            logger.error("❌ No valid synthetic data generated after cleaning")
        return []
    
    # Add to cache
    cache_manager.add_synthetic_data(cache_key, synthetic_ner)
    
    if logger:
        logger.info(f"💾 Updated synthetic cache: {len(cache_manager.get_synthetic_data(cache_key))} total examples for {num_examples} corrected examples")
    
    return cache_manager.get_synthetic_data(cache_key)[:num_synthetic_needed]
