"""
Simple Synthetic Data Generator with Token Tracking
Sweet, Simple, and Solid - Fully Generic Approach with Token Metrics

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

import ollama
import json
import random
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data


class SyntheticDataGenerator:
    """Simple, generic synthetic data generator using Ollama with custom Mistral model and token tracking"""
    
    def __init__(self, model_name: str = "gemma3:12b"):
        """
        Initialize the generator
        
        Args:
            model_name: Ollama model to use for generation (default: mistral_32k with 32k context)
        """

        self.model_name = model_name
        print(f"{self.model_name}")
        print(self.model_name)
    

    
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
    
    def generate(self, corrected_examples: List[Dict], num_samples: int, 
                entity_types: List[str], countries: List[str], genres: List[str], syn_cache: List[str],
                subject: str, verbose: bool = True) -> Tuple[List[Dict], float, Dict[str, Any]]:
        """
        Generate synthetic data with full user control and exact token tracking from Ollama
        
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
            print("="*60)
            print("SYNTHETIC DATA GENERATION")
            print("="*60)
            print(f"Model: {self.model_name}")
            print(f"Subject: {subject}")
            print(f"Entity types: {entity_types}")
            print(f"Countries: {countries}")
            print(f"Genres: {genres}")
            print(f"Template examples: {len(corrected_examples)}")
            print(f"Target samples: {num_samples}")
            print(f"Cached samples: {len(syn_cache)}")
            print("="*60)
        
        # Initialize token tracking
        input_tokens_list = []
        output_tokens_list = []
        
        # Model configuration (from your current settings)
        model_context_limit = 128000
        model_output_limit = 800
        model_limits = (model_context_limit, model_output_limit)
        
        # Calculate how many new samples we actually need
        if len(syn_cache) >= num_samples:
            if verbose:
                print(f"✅ Using {num_samples} samples from cache (no generation needed)")
            
            avg_entities = sum(len(ex['ner']) for ex in syn_cache[:num_samples]) / len(syn_cache[:num_samples])
            
            token_metrics = {
                'avg_input_tokens': 0,  # No generation, so no input
                'model_input_output': model_limits,
                'avg_output_tokens': 0,  # No generation, so no output
            }
            
            if verbose:
                print(f"📊 Token metrics (cached): model_limits={model_limits}, no generation needed")
            
            return syn_cache[:num_samples], avg_entities, token_metrics
        
        no_new_syn_needed = num_samples - len(syn_cache)
        if verbose:
            print(f"📝 Need to generate {no_new_syn_needed} new samples ({len(syn_cache)} already cached)")
        
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
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Generate with Ollama (using custom model with 32k context)
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={
                            'top_k': 100,
                            'top_p': 0.8,
                            'num_predict': model_output_limit,
                            'temperature': 0.7,
                            'stop': ['<end>'],
                            'num_ctx': model_context_limit
                        },
                        
                    )
                    
                    # Extract EXACT token counts from Ollama response
                    input_tokens = response.get('prompt_eval_count', 0)  # Exact input tokens
                    output_tokens = response.get('eval_count', 0)        # Exact output tokens
                    
                    input_tokens_list.append(input_tokens)
                    output_tokens_list.append(output_tokens)
                    
                    # Parse JSON response
                    raw_response = response['response'].strip()
                    response_text = raw_response
                    # Clean up response (remove any markdown formatting)
                    if response_text.startswith('```json'):
                        response_text = response_text.replace('```json', '').replace('```', '').strip()
                    
                    js = json.loads(response_text)
                    synthetic_outputs.append(js)
                    success = True
                    print(f"used [{country,subject,genre} to get sample: {js}] ")
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"⚠️ JSON parsing failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)  # flush=True for Jupyter
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
                        print(f"\n{error_msg}", flush=True)  # flush=True for Jupyter
                        sys.stdout.flush()  # Force flush for Jupyter notebooks
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                        # Add zeros for failed attempts (no token data available)
                        input_tokens_list.append(0)
                        output_tokens_list.append(0)
                
            if verbose and success and i % 10 == 0:
                # Print token metrics every 10 samples using EXACT counts
                avg_input_so_far = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
                avg_output_so_far = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
                print(f"\n✅ Generated {i+1}/{no_new_syn_needed} samples...")
                print(f"📊 EXACT Token metrics: avg_input={avg_input_so_far:.0f}, avg_output={avg_output_so_far:.0f}, limits={model_limits}", flush=True)
                sys.stdout.flush()
        
        if verbose:
            print(f"\n✅ Successfully generated {len(synthetic_outputs)}/{no_new_syn_needed} raw samples")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            print(f"📝 Converted to NER format: {len(ner_formatted_data)} examples")
        
        # Clean and validate using existing pipeline
        cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types)
        
        if verbose:
            print(f"🧹 Final cleaned examples: {len(cleaned_data)}")
        
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
            print(f"💾 Cache updated: {len(syn_cache)} total samples")
            print("="*60)
            
            # Show token metrics
            print(f"📊 TOKEN METRICS (EXACT FROM OLLAMA):")
            print(f"   Average input tokens: {token_metrics['avg_input_tokens']:.0f} (EXACT)")
            print(f"   Model limits (input,output): {token_metrics['model_input_output']}")
            print(f"   Average output tokens: {token_metrics['avg_output_tokens']:.0f} (EXACT)")
            
            # Warning if approaching limits
            if token_metrics['avg_input_tokens'] > model_context_limit * 0.9:
                print(f"⚠️  WARNING: Input tokens approaching context limit!")
            if token_metrics['avg_output_tokens'] > model_output_limit * 0.9:
                print(f"⚠️  WARNING: Output tokens approaching generation limit!")
            
            # Show some stats for all data (cached + new)
            if samples_for_stats:
                total_entities = sum(len(ex['ner']) for ex in samples_for_stats)
                print(f"📊 Average entities per example: {avg_entities:.1f}")
                
                # Entity type distribution
                entity_counts = {}
                for ex in samples_for_stats:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                print(f"📈 Entity distribution: {entity_counts}")
            print("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return syn_cache[:num_samples], avg_entities, token_metrics