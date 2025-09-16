"""
Simple Synthetic Data Generator with Mistral Inference
Sweet, Simple, and Solid - Fully Generic Approach

Usage:
    generator = SyntheticDataGenerator()
    data = generator.generate(
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
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data


class SyntheticDataGenerator:
    """Simple, generic synthetic data generator using Mistral Inference"""
    
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
        
        print(f"Loading Mistral model from: {self.model_path}")
        
        # Initialize tokenizer and model
        tokenizer_path = self.model_path / "tokenizer.model.v3"
        self.tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
        self.model = Transformer.from_folder(self.model_path)
        self.model_name = "MISTRAL 7B v0.3 (mistral inference)"
        print(f"✅ Mistral model loaded successfully")
        print(f"Mistral model loaded from {self.model_path}")
    
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
    
    def _generate_single_sample(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """
        Generate a single sample using Mistral inference
        
        Args:
            prompt: The generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated text response
        """
        # Create chat completion request
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        
        # Encode the prompt
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        
        # Generate response
        out_tokens, _ = generate(
            [tokens], 
            self.model, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        
        # Decode the response
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        
        return result
    
    def generate(self, corrected_examples: List[Dict], num_samples: int, 
                entity_types: List[str], countries: List[str], genres: List[str], syn_cache: List[str],
                subject: str, verbose: bool = True) -> List[Dict]:
        """
        Generate synthetic data with full user control
        
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
            List of cleaned NER formatted examples (cached + newly generated)
        """
        if verbose:
            print("="*60)
            print("SYNTHETIC DATA GENERATION (MISTRAL)")
            print("="*60)
            print(f"Model path: {self.model_path}")
            print(f"Subject: {subject}")
            print(f"Entity types: {entity_types}")
            print(f"Countries: {countries}")
            print(f"Genres: {genres}")
            print(f"Template examples: {len(corrected_examples)}")
            print(f"Target samples: {num_samples}")
            print(f"Cached samples: {len(syn_cache)}")
            print("="*60)
        
        # Calculate how many new samples we actually need
        if len(syn_cache) >= num_samples:
            if verbose:
                print(f"✅ Using {num_samples} samples from cache (no generation needed)")
            return syn_cache[:num_samples]
        
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
            print(f"generating {i+1} sample")
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Generate with Mistral inference
                    response_text = self._generate_single_sample(
                        prompt=prompt,
                        max_tokens=800,
                        temperature=0.7
                    )
                    
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
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"⚠️ JSON parsing failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)  # flush=True for Jupyter
                        sys.stdout.flush()  # Force flush for Jupyter notebooks
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                except Exception as e:
                    error_msg = f"❌ Generation failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)  # flush=True for Jupyter
                        sys.stdout.flush()  # Force flush for Jupyter notebooks
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                
            if verbose and success and i % 10 == 0:
                print(f"\n✅ Generated {i+1}/{no_new_syn_needed} samples...", flush=True)
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
        
        if verbose:
            print(f"💾 Cache updated: {len(syn_cache)} total samples")
            print("="*60)
            
            # Show some stats for all data (cached + new)
            if syn_cache:
                # Use first num_samples for stats
                samples_for_stats = syn_cache[:num_samples]
                total_entities = sum(len(ex['ner']) for ex in samples_for_stats)
                avg_entities = total_entities / len(samples_for_stats)
                print(f"📊 Average entities per example: {avg_entities:.1f}")
                
                # Entity type distribution
                entity_counts = {}
                for ex in samples_for_stats:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                print(f"📈 Entity distribution: {entity_counts}")
            print("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return syn_cache[:num_samples],avg_entities