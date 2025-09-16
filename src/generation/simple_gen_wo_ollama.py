"""
Direct Mistral Synthetic Data Generator
Using HuggingFace Transformers - No Ollama bugs!

Usage:
    generator = SyntheticDataGenerator()
    data = generator.generate(
        corrected_examples=examples,
        num_samples=100,
        entity_types=["PERSON", "ORG", "LOCATION"],
        countries=["USA", "France", "Japan"],
        genres=["news articles", "reports"],
        subject="healthcare"
    )
"""

import json
import random
import sys
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data


class SyntheticDataGenerator:
    """Direct Mistral synthetic data generator using HuggingFace transformers"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """
        Initialize the generator
        
        Args:
            model_name: HuggingFace model to use for generation
        """
        self.model_name = model_name
        print(f"🚀 Loading {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model max context: {self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else '32768 (estimated)'}")
    
    def _create_prompt(self, corrected_examples: List[Dict], entity_types: List[str], 
                      subject: str, country: str, genre: str) -> str:
        """
        Create prompt - ACTUALLY limit examples to prevent huge prompts
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
        
        # ACTUALLY limit to 2 examples to prevent massive prompts
        limited_examples = corrected_examples[:2]
        
        for i, example in enumerate(limited_examples):
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

    def _generate_with_model(self, prompt: str) -> str:
        """Generate text using the loaded model"""
        # Format as chat message for instruct model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=30000,  # Use most of the 32k context, leave room for response
            return_dict=True
        )
        
        # Check prompt size
        input_length = inputs['input_ids'].shape[1]
        print(f"📏 Prompt tokens: {input_length}", flush=True)
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.7,
                do_sample=True,
                top_p=0.8,
                top_k=100,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens (response)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def generate(self, corrected_examples: List[Dict], num_samples: int, 
                entity_types: List[str], countries: List[str], genres: List[str], 
                subject: str, verbose: bool = True) -> List[Dict]:
        """
        Generate synthetic data with full user control
        """
        if verbose:
            print("="*60)
            print("DIRECT MISTRAL DATA GENERATION")
            print("="*60)
            print(f"Model: {self.model_name}")
            print(f"Subject: {subject}")
            print(f"Entity types: {entity_types}")
            print(f"Countries: {countries}")
            print(f"Genres: {genres}")
            print(f"Template examples: {len(corrected_examples)} (using first 2)")
            print(f"Target samples: {num_samples}")
            print("="*60)
        
        synthetic_outputs = []
        
        # Generation loop with immediate retry logic
        for i in tqdm(range(num_samples), desc="Generating", disable=not verbose):
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
            
            for attempt in range(max_retries + 1):
                try:
                    # Generate with direct model call
                    response_text = self._generate_with_model(prompt)
                    
                    # Clean up response (remove any markdown formatting)
                    if response_text.startswith('```json'):
                        response_text = response_text.replace('```json', '').replace('```', '').strip()
                    
                    # Handle cases where model adds extra text before/after JSON
                    if '{' in response_text and '}' in response_text:
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        response_text = response_text[start_idx:end_idx]
                    
                    js = json.loads(response_text)
                    synthetic_outputs.append(js)
                    success = True
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    error_msg = f"⚠️ JSON parsing failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)
                        print(f"🔍 Raw response: {response_text[:200]}...", flush=True)
                        sys.stdout.flush()
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                            
                except Exception as e:
                    error_msg = f"❌ Generation failed for sample {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        print(f"\n{error_msg}", flush=True)
                        sys.stdout.flush()
                    if attempt == max_retries:
                        if verbose:
                            print(f"❌ FINAL FAILURE: Sample {i+1} failed after all retries", flush=True)
                
            if verbose and success and i % 10 == 0:
                print(f"\n✅ Generated {i+1}/{num_samples} samples...", flush=True)
                sys.stdout.flush()
        
        if verbose:
            print(f"\n✅ Successfully generated {len(synthetic_outputs)}/{num_samples} raw samples")
        
        # Convert to NER format using existing pipeline
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            print(f"📝 Converted to NER format: {len(ner_formatted_data)} examples")
        
        # Clean and validate using existing pipeline
        cleaned_data = validate_and_clean_ner_data(ner_formatted_data, entity_types)
        
        if verbose:
            print(f"🧹 Final cleaned examples: {len(cleaned_data)}")
            print("="*60)
            
            # Show some stats
            if cleaned_data:
                total_entities = sum(len(ex['ner']) for ex in cleaned_data)
                avg_entities = total_entities / len(cleaned_data)
                print(f"📊 Average entities per example: {avg_entities:.1f}")
                
                # Entity type distribution
                entity_counts = {}
                for ex in cleaned_data:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                print(f"📈 Entity distribution: {entity_counts}")
            print("="*60)
        
        return cleaned_data


# Memory cleanup function
def cleanup_model():
    """Call this to free GPU memory when done"""
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU memory cleaned up")