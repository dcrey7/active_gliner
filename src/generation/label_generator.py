"""
Unified NER Label Generator
Uses backend + prompt + parsing + caching abstractions
Extracted and refactored from gemma_labeler.py, mistral_labeler.py, api_labeler.py
"""

from typing import List, Dict, Any
from tqdm import tqdm

from llm_backends import BackendFactory, LLMBackend
from prompting import StandardPromptBuilder, StructuredPromptBuilder
from parsing import ResponseParser
from caching import Cache, MemoryCache, DiskCache
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from utils.logging import get_logger


class NERLabelGenerator:
    """
    Unified NER label generator for training data generation

    Features:
    - Uses any LLM backend (Ollama, Mistral, Cerebras)
    - Automatic prompt strategy selection
    - Retry logic (max 3 attempts)
    - Caching (memory or disk)
    - Validation with cleaning
    """

    def __init__(
        self,
        backend: LLMBackend,
        cache: Cache,
        logger=None
    ):
        """
        Initialize NER label generator

        Args:
            backend: LLM backend instance
            cache: Cache instance (MemoryCache or DiskCache)
            logger: Logger instance (optional)
        """
        self.backend = backend
        self.cache = cache
        self.logger = logger or get_logger("NERLabelGenerator")

        # Choose prompt strategy based on backend capability
        if backend.supports_structured_output():
            self.prompt_builder = StructuredPromptBuilder()
            self.logger.info(f"Using StructuredPromptBuilder (backend supports schema validation)")
        else:
            self.prompt_builder = StandardPromptBuilder()
            self.logger.info(f"Using StandardPromptBuilder (normal JSON prompting)")

        # Response parser for extracting JSON
        self.parser = ResponseParser()

    def generate(
        self,
        low_n_examples: List[Dict],
        num_samples: int,
        entity_types: List[str],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Generate labels for low confidence examples

        Args:
            low_n_examples: Low confidence examples with tokenized_text
            num_samples: Number of examples to label
            entity_types: Entity types to identify
            verbose: Whether to show progress

        Returns:
            List of cleaned NER formatted examples
        """
        if verbose:
            self.logger.info("="*60)
            self.logger.info("NER LABEL GENERATION")
            self.logger.info("="*60)
            self.logger.info(f"Backend: {self.backend.model_name}")
            self.logger.info(f"Prompt strategy: {type(self.prompt_builder).__name__}")
            self.logger.info(f"Cache type: {type(self.cache).__name__}")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Low confidence examples available: {len(low_n_examples)}")
            self.logger.info(f"Target labels: {num_samples}")
            self.logger.info(f"Cached labels: {self.cache.size()}")
            self.logger.info("="*60)

        # Calculate how many new labels we actually need
        if self.cache.size() >= num_samples:
            if verbose:
                self.logger.info(f"Using {num_samples} labels from cache (no generation needed)")
            return self.cache.get_subset(num_samples)

        no_new_labels_needed = num_samples - self.cache.size()
        if verbose:
            self.logger.info(f"Need to generate {no_new_labels_needed} new labels ({self.cache.size()} already cached)")

        # Check if we have enough examples to label
        available_examples = len(low_n_examples) - self.cache.size()
        if available_examples < no_new_labels_needed:
            self.logger.warning(f"Not enough examples! Need {no_new_labels_needed}, have {available_examples}")
            no_new_labels_needed = available_examples

        synthetic_outputs = []

        # Generation loop with immediate retry logic (max 3 retries)
        for i in tqdm(range(no_new_labels_needed), desc="Labeling", disable=not verbose):
            # Get next example to label (skip already cached ones)
            example_idx = self.cache.size() + i
            example = low_n_examples[example_idx]
            # print(example)
            tokenized_text = example['tokenized_text']

            # Create labeling prompt
            prompt = self.prompt_builder.build(tokenized_text, entity_types)

            # Immediate retry logic (current approach: max 3 retries)
            max_retries = 3
            success = False

            for attempt in range(max_retries + 1):
                try:
                    # Generate with backend
                    response_text, input_tokens, output_tokens = self.backend.generate(prompt)

                    # Parse JSON response
                    js = self.parser.extract_json(response_text)
                    # print(f"js : {js}")
                    synthetic_outputs.append(js)
                    # print(f"synthetic_outputs: {synthetic_outputs}")
                    success = True

                    if verbose:
                        self.logger.info(f"Labeled example {i+1} {tokenized_text}: {js.get('entities', [])}")

                    break  # Success, exit retry loop

                except Exception as e:
                    error_msg = f"Labeling failed for example {i+1}, attempt {attempt+1}/{max_retries+1}: {str(e)[:100]}"
                    if verbose:
                        self.logger.warning(error_msg)
                    if attempt == max_retries:
                        if verbose:
                            self.logger.error(f"FINAL FAILURE: Example {i+1} failed after all retries")

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
        self.cache.extend(cleaned_data)

        if verbose:
            self.logger.info(f"Cache updated: {self.cache.size()} total labeled examples")

            # Show some stats for all data (cached + new)
            if self.cache.size() > 0:
                all_cached = self.cache.get_all()
                avg_entities = sum(len(ex['ner']) for ex in all_cached) / len(all_cached)
                self.logger.info(f"Average entities per example: {avg_entities:.1f}")

                # Entity type distribution
                entity_counts = {}
                for ex in all_cached:
                    for _, _, label in ex['ner']:
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                self.logger.info(f"Entity distribution: {entity_counts}")

            self.logger.info("="*60)

        # Return exactly num_samples (from cache + newly generated)
        return self.cache.get_subset(num_samples)


def create_label_generator(
    backend_type: str,
    model_name: str = None,
    cache_type: str = "memory",
    cache_model_name: str = None,
    use_structured_output: bool = False,
    logger=None
) -> NERLabelGenerator:
    """
    Factory function to create NER label generator with all components

    Args:
        backend_type: Type of backend ('ollama', 'mistral', 'cerebras')
        model_name: Model name (optional, uses backend default if None)
        cache_type: Type of cache ('memory' or 'disk')
        cache_model_name: Model name for cache folder (uses model_name if None)
        use_structured_output: Use structured output if available (for Cerebras)
        logger: Logger instance (optional)

    Returns:
        Configured NERLabelGenerator instance

    Example:
        # Simple usage
        generator = create_label_generator('ollama', model_name='gemma3:12b')

        # With disk cache
        generator = create_label_generator(
            backend_type='cerebras',
            model_name='qwen-3-235b-a22b-instruct-2507',
            cache_type='disk',
            cache_model_name='qwen_3_235b'
        )

        # With structured output
        generator = create_label_generator(
            backend_type='cerebras',
            model_name='qwen-3-235b-a22b-thinking-2507',
            use_structured_output=True,
            cache_type='disk'
        )
    """
    # Create backend
    backend = BackendFactory.create(
        backend_type=backend_type,
        model_name=model_name,
        use_structured_output=use_structured_output
    )

    # Create cache
    if cache_type == 'disk':
        cache_name = cache_model_name or model_name or backend.model_name
        cache = DiskCache(
            cache_type="labelling",
            model_name=cache_name
        )
    else:
        cache = MemoryCache()

    # Create generator
    generator = NERLabelGenerator(
        backend=backend,
        cache=cache,
        logger=logger
    )

    return generator
