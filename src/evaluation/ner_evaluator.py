"""
NER Evaluator using LLM
Uses backend + prompt + parsing + caching abstractions
Preserves all indices (empty NER if invalid) for evaluation consistency
"""

from typing import List, Dict, Any
from tqdm import tqdm

from llm_backends.factory import BackendFactory
from llm_backends.base import LLMBackend
from prompting.standard_prompt import StandardPromptBuilder
from prompting.structured_prompt import StructuredPromptBuilder
from parsing.response_parser import ResponseParser
from caching.base import Cache
from caching.memory_cache import MemoryCache
from caching.disk_cache import DiskCache
from data.transforms import convert_synthetic_to_ner_format
from data.validator import NERValidator
from utils.logging import get_logger


class NEREvaluator:
    """
    NER evaluator for generating predictions using LLM

    Key Difference from LabelGenerator:
    - MUST preserve all indices (empty NER if invalid)
    - Used for evaluation, not training
    """

    def __init__(
        self,
        backend: LLMBackend,
        cache: Cache,
        logger=None
    ):
        """
        Initialize NER evaluator

        Args:
            backend: LLM backend instance
            cache: Cache instance (MemoryCache or DiskCache)
            logger: Logger instance (optional)
        """
        self.backend = backend
        self.cache = cache
        self.logger = logger or get_logger("NEREvaluator")

        # Choose prompt strategy based on backend capability
        if backend.supports_structured_output():
            self.prompt_builder = StructuredPromptBuilder()
            self.logger.info(f"Using StructuredPromptBuilder (backend supports schema validation)")
        else:
            self.prompt_builder = StandardPromptBuilder()
            self.logger.info(f"Using StandardPromptBuilder (normal JSON prompting)")

        # Response parser for extracting JSON
        self.parser = ResponseParser()

    def evaluate(
        self,
        test_data: List[Dict],
        entity_types: List[str],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Generate predictions for test examples

        CRITICAL: Must preserve all indices - return empty NER for invalid examples

        Args:
            test_data: Test dataset with tokenized_text
            entity_types: List of entity types
            verbose: Whether to show progress

        Returns:
            List of NER predictions (same length as test_data, indices preserved)
        """
        if verbose:
            self.logger.info("="*60)
            self.logger.info("NER EVALUATION")
            self.logger.info("="*60)
            self.logger.info(f"Backend: {self.backend.model_name}")
            self.logger.info(f"Prompt strategy: {type(self.prompt_builder).__name__}")
            self.logger.info(f"Cache type: {type(self.cache).__name__}")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Test examples: {len(test_data)}")
            self.logger.info(f"Cached evaluations: {self.cache.size()}")
            self.logger.info("="*60)

        # Calculate how many new evaluations we need
        if self.cache.size() >= len(test_data):
            if verbose:
                self.logger.info(f"Using {len(test_data)} evaluations from cache (no evaluation needed)")
            return self.cache.get_subset(len(test_data))

        no_new_evals_needed = len(test_data) - self.cache.size()
        if verbose:
            self.logger.info(f"Need to evaluate {no_new_evals_needed} new examples ({self.cache.size()} already cached)")

        synthetic_outputs = []

        # Evaluation loop with retry logic
        for i in tqdm(range(no_new_evals_needed), desc="Evaluating", disable=not verbose):
            # Get next example to evaluate (skip already cached ones)
            example_idx = self.cache.size() + i
            example = test_data[example_idx]
            tokenized_text = example['tokenized_text']

            # Create evaluation prompt
            prompt = self.prompt_builder.build(tokenized_text, entity_types)

            # Retry logic (max 3 retries)
            max_retries = 3
            success = False

            for attempt in range(max_retries + 1):
                try:
                    # Generate with backend
                    response_text, input_tokens, output_tokens = self.backend.generate(prompt)

                    # Parse JSON response
                    js = self.parser.extract_json(response_text)
                    synthetic_outputs.append(js)
                    success = True

                    if verbose:
                        self.logger.info(f"Evaluated example {i+1}: {tokenized_text} -> {js.get('entities', [])}")

                    break

                except Exception as e:
                    if attempt == max_retries:
                        self.logger.error(f"Failed to evaluate example {i+1} after {max_retries+1} attempts")
                        # Add empty prediction for failed examples (PRESERVE INDEX!)
                        synthetic_outputs.append({
                            "text": " ".join(tokenized_text),
                            "entities": []
                        })
                        break

        if verbose:
            self.logger.info(f"Successfully evaluated {len(synthetic_outputs)}/{no_new_evals_needed} new examples")

        # Convert to NER format
        ner_formatted_data = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            self.logger.info(f"Converted to NER format: {len(ner_formatted_data)} examples")

        # CRITICAL: Validate with strict=False to preserve indices
        validator = NERValidator(entity_types, self.logger)
        cleaned_predictions, report = validator.validate(ner_formatted_data, strict=False)

        if verbose:
            self.logger.info(f"Final predictions: {len(cleaned_predictions)} (index-preserved)")
            self.logger.info(report.summary())

        # Add to cache
        self.cache.extend(cleaned_predictions)

        if verbose:
            self.logger.info(f"Cache updated: {self.cache.size()} total evaluations")
            self.logger.info("="*60)

        # Return exactly len(test_data) predictions (from cache + newly evaluated)
        return self.cache.get_subset(len(test_data))


def create_ner_evaluator(
    backend_type: str,
    model_name: str = None,
    cache_type: str = "memory",
    cache_model_name: str = None,
    use_structured_output: bool = False,
    logger=None
) -> NEREvaluator:
    """
    Factory function to create NER evaluator

    Args:
        backend_type: Type of backend ('ollama', 'mistral', 'cerebras')
        model_name: Model name (optional)
        cache_type: Type of cache ('memory' or 'disk')
        cache_model_name: Model name for cache folder
        use_structured_output: Use structured output if available
        logger: Logger instance (optional)

    Returns:
        Configured NEREvaluator instance

    Example:
        evaluator = create_ner_evaluator(
            backend_type='ollama',
            model_name='gemma3:12b',
            cache_type='disk',
            cache_model_name='gemma3_12b'
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
            cache_type="evaluation",  # Use evaluation cache type
            model_name=cache_name
        )
    else:
        cache = MemoryCache()

    # Create evaluator
    evaluator = NEREvaluator(
        backend=backend,
        cache=cache,
        logger=logger
    )

    return evaluator
