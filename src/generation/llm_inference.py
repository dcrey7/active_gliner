"""
Unified LLM Inference for NER
Handles both training label generation and evaluation prediction

Single class, two modes:
- Mode 'training': Strict validation, removes invalid examples (for fine-tuning)
- Mode 'evaluation': Non-strict validation, preserves all indices (for testing)

Replaces:
- generation/label_generator.py (NERLabelGenerator)
- evaluation/ner_evaluator.py (NEREvaluator)
"""

from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from llm_backends.base import LLMBackend
from prompting.standard_prompt import StandardPromptBuilder
from prompting.structured_prompt import StructuredPromptBuilder
from parsing.response_parser import ResponseParser
from caching.base import Cache
from data.transforms import convert_synthetic_to_ner_format
from data.validator import NERValidator
from utils.logging import get_logger


class LLMInference:
    """
    Unified LLM inference for NER tasks
    
    Supports two modes:
    1. 'training': For generating training labels
       - Strict validation (removes invalid examples)
       - Used with label_generator factory
       
    2. 'evaluation': For generating test predictions
       - Non-strict validation (preserves all indices)
       - Used with predictor factory
    
    Example:
        # Training mode
        inference = LLMInference(backend, cache, entity_types, mode='training')
        labels = inference.generate(unlabeled_examples, entity_types)
        
        # Evaluation mode
        inference = LLMInference(backend, cache, entity_types, mode='evaluation')
        predictions = inference.generate(test_examples, entity_types)
    """

    def __init__(
        self,
        backend: LLMBackend,
        cache: Cache,
        entity_types: List[str],
        mode: str = 'training',
        logger=None
    ):
        """
        Initialize LLM inference engine
        
        Args:
            backend: LLM backend instance (Ollama, Mistral, Cerebras)
            cache: Cache instance (MemoryCache or DiskCache)
            entity_types: List of valid entity types
            mode: 'training' (strict) or 'evaluation' (preserve indices)
            logger: Logger instance (optional)
        """
        if mode not in ['training', 'evaluation']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'training' or 'evaluation'")
        
        self.backend = backend
        self.cache = cache
        self.entity_types = entity_types
        self.mode = mode
        self.logger = logger or get_logger("LLMInference")

        # Choose prompt strategy based on backend capability
        if backend.supports_structured_output():
            self.prompt_builder = StructuredPromptBuilder()
            self.logger.info(f"Using StructuredPromptBuilder (backend supports schema)")
        else:
            self.prompt_builder = StandardPromptBuilder()
            self.logger.info(f"Using StandardPromptBuilder (normal JSON)")

        # Response parser
        self.parser = ResponseParser()
        
        # Validator with mode-specific behavior
        self.validator = NERValidator(entity_types=entity_types, logger=logger)
        
        self.logger.info(f"LLMInference initialized in '{mode}' mode")

    def generate(
        self,
        examples: List[Dict],
        entity_types: List[str],
        num_samples: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate labels/predictions for examples
        
        Args:
            examples: Examples with 'tokenized_text' (and optionally 'ner' for ground truth)
            entity_types: Entity types to identify
            num_samples: Number of examples to process (default: all)
            verbose: Show progress bars and logs
            
        Returns:
            Dictionary with:
            - 'all_labels': List of labeled/predicted examples
            - 'total_input_tokens': Total input tokens used
            - 'total_output_tokens': Total output tokens used
        """
        if num_samples is None:
            num_samples = len(examples)
        
        if verbose:
            mode_desc = "TRAINING LABEL GENERATION" if self.mode == 'training' else "EVALUATION PREDICTION"
            self.logger.info("=" * 60)
            self.logger.info(mode_desc)
            self.logger.info("=" * 60)
            self.logger.info(f"Backend: {self.backend.model_name}")
            self.logger.info(f"Prompt strategy: {type(self.prompt_builder).__name__}")
            self.logger.info(f"Cache type: {type(self.cache).__name__}")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Validation: {'STRICT (removes invalid)' if self.mode == 'training' else 'NON-STRICT (preserves indices)'}")
            self.logger.info(f"Entity types: {entity_types}")
            self.logger.info(f"Examples available: {len(examples)}")
            self.logger.info(f"Target samples: {num_samples}")
            self.logger.info(f"Cached: {self.cache.size()}")
            self.logger.info("=" * 60)

        # Token tracking
        total_input_tokens = 0
        total_output_tokens = 0

        # Check cache
        if self.cache.size() >= num_samples:
            if verbose:
                self.logger.info(f"Using {num_samples} from cache (no generation needed)")
            return {
                'all_labels': self.cache.get_subset(num_samples),
                'total_input_tokens': 0,
                'total_output_tokens': 0
            }

        # Calculate new examples needed
        num_new_needed = num_samples - self.cache.size()
        if verbose:
            self.logger.info(f"Need to generate {num_new_needed} new ({self.cache.size()} cached)")

        # Check availability
        available = len(examples) - self.cache.size()
        if available < num_new_needed:
            self.logger.warning(f"Not enough examples! Need {num_new_needed}, have {available}")
            num_new_needed = available

        synthetic_outputs = []

        # Generation loop with retry logic
        desc = "Labeling" if self.mode == 'training' else "Predicting"
        for i in tqdm(range(num_new_needed), desc=desc, disable=not verbose):
            # Get next example (skip cached)
            example_idx = self.cache.size() + i
            example = examples[example_idx]
            tokenized_text = example['tokenized_text']

            # Create prompt
            prompt = self.prompt_builder.build(tokenized_text, entity_types)

            # Retry logic (max 3 attempts)
            max_retries = 3
            success = False

            for attempt in range(max_retries + 1):
                try:
                    # Generate with backend
                    response_text, input_tokens, output_tokens = self.backend.generate(prompt)

                    # Track tokens
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    # Parse JSON response
                    js = self.parser.extract_json(response_text)
                    synthetic_outputs.append(js)
                    success = True

                    # if verbose and i % 20 == 0:
                    self.logger.info(f"Generated {i+1}/{num_new_needed}: {tokenized_text[:5]}... -> {js.get('entities', [])[:2]}")

                    break  # Success

                except Exception as e:
                    if verbose and attempt == max_retries:
                        self.logger.error(f"Failed example {i+1} after {max_retries+1} attempts: {str(e)[:100]}")
                    if attempt == max_retries:
                        # Add empty for failed examples in evaluation mode
                        if self.mode == 'evaluation':
                            synthetic_outputs.append({
                                "text": " ".join(tokenized_text),
                                "entities": []
                            })

        if verbose:
            self.logger.info(f"Successfully generated {len(synthetic_outputs)}/{num_new_needed}")

        # Convert to NER format
        ner_formatted = convert_synthetic_to_ner_format(synthetic_outputs)
        if verbose:
            self.logger.info(f"Converted to NER format: {len(ner_formatted)} examples")

        # Validate with mode-specific behavior
        strict = (self.mode == 'training')
        cleaned_data, report = self.validator.validate(ner_formatted, strict=strict)

        if verbose:
            self.logger.info(f"Validation complete: {len(cleaned_data)} valid")
            self.logger.info(report.summary())

        # Add to cache
        self.cache.extend(cleaned_data)

        if verbose:
            self.logger.info(f"Cache updated: {self.cache.size()} total")
            
            # Stats
            if self.cache.size() > 0:
                all_cached = self.cache.get_all()
                avg_entities = sum(len(ex['ner']) for ex in all_cached) / len(all_cached)
                self.logger.info(f"Average entities per example: {avg_entities:.1f}")

            # Token usage
            if total_input_tokens > 0:
                self.logger.info(f"Token usage: {total_input_tokens} input, {total_output_tokens} output")
            
            self.logger.info("=" * 60)

        # Return results
        return {
            'all_labels': self.cache.get_subset(num_samples),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens
        }


# Factory functions for convenience

def create_llm_train_labels(
    backend_type: str,
    model_name: str = None,
    entity_types: List[str] = None,
    cache_type: str = "memory",
    use_structured_output: bool = False,
    logger=None
) -> LLMInference:
    """
    Factory for creating LLM label generator (TRAINING mode)
    
    Args:
        backend_type: 'ollama', 'mistral', or 'cerebras'
        model_name: Model name (optional, uses backend default)
        entity_types: Valid entity types
        cache_type: 'memory' or 'disk'
        use_structured_output: Use structured output (Cerebras only)
        logger: Logger instance
        
    Returns:
        LLMInference instance in training mode
    """
    from llm_backends.factory import BackendFactory
    from caching.memory_cache import MemoryCache
    from caching.disk_cache import DiskCache
    
    # Create backend
    backend = BackendFactory.create(
        backend_type=backend_type,
        model_name=model_name,
        use_structured_output=use_structured_output
    )
    
    # Create cache
    if cache_type == 'disk':
        cache_name = model_name or backend.model_name
        cache = DiskCache(cache_type="labelling", model_name=cache_name)
    else:
        cache = MemoryCache()
    
    # Create inference in training mode
    return LLMInference(
        backend=backend,
        cache=cache,
        entity_types=entity_types or [],
        mode='training',
        logger=logger
    )


def create_llm_eval_labels(
    backend_type: str,
    model_name: str = None,
    entity_types: List[str] = None,
    cache_type: str = "disk",
    use_structured_output: bool = False,
    logger=None
) -> LLMInference:
    """
    Factory for creating LLM predictor (EVALUATION mode)
    
    Args:
        backend_type: 'ollama', 'mistral', or 'cerebras'
        model_name: Model name (optional, uses backend default)
        entity_types: Valid entity types
        cache_type: 'memory' or 'disk' (default: disk for persistence)
        use_structured_output: Use structured output (Cerebras only)
        logger: Logger instance
        
    Returns:
        LLMInference instance in evaluation mode
    """
    from llm_backends.factory import BackendFactory
    from caching.memory_cache import MemoryCache
    from caching.disk_cache import DiskCache
    
    # Create backend
    backend = BackendFactory.create(
        backend_type=backend_type,
        model_name=model_name,
        use_structured_output=use_structured_output
    )
    
    # Create cache (default disk for evaluation)
    if cache_type == 'disk':
        cache_name = model_name or backend.model_name
        cache = DiskCache(cache_type="evaluation", model_name=cache_name)
    else:
        cache = MemoryCache()
    
    # Create inference in evaluation mode
    return LLMInference(
        backend=backend,
        cache=cache,
        entity_types=entity_types or [],
        mode='evaluation',
        logger=logger
    )
