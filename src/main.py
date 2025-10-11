"""
Active GLiNER - Main Entry Point
Public API for information extraction with GLiNER and LLM distillation

Five core functions:
1. zeroshot()  - Zero-shot prediction with GLiNER (optional LoRA)
2. ranking()   - Rank examples by uncertainty for active learning
3. finetune()  - Fine-tune GLiNER with GT + LLM mixed labels
4. predict()   - Predict with fine-tuned adapter
5. evaluate()  - Evaluate predictions or models
"""
import os
import gc
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

from models.gloner import GLONER
from training.trainer import train_lora_model
from evaluation.eval import evaluate_gloner, evaluate_llm
from generation.llm_inference import create_llm_train_labels
from data.transforms import create_mixed_training_data
from selection.strategies import get_highest_mse_examples_sorted, get_lowest_score_examples_sorted
from config.lora_defaults import DEFAULT_GLINER_MODEL, DEFAULT_LORA_CONFIG
from config.training_config import TRAINING_CONFIG
from utils.logging import get_logger


def zeroshot(
    data: List[Dict[str, Any]],
    entity_types: List[str],
    base_model_path: str = None,
    lora_config: Dict = None,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "cuda",
    logger = None
) -> List:
    """
    Zero-shot prediction with GLiNER model (optionally with LoRA initialization)

    Args:
        data: List of examples (NER format with tokenized_text field)
        entity_types: List of entity types to extract
        base_model_path: Path to base GLiNER model (default: from lora_defaults)
        lora_config: LoRA configuration dict (if provided, initializes LoRA)
        threshold: Confidence threshold for predictions
        batch_size: Batch size for inference
        device: Device to run on (cuda or cpu)
        logger: Optional logger instance

    Returns:
        predictions: GLiNER format predictions

    Example:
        predictions = zeroshot(
            data=test_data,
            entity_types=['person', 'location'],
            threshold=0.5
        )
    """
    if logger is None:
        logger = get_logger("zeroshot")

    logger.info(f"Running zero-shot prediction on {len(data)} examples")
    logger.info(f"Entity types: {entity_types}")

    # Initialize model
    if lora_config:
        gloner = GLONER.for_training(
            base_model_path=base_model_path,
            lora_config=lora_config,
            logger=logger
        )
    else:
        gloner = GLONER.for_training(
            base_model_path=base_model_path,
            logger=logger
        )

    device_obj = torch.device(device)
    gloner.to(device_obj)

    # Predict
    predictions = gloner.predict(
        data=data,
        entity_types=entity_types,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
        flat_ner=True
    )

    logger.info(f"Generated predictions for {len(predictions)} examples")

    # Cleanup
    del gloner
    torch.cuda.empty_cache()
    gc.collect()

    return predictions


def ranking(
    predictions: List,
    data: List[Dict[str, Any]],
    entity_types: List[str],
    n_examples: int,
    strategy: str = "mse",
    logger = None
) -> List[Dict[str, Any]]:
    """
    Rank examples by uncertainty for active learning selection

    Args:
        predictions: GLiNER predictions with confidence scores
        data: Original data (must match predictions)
        entity_types: Entity types used
        n_examples: Number of top uncertain examples to return
        strategy: Ranking strategy (mse for MSE-based, min_score for minimum confidence)
        logger: Optional logger instance

    Returns:
        ranked_examples: Top n uncertain examples in ranked order

    Example:
        uncertain_examples = ranking(
            predictions=predictions,
            data=test_data,
            entity_types=['person', 'location'],
            n_examples=100,
            strategy='mse'
        )
    """
    if logger is None:
        logger = get_logger("ranking")

    logger.info(f"Ranking {len(data)} examples by uncertainty")
    logger.info(f"Strategy: {strategy}, requesting top {n_examples}")

    # Evaluate to get results with scores
    eval_results = evaluate_gloner(
        predictions=predictions,
        data=data,
        entity_types=entity_types,
        has_ground_truth=True if "ner" in data[0] else False
    )

    # Select strategy
    if strategy == "mse":
        ranked = get_highest_mse_examples_sorted(
            eval_results,
            n=n_examples,
            logger=logger
        )
    elif strategy == "min_score":
        ranked = get_lowest_score_examples_sorted(
            eval_results,
            n=n_examples,
            logger=logger
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'mse' or 'min_score'")

    logger.info(f"Selected {len(ranked)} most uncertain examples")
    return ranked


def finetune(
    training_data: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
    entity_types: List[str],
    adapter_save_path: str,
    base_model_path: str = None,
    llm_backend: str = None,
    llm_model: str = None,
    mix_ratio: int = 100,
    training_config: Dict = None,
    lora_config: Dict = None,
    device: str = "cuda",
    logger = None
) -> str:
    """
    Fine-tune GLiNER with LoRA on GT + LLM mixed labels

    Args:
        training_data: Training examples with ground truth labels
        eval_data: Evaluation data for validation
        entity_types: List of entity types
        adapter_save_path: Path to save trained LoRA adapter
        base_model_path: Base GLiNER model path (default: from lora_defaults)
        llm_backend: LLM backend for label generation (ollama, cerebras, mistral)
                     If None, uses only training_data ground truth
        llm_model: LLM model name (required if llm_backend provided)
        mix_ratio: Percentage of ground truth labels to use (0-100)
                   Remaining percentage comes from LLM
        training_config: Training configuration dict (default: TRAINING_CONFIG)
        lora_config: LoRA configuration dict (default: DEFAULT_LORA_CONFIG)
        device: Device to train on
        logger: Optional logger instance

    Returns:
        adapter_path: Path to saved adapter

    Example:
        # Pure GT training
        adapter = finetune(
            training_data=train_data,
            eval_data=test_data,
            entity_types=['person', 'location'],
            adapter_save_path='./models/adapter_gt'
        )

        # Mixed GT + LLM training
        adapter = finetune(
            training_data=train_data,
            eval_data=test_data,
            entity_types=['person', 'location'],
            adapter_save_path='./models/adapter_mixed',
            llm_backend='ollama',
            llm_model='gemma3:12b',
            mix_ratio=50  # 50% GT + 50% LLM
        )
    """
    if logger is None:
        logger = get_logger("finetune")

    logger.info(f"Fine-tuning GLiNER on {len(training_data)} examples")
    logger.info(f"Mix ratio: {mix_ratio}% GT, {100-mix_ratio}% LLM")

    # Generate LLM labels if requested
    if llm_backend:
        if not llm_model:
            raise ValueError("llm_model required when llm_backend is provided")

        logger.info(f"Generating LLM labels using {llm_backend}/{llm_model}")

        label_generator = create_llm_train_labels(
            backend_type=llm_backend,
            model_name=llm_model,
            entity_types=entity_types,
            cache_type="disk",
            logger=logger
        )

        llm_results = label_generator.generate(
            examples=training_data,
            entity_types=entity_types,
            num_samples=len(training_data),
            verbose=True
        )

        llm_labeled_data = llm_results['all_labels']
        logger.info(f"Generated {len(llm_labeled_data)} LLM labels")

        # Mix GT and LLM labels
        if mix_ratio == 0:
            mixed_data = llm_labeled_data
        elif mix_ratio == 100:
            mixed_data = training_data
        else:
            mixed_data = create_mixed_training_data(
                training_data,
                llm_labeled_data,
                mix_ratio
            )

        logger.info(f"Created mixed dataset: {len(mixed_data)} examples")
    else:
        # Pure GT training
        mixed_data = training_data
        logger.info(f"Using pure GT training: {len(mixed_data)} examples")

    # Initialize model with LoRA
    gloner = GLONER.for_training(
        base_model_path=base_model_path,
        lora_config=lora_config,
        logger=logger
    )

    device_obj = torch.device(device)
    gloner.to(device_obj)

    # Train
    train_config = training_config or TRAINING_CONFIG

    train_lora_model(
        model=gloner.model,
        train_data=mixed_data,
        eval_data=eval_data,
        training_config=train_config,
        adapter_save_path=adapter_save_path,
        logger=logger
    )

    logger.info(f"Adapter saved to: {adapter_save_path}")

    # Cleanup
    del gloner
    torch.cuda.empty_cache()
    gc.collect()

    return adapter_save_path


def predict(
    data: List[Dict[str, Any]],
    entity_types: List[str],
    adapter_path: str,
    base_model_path: str = None,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "cuda",
    logger = None
) -> List:
    """
    Predict with fine-tuned GLiNER adapter

    Args:
        data: Examples to predict on
        entity_types: Entity types to extract
        adapter_path: Path to trained LoRA adapter
        base_model_path: Base model path (default: from lora_defaults)
        threshold: Confidence threshold
        batch_size: Batch size for inference
        device: Device to run on
        logger: Optional logger instance

    Returns:
        predictions: GLiNER format predictions

    Example:
        predictions = predict(
            data=test_data,
            entity_types=['person', 'location'],
            adapter_path='./models/adapter_gt'
        )
    """
    if logger is None:
        logger = get_logger("predict")

    logger.info(f"Loading adapter from: {adapter_path}")
    logger.info(f"Predicting on {len(data)} examples")

    # Load model with adapter
    gloner = GLONER.for_inference(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        logger=logger
    )

    device_obj = torch.device(device)
    gloner.to(device_obj)

    # Predict
    predictions = gloner.predict(
        data=data,
        entity_types=entity_types,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
        flat_ner=True
    )

    logger.info(f"Generated predictions for {len(predictions)} examples")

    # Cleanup
    del gloner
    torch.cuda.empty_cache()
    gc.collect()

    return predictions


def evaluate(
    data: List[Dict[str, Any]],
    entity_types: List[str],
    predictions: List = None,
    model_path: str = None,
    adapter_path: str = None,
    model_type: str = "gloner",
    has_ground_truth: bool = True,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "cuda",
    logger = None
) -> Dict[str, Any]:
    """
    Evaluate predictions or model directly

    Two modes:
    1. Evaluate pre-computed predictions (pass predictions argument)
    2. Load model, predict, then evaluate (pass model_path/adapter_path)

    Args:
        data: Data with optional ground truth
        entity_types: Entity types
        predictions: Pre-computed predictions (if provided, skip prediction step)
        model_path: Base model path (for mode 2)
        adapter_path: Adapter path (for mode 2)
        model_type: gloner or llm
        has_ground_truth: Whether data has ner field for evaluation
        threshold: Confidence threshold (for mode 2)
        batch_size: Batch size (for mode 2)
        device: Device (for mode 2)
        logger: Optional logger instance

    Returns:
        evaluation_results: Dict with metrics

    Example:
        # Mode 1: Evaluate predictions
        results = evaluate(
            data=test_data,
            entity_types=['person', 'location'],
            predictions=predictions,
            model_type='gloner'
        )

        # Mode 2: Load model and evaluate
        results = evaluate(
            data=test_data,
            entity_types=['person', 'location'],
            adapter_path='./models/adapter_gt',
            model_type='gloner'
        )
    """
    if logger is None:
        logger = get_logger("evaluate")

    # Mode 2: Load model and predict
    if predictions is None:
        if model_type == "gloner":
            if not adapter_path and not model_path:
                raise ValueError("Must provide adapter_path or model_path for gloner evaluation")

            logger.info("Loading model and generating predictions")
            predictions = predict(
                data=data,
                entity_types=entity_types,
                adapter_path=adapter_path,
                base_model_path=model_path,
                threshold=threshold,
                batch_size=batch_size,
                device=device,
                logger=logger
            )
        else:
            raise ValueError("LLM predictions must be provided directly (cannot load LLM model)")

    # Evaluate
    logger.info(f"Evaluating {len(predictions)} predictions")
    logger.info(f"Model type: {model_type}, Has GT: {has_ground_truth}")

    if model_type == "gloner":
        results = evaluate_gloner(
            predictions=predictions,
            data=data,
            entity_types=entity_types,
            has_ground_truth=has_ground_truth
        )
    elif model_type == "llm":
        if not has_ground_truth:
            raise ValueError("LLM evaluation requires ground truth")
        results = evaluate_llm(
            predictions=predictions,
            data=data,
            entity_types=entity_types
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'gloner' or 'llm'")

    logger.info("Evaluation complete")
    if has_ground_truth and 'overall_metrics' in results:
        logger.info(f"Overall F1: {results['overall_metrics']['overall_f1_pct']:.2f}%")

    return results
