#!/usr/bin/env python3
"""
Direct LLM Evaluation Script
Evaluate Mistral/Gemma predictions directly on test set without fine-tuning
"""

import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)

# Import modules
from config.settings import Settings
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.loader import load_mit_dataset
from evaluation.evaluator import enhanced_evaluate
from evaluation.llm_evaluator import LLMEvaluator, convert_ner_to_gliner_format, FakeModelWrapper

def main():
    """Main evaluation function"""
    
    # ===============================================================================
    # 1. Setup and Configuration
    # ===============================================================================
    
    # Setup
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    # Load test data
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"Loaded test data: {len(test_data)} examples, {len(entity_types)} entity types")
    
    # ===============================================================================
    # 2. LLM Evaluation Setup
    # ===============================================================================
    
    # Choose LLM model - CHANGE THIS TO SWAP MODELS
    model_type = "ollama"  # "ollama" or "mistral"
    model_name = "gemma3:12b"  # for ollama
    # model_type = "mistral"  # uncomment for mistral
    # model_name = None  # for mistral (uses default path)
    
    logger.info(f"Evaluating {model_type.upper()} model: {model_name or 'default'}")
    
    # Initialize LLM evaluator
    llm_evaluator = LLMEvaluator(
        model_type=model_type,
        model_name=model_name
    )
    
    # ===============================================================================
    # 3. Generate LLM Predictions
    # ===============================================================================
    
    logger.info("Starting LLM prediction generation...")
    
    # Get predictions for all test examples
    cleaned_predictions = llm_evaluator.predict_all(test_data, entity_types)
    
    logger.info(f"Generated predictions for {len(cleaned_predictions)} examples")
    
    # ===============================================================================
    # 4. Convert to GLiNER Format
    # ===============================================================================
    
    logger.info("Converting predictions to GLiNER format...")
    
    # Convert to GLiNER prediction format
    gliner_format_predictions = convert_ner_to_gliner_format(cleaned_predictions)
    
    logger.info(f"Converted {len(gliner_format_predictions)} predictions to GLiNER format")
    
    # ===============================================================================
    # 5. Evaluation Using Enhanced Evaluate
    # ===============================================================================
    
    logger.info("Running enhanced evaluation...")
    
    # Create fake model wrapper
    fake_model = FakeModelWrapper(gliner_format_predictions)
    
    # Run enhanced evaluation
    evaluation_results = enhanced_evaluate(
        model=fake_model,
        data=test_data,
        entity_types=entity_types,
        threshold=0.5,
        batch_size=16,
        has_ground_truth=True,
        logger=logger
    )
    
    # ===============================================================================
    # 6. Results Analysis
    # ===============================================================================
    
    logger.info("="*60)
    logger.info(f"{model_type.upper()} DIRECT EVALUATION RESULTS")
    logger.info("="*60)
    
    # Extract key results
    overall_metrics = evaluation_results['overall_metrics']
    f1_score = overall_metrics['overall_f1_pct']
    confidence = overall_metrics['overall_confidence_pct']
    
    logger.info(f"F1 Score: {f1_score:.1f}%")
    logger.info(f"Confidence: {confidence:.1f}%")
    logger.info(f"Total Examples: {overall_metrics['total_examples']}")
    logger.info(f"Correct Examples: {overall_metrics['correct_examples']}")
    logger.info(f"Incorrect Examples: {len(overall_metrics['incorrect_examples'])}")
    
    # Comparison with baselines
    baseline_f1 = 46.95
    synthetic_f1 = 62.0  # From previous labeling experiment
    
    logger.info("="*60)
    logger.info("COMPARISON WITH BASELINES")
    logger.info("="*60)
    logger.info(f"GLiNER Baseline F1: {baseline_f1:.1f}%")
    logger.info(f"{model_type.upper()} Direct F1: {f1_score:.1f}%")
    logger.info(f"GLiNER + Synthetic Labels F1: {synthetic_f1:.1f}%")
    
    improvement_over_baseline = f1_score - baseline_f1
    logger.info(f"{model_type.upper()} vs Baseline: {improvement_over_baseline:+.1f}% F1")
    
    if f1_score < synthetic_f1:
        synthetic_boost = synthetic_f1 - f1_score  
        logger.info(f"Synthetic Labeling Boost: +{synthetic_boost:.1f}% F1")
    else:
        logger.info(f"{model_type.upper()} direct prediction outperforms synthetic approach!")
    
    # ===============================================================================
    # 7. Detailed Analysis
    # ===============================================================================
    
    logger.info("="*60)
    logger.info("DETAILED ANALYSIS")
    logger.info("="*60)
    
    # Classification report analysis
    if 'classification_report_df' in evaluation_results:
        class_report = evaluation_results['classification_report_df']
        
        # Show per-entity performance
        for _, row in class_report.iterrows():
            if row['entity_type'] not in ['micro_avg', 'macro_avg']:
                logger.info(f"{row['entity_type']}: F1={row['f1']:.3f}, Support={row['support']}")
        
        # Show micro/macro averages
        micro_row = class_report[class_report['entity_type'] == 'micro_avg'].iloc[0]
        macro_row = class_report[class_report['entity_type'] == 'macro_avg'].iloc[0]
        
        logger.info(f"Micro Avg: P={micro_row['precision']:.3f}, R={micro_row['recall']:.3f}, F1={micro_row['f1']:.3f}")
        logger.info(f"Macro Avg: P={macro_row['precision']:.3f}, R={macro_row['recall']:.3f}, F1={macro_row['f1']:.3f}")
    
    # Entity distribution analysis
    total_entities = sum(len(ex['ner']) for ex in cleaned_predictions)
    avg_entities = total_entities / len(cleaned_predictions) if cleaned_predictions else 0
    logger.info(f"Average entities per example: {avg_entities:.1f}")
    
    # Count entity types
    entity_counts = {}
    for ex in cleaned_predictions:
        for _, _, label in ex['ner']:
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    logger.info("Entity type distribution:")
    for entity_type, count in sorted(entity_counts.items()):
        logger.info(f"  {entity_type}: {count}")
    
    logger.info("="*60)
    logger.info(f"{model_type.upper()} EVALUATION COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()