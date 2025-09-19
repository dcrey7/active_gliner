#!/usr/bin/env python3
"""
Simple LLM Evaluation Script
Choose your model and dataset - get results
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

# Suppress warnings
import warnings
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
from evaluation.llm_evaluator import LLMEvaluationPipeline, LLMModelWrapper


def main():
    """Simple evaluation with model and dataset selection"""
    
    # ===============================================================================
    # Configuration (Edit these as needed)
    # ===============================================================================
    
    # MODEL SELECTION
    # MODEL_TYPE = "ollama"           # "ollama" or "mistral"
    # MODEL_NAME = "gemma3:12b"       # For ollama: "gemma3:12b", "llama2", etc.
                                    # For mistral: path or model name
    MODEL_TYPE = "mistral"
    MODEL_NAME = "mistral-7b-instruct"
    # DATASET SELECTION  
    DATASET_SIZE = 100            # None for full dataset, or specify number (e.g., 100)
    
    # ===============================================================================
    # Setup
    # ===============================================================================
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    # Load dataset
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    
    # Apply dataset size limit if specified
    if DATASET_SIZE:
        test_data = test_data[:DATASET_SIZE]
    
    logger.info(f"Dataset: {len(test_data)} examples")
    logger.info(f"Model: {MODEL_TYPE.upper()} - {MODEL_NAME}")
    logger.info(f"Entity types: {entity_types}")
    
    # ===============================================================================
    # Generate Predictions
    # ===============================================================================
    
    # Initialize evaluation pipeline
    evaluation_pipeline = LLMEvaluationPipeline(
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME
    )
    
    # Generate predictions
    gliner_predictions = evaluation_pipeline.evaluate_dataset(test_data, entity_types)
    
    # ===============================================================================
    # Evaluate
    # ===============================================================================
    
    # Run evaluation
    model_wrapper = LLMModelWrapper(gliner_predictions)
    results = enhanced_evaluate(
        model=model_wrapper,
        data=test_data,
        entity_types=entity_types,
        threshold=0.5,
        batch_size=8,
        has_ground_truth=True,
        logger=logger
    )
    
    # ===============================================================================
    # Results (Clean, Simple)
    # ===============================================================================
    
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    # Configure pandas for full display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Overall metrics
    overall_metrics = results['overall_metrics']
    logger.info(f"F1 Score: {overall_metrics.get('overall_f1_pct', 0):.2f}%")
    logger.info(f"Total Examples: {overall_metrics.get('total_examples', 0)}")
    logger.info(f"Correct Examples: {overall_metrics.get('correct_examples', 0)}")
    
    # Classification report
    if 'classification_report_df' in results:
        logger.info("\nCLASSIFICATION REPORT:")
        logger.info("\n" + results['classification_report_df'].to_string(index=False))
    
    # Confidence analysis
    if 'confidence_bins' in results:
        confidence_df = results['confidence_bins'].data if hasattr(results['confidence_bins'], 'data') else results['confidence_bins']
        logger.info("\nCONFIDENCE DISTRIBUTION:")
        logger.info("\n" + confidence_df.to_string())
    
    # Entity performance ranking
    if 'classification_report_df' in results:
        df = results['classification_report_df']
        entity_rows = df[~df['entity_type'].isin(['micro_avg', 'macro_avg'])].copy()
        if len(entity_rows) > 0:
            entity_rows_sorted = entity_rows.sort_values('f1', ascending=False)
            logger.info("\nENTITY PERFORMANCE RANKING:")
            for _, row in entity_rows_sorted.iterrows():
                entity_type = row['entity_type']
                f1_score = row['f1']
                support = row['support']
                logger.info(f"   {entity_type:15s}: F1={f1_score:.3f} ({f1_score*100:5.1f}%) | Support={support:4.0f}")
    
    # Reset pandas options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    logger.info("="*60)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()