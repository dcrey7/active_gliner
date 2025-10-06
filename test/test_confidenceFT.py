import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import gc
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
from generation import create_label_generator
from training.trainer import train_lora_model
from models.gloner import GLONER
from caching import disk_cache, memory_cache



def main():
    """Confidence Analysis: Fine-tuning Performance on Worst Examples"""
    
    # ===============================================================================
    # Setup and Configuration
    # ===============================================================================
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    # Load FULL test data for evaluation
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"📊 Loaded FULL test data: {len(test_data)} examples, {len(entity_types)} entity types")
    
    # Load pre-saved low confidence examples for training
    logger.info("📂 Loading pre-saved low confidence examples...")
    with open('../results/high_mse_2500_examples.json', 'r') as file:
        low_n = json.load(file)
    logger.info(f"📊 Loaded {len(low_n)} low confidence examples for training")
    
    generator = create_label_generator('ollama', model_name='gemma3:4b')
    print(f"Initialized label generator with Ollama Gemma3:4b : {generator}")

    training_config = {
        'num_steps': 1000,
        'train_batch_size': 8,
        'gradient_accumulation_steps': 1,
        'learning_rate': 0.00021008343694753508,
        'others_lr': 0.00021008343694753508,
        'warmup_ratio': 0.07064690788186724,
        'eval_steps': 100,
        'save_steps': 100,
        'logging_steps': 10,
        'max_grad_norm': 1,
        'weight_decay': 0.020216630535603918,
        'others_weight_decay': 0.020216630535603918,
        'focal_loss_alpha': 0.75,
        'focal_loss_gamma': 1.0,
        'patience': 3
    }
    
    logger.info("⚙️ Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"   • {key}: {value}")
    
    # ===============================================================================
    # Experiment Parameters
    # ===============================================================================
    subset_sizes = [10,25,50]
    
    # Results storage
    results = {
        'no_worst_examples': [],
        'gliner_ft_llm_f1': [],
        'gliner_ft_gt_f1': [],
        'confidence': [],
        'avg_entities': [],
        'avg_input_tokens': [],
        'model_input_output': [],
        'avg_output_tokens': []
    }

    total_iterations = len(subset_sizes) * 2  # 2 experiments per subset (LLM + GT)
    logger.info(f"\n🔬 Experiment Overview:")
    logger.info(f"   • Subset sizes to test: {subset_sizes}")
    logger.info(f"   • Total training experiments: {total_iterations}")
    logger.info(f"   • Evaluation dataset: FULL test set ({len(test_data)} examples)")
    logger.info(f"   • Label cache initialized: {len(label_cache)} examples")
    
    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================
    label_cache=[]
    logger.info(f"\n🚀 Starting Fine-tuning Analysis...")
    logger.info("-" * 60)
    
    for n_examples in tqdm(subset_sizes, desc="Training Experiments", position=0):
        logger.info(f"\n📝 Training on {n_examples} worst confidence examples")
        print(f"n_examples : {n_examples}")
        # Get subset for training
        train_subset = low_n[:n_examples]
        logger.info(f"Training subset size: {len(train_subset)} examples")
        
        # ===============================================================================
        # Generate LLM Labels WITH CACHING
        # ===============================================================================
        
        logger.info(f"🤖 Generating LLM labels for {n_examples} examples (with caching)...")

        llm_labeled_data = create_label_generator(
            low_n_examples=train_subset,  # Use the subset as input 
            num_samples=n_examples,       # How many we want
            entity_types=entity_types,
            label_cache=label_cache,      # This persists and accumulates
            verbose=True
        )
        
        # Calculate metrics from generated data
        if len(llm_labeled_data) > 0:
            avg_entities = sum(len(ex['ner']) for ex in llm_labeled_data) / len(llm_labeled_data)
            
            # Token metrics - these would come from the labeler if it tracked them
            # For now using estimates based on Gemma performance
            token_metrics = {
                'avg_input_tokens': 450.0,  # Approximate for labeling prompts
                'model_input_output': (128000, 500),  # Gemma context limits
                'avg_output_tokens': 120.0  # Approximate for label generation
            }
        else:
            avg_entities = 0.0
            token_metrics = {
                'avg_input_tokens': 0.0,
                'model_input_output': (128000, 500),
                'avg_output_tokens': 0.0
            }
        
        logger.info(f"📊 Generated/Retrieved: {len(llm_labeled_data)} examples, avg entities: {avg_entities:.1f}")
        logger.info(f"💾 Label cache now contains: {len(label_cache)} total examples")
        

if __name__ == "__main__":
    main()