"""
Settings management for Active Learning Pipeline
Updated with synthetic generation configuration
"""

import os
from pathlib import Path
from .constants import *


class Settings:
    """Simple settings class to organize configuration"""
    
    def __init__(self):
        # Core settings
        self.global_seed = GLOBAL_SEED
        self.batch_size = BATCH_SIZE
        
        # Model settings
        self.model_name = DEFAULT_MODEL
        self.model_max_length = MODEL_MAX_LENGTH
        self.prediction_threshold = 0.5
        
        # Training settings
        self.training_steps = TRAINING_STEPS
        self.learning_rate = 5e-4
        self.warmup_ratio = 0.1
        self.eval_steps = EVAL_STEPS
        self.save_steps = SAVE_STEPS
        self.logging_steps = LOGGING_STEPS
        
        # LoRA settings
        self.lora_r = 32
        self.lora_alpha = 64
        self.lora_dropout = 0.1
        
        # Ollama settings
        self.ollama_model = OLLAMA_MODEL
        self.ollama_max_retries = OLLAMA_MAX_RETRIES
        self.ollama_batch_size = OLLAMA_BATCH_SIZE
        self.ollama_temperature = 0.2
        self.ollama_top_p = 0.9
        
        # File paths - resolve relative to project root, not src/
        project_root = Path(__file__).parent.parent.parent  # Go up from src/config/ to root
        self.data_path = project_root / "data" / "mit-movie"
        self.train_file = MIT_TRAIN_FILE
        self.test_file = MIT_TEST_FILE
        self.labels_file = MIT_LABELS_FILE
        self.logs_dir = project_root / "logs"
        self.models_dir = project_root / "models"
        self.cache_dir = project_root / "cache"
        
        # Environment settings
        self.cuda_device = CUDA_DEVICE
        
    def setup_environment(self):
        """Apply environment variables like in your original code"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        os.environ["TORCH_COMPILE"] = TORCH_COMPILE
        os.environ["TORCHINDUCTOR_DISABLE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
        os.environ['PYTHONHASHSEED'] = str(self.global_seed)
    
    def create_directories(self):
        """Create necessary directories"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        """Complete setup - environment and directories"""
        self.setup_environment()
        self.create_directories()
    
    def get_lora_target_modules(self):
        """Return LoRA target modules from your original code"""
        return [
            "dense", "projection", "Wqkv", "Wo", "Wi",
            "query", "key", "value",
            "intermediate.dense", "output.dense",
            "span_rep_layer.span_rep_layer.project_start.3",
            "span_rep_layer.span_rep_layer.project_start.0",
            "span_rep_layer.span_rep_layer.project_end.3",
            "span_rep_layer.span_rep_layer.project_end.0",
            "span_rep_layer.span_rep_layer.out_project.3",
            "span_rep_layer.span_rep_layer.out_project.0",
            'prompt_rep_layer.3','prompt_rep_layer.0',
        ]
    
    def get_generation_config(self):
        """Get synthetic data generation configuration"""
        return {
            'countries': GENERATION_COUNTRIES,
            'language': GENERATION_LANGUAGE,
            'default_domain_focus': DEFAULT_DOMAIN_FOCUS,
            'baseline_examples_count': BASELINE_EXAMPLES_COUNT,
            'max_template_examples': MAX_TEMPLATE_EXAMPLES
        }
    
    def get_experiment_configs(self):
        """Get experiment configuration from your original code"""
        return {
            'corrected_examples_options': [0, 5, 10],
            'synthetic_examples_options': [0, 5, 10]
        }
    
    def get_domain_focus(self, final_summary=None):
        """Get domain focus from analysis or fallback to default"""
        if final_summary and 'domain_summary' in final_summary:
            # Extract domain focus from analysis
            domain_summary = final_summary['domain_summary']
            return domain_summary[:100] + "..." if len(domain_summary) > 100 else domain_summary
        return DEFAULT_DOMAIN_FOCUS
    
    def __repr__(self):
        return f"Settings(seed={self.global_seed}, batch_size={self.batch_size}, model={self.model_name})"


# Global settings instance (like your original pattern)
settings = Settings()
