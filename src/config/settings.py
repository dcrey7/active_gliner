"""
Settings management for Active Learning Pipeline
Centralized paths and basic configuration
"""

import os
from pathlib import Path


class Settings:
    """Simple settings class to organize configuration"""

    def __init__(self):
        # File paths - resolve relative to project root, not src/
        project_root = Path(__file__).parent.parent.parent  # Go up from src/config/ to root
        self.data_path = project_root / "data" / "mit-movie"
        self.train_file = "train.json"
        self.test_file = "test.json"
        self.labels_file = "labels.json"
        self.logs_dir = project_root / "logs"
        self.models_dir = project_root / "models"

        # System settings
        self.global_seed = 42
        self.cuda_device = "0"

        # Model defaults (for reference, actual config in lora_defaults.py)
        self.model_name = "knowledgator/modern-gliner-bi-large-v1.0"
        self.model_max_length = 8192
        self.batch_size = 8
        self.prediction_threshold = 0.5
        
    def setup_environment(self):
        """Set environment variables"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ['PYTHONHASHSEED'] = str(self.global_seed)

    def create_directories(self):
        """Create necessary directories"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        """Complete setup - environment and directories"""
        self.setup_environment()
        self.create_directories()
