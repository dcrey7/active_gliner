"""
Constants extracted from your original active learning pipeline
Updated with synthetic generation configuration
"""

# Core pipeline constants (from your original code)
GLOBAL_SEED = 42
BATCH_SIZE = 8

# Model settings
DEFAULT_MODEL = "knowledgator/modern-gliner-bi-large-v1.0"
MODEL_MAX_LENGTH = 8192

# Ollama LLM settings
OLLAMA_MODEL = "mistral:latest"
OLLAMA_MAX_RETRIES = 3
OLLAMA_BATCH_SIZE = 10

# File paths (from your original code)
MIT_DATA_PATH = "../../data/mit-movie"
MIT_TRAIN_FILE = "train.json"
MIT_TEST_FILE = "test.json"
MIT_LABELS_FILE = "labels.json"

# Output directories  
LOGS_DIR = "../../logs"
MODELS_DIR = "../../models"
CACHE_DIR = "../../cache"

# Training constants (from your pipeline)
TRAINING_STEPS = 200
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10

# Cache prefixes (extracted from your global cache variables)
BATCH_ANALYSIS_CACHE_PREFIX = "batch_analysis_"
FINAL_SUMMARY_CACHE_PREFIX = "final_summary_"
SYNTHETIC_CACHE_PREFIX = "synthetic_data_"

# Synthetic data generation settings (NEW - extracted from hardcoded values)
GENERATION_COUNTRIES = [
    "usa", "uk", "australia", "canada", "ireland", 
    "new zealand", "south africa", "india"
]

GENERATION_LANGUAGE = "english"
DEFAULT_DOMAIN_FOCUS = "text passages"  # Fallback if no domain analysis available

# Prompt template settings
BASELINE_EXAMPLES_COUNT = 2
MAX_TEMPLATE_EXAMPLES = 10

# Environment variables (from your original setup)
CUDA_DEVICE = "0"
TORCH_COMPILE = "0"
TOKENIZERS_PARALLELISM = "true"
