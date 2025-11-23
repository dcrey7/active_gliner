"""
LLM Backend Configuration

Centralized configuration for all LLM backends.
User should update cost_per_million_* values based on current pricing.
"""

# Cerebras Cloud Configuration
CEREBRAS_DEFAULT = {
    # Model
    'model_name': 'qwen-3-235b-a22b-instruct-2507',

    # Generation parameters
    'temperature': 0.3,
    'max_completion_tokens': 50000,
    'top_p': 0.8,
    'timeout': 90.0,

    # Retry & rate limiting
    'max_retries': 3,
    'retry_backoff_base': 2,
    'max_requests_per_minute': 30,
    'max_tokens_per_minute': 60000,
    'max_requests_per_hour': 90,
    'max_tokens_per_hour': 1000000,
    'max_tokens_per_day': 1000000,

    # Cost tracking (UPDATE THESE based on current pricing!)
    # Example: $0.10 per 1K tokens = $100 per 1M tokens
    'cost_per_million_input_tokens': 0.6,
    'cost_per_million_output_tokens': 0.4,

    # Limits
    'context_limit': 128000,
}

# Ollama Local Configuration
OLLAMA_DEFAULT = {
    # Model
    'model_name': 'gemma3:4b',

    # Generation parameters
    'temperature': 0.3,
    'num_predict': 50000,
    'top_k': 50,
    'top_p': 0.8,

    # Retry
    'max_retries': 3,
    'retry_backoff_base': 2,

    # Cost (local = free)
    'cost_per_million_input_tokens': 0.0,
    'cost_per_million_output_tokens': 0.0,

    # Limits
    'context_limit': 128000,
}

# Mistral Local Configuration
MISTRAL_DEFAULT = {
    # Model
    'model_name': '7B-Instruct-v0.3',

    # Generation parameters
    'temperature': 0.3,
    'max_tokens': 50000,

    # Retry
    'max_retries': 3,
    'retry_backoff_base': 2,

    # Cost (local = free)
    'cost_per_million_input_tokens': 0.0,
    'cost_per_million_output_tokens': 0.0,

    # Limits
    'context_limit': 32768,
}
