"""
LLM Backend Configuration
Centralized configuration for all LLM backends
"""

# Ollama Configuration
OLLAMA_CONFIG = {
    'top_k': 50,
    'top_p': 0.8,
    'num_predict': 500,
    'temperature': 0.3,
}

# Mistral Configuration
MISTRAL_CONFIG = {
    'max_tokens': 500,
    'temperature': 0.3,
    'context_limit': 32768,
}

# Cerebras Configuration
CEREBRAS_CONFIG = {
    'temperature': 0.3,
    'max_completion_tokens': 500,
    'top_p': 0.8,
    'context_limit': 65536,
    'max_requests_per_minute': 30,
    'max_tokens_per_minute': 60000,
}

# Cerebras Structured Output Configuration
CEREBRAS_STRUCTURED_CONFIG = {
    'temperature': 0.3,
    'max_completion_tokens': 60000,
    'top_p': 0.8,
    'context_limit': 65536,
    'max_requests_per_minute': 30,
    'max_tokens_per_minute': 60000,
}

# NER Label Schema for Structured Outputs
NER_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "types": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["entity", "types"],
                "additionalProperties": False
            }
        }
    },
    "required": ["text", "entities"],
    "additionalProperties": False
}
