# LLM Backends Module

Abstraction layer for different LLM providers (Ollama, Mistral, Cerebras).

## Usage

### Basic Usage

```python
from llm_backends import BackendFactory

# Create Ollama backend
backend = BackendFactory.create('ollama', model_name='gemma3:12b')

# Create Mistral backend
backend = BackendFactory.create('mistral')

# Create Cerebras backend (standard)
backend = BackendFactory.create('cerebras', model_name='qwen-3-235b-a22b-instruct-2507')

# Create Cerebras backend (structured output)
backend = BackendFactory.create(
    'cerebras',
    model_name='qwen-3-235b-a22b-thinking-2507',
    use_structured_output=True
)

# Generate response
prompt = "What is NER?"
response_text, input_tokens, output_tokens = backend.generate(prompt)

# Check capabilities
if backend.supports_structured_output():
    print("Backend supports structured JSON output")

context_limit = backend.get_context_limit()
print(f"Context limit: {context_limit} tokens")
```

## Available Backends

### OllamaBackend
- **Type**: `'ollama'`
- **Models**: Any Ollama model (e.g., `gemma3:12b`, `llama3`, `mistral`)
- **Structured Output**: No
- **Token Tracking**: No (returns 0 for both)

### MistralBackend
- **Type**: `'mistral'`
- **Models**: Local Mistral Inference models
- **Structured Output**: No
- **Token Tracking**: Yes (exact counts)
- **Default Path**: `~/mistral_models/7B-Instruct-v0.3`

### CerebrasBackend
- **Type**: `'cerebras'`
- **Models**: Cerebras API models (e.g., `qwen-3-235b-a22b-instruct-2507`)
- **Structured Output**: No (standard prompting)
- **Token Tracking**: Yes (from API)
- **Rate Limiting**: Automatic (30 req/min, 60k tokens/min)

### StructuredCerebrasBackend
- **Type**: `'cerebras'` with `use_structured_output=True`
- **Models**: Cerebras thinking models (e.g., `qwen-3-235b-a22b-thinking-2507`)
- **Structured Output**: Yes (JSON schema validation)
- **Token Tracking**: Yes (from API)
- **Rate Limiting**: Automatic (30 req/min, 60k tokens/min)
- **Graceful Quota Handling**: Yes (raises `RateLimitError` for hard quota limits)

## Configuration

All configuration is centralized in `config/llm_config.py`:

- `OLLAMA_CONFIG`: Ollama generation parameters
- `MISTRAL_CONFIG`: Mistral generation parameters
- `CEREBRAS_CONFIG`: Cerebras standard prompting parameters
- `CEREBRAS_STRUCTURED_CONFIG`: Cerebras structured output parameters
- `NER_LABEL_SCHEMA`: JSON schema for NER structured output

## Error Handling

All backends re-raise exceptions for retry handling by caller:

- `cerebras.cloud.sdk.RateLimitError`: Rate limit hit (transient or hard quota)
- `cerebras.cloud.sdk.APITimeoutError`: API timeout
- `cerebras.cloud.sdk.APIConnectionError`: Connection error
- `cerebras.cloud.sdk.APIStatusError`: API error (4xx/5xx)

Backends handle automatic waiting for rate limits internally.

## Architecture

```
llm_backends/
├── __init__.py           # Exports BackendFactory
├── base.py               # Abstract LLMBackend interface
├── factory.py            # BackendFactory implementation
├── ollama.py             # OllamaBackend
├── mistral.py            # MistralBackend
├── cerebras.py           # CerebrasBackend (standard)
└── cerebras_structured.py # StructuredCerebrasBackend
```

## Adding New Backends

1. Create new file (e.g., `openai.py`)
2. Implement `LLMBackend` interface
3. Register in `factory.py`
4. Add config to `config/llm_config.py`
