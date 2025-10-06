"""
Ollama Backend Implementation
Extracted from gemma_labeler.py
"""

import ollama
from typing import Tuple
from .base import LLMBackend
from config.llm_config import OLLAMA_CONFIG


class OllamaBackend(LLMBackend):
    """Ollama LLM backend"""

    def __init__(self, model_name: str = "gemma3:12b"):
        """
        Initialize Ollama backend

        Args:
            model_name: Ollama model name (default: gemma3:12b)
        """
        super().__init__(model_name)
        self.config = OLLAMA_CONFIG.copy()

    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """
        Generate response using Ollama

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
            Note: Ollama doesn't provide token counts, returns 0 for both
        """
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.config
        )

        response_text = response['response'].strip()

        # Ollama doesn't provide token counts
        input_tokens = 0
        output_tokens = 0

        return response_text, input_tokens, output_tokens

    def supports_structured_output(self) -> bool:
        """Ollama doesn't support structured output"""
        return False

    def get_context_limit(self) -> int:
        """Ollama context limit (varies by model, using common default)"""
        return 128000

    def get_model_limits(self) -> Tuple[int, int]:
        """Get Ollama model limits"""
        return (128000, 500)
