"""
Base LLM Backend Interface
Defines the contract that all LLM backends must follow
"""

from abc import ABC, abstractmethod
from typing import Tuple


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    def __init__(self, model_name: str):
        """
        Initialize LLM backend

        Args:
            model_name: Name or path of the model
        """
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """
        Generate response from LLM

        Args:
            prompt: Input prompt string

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        pass

    @abstractmethod
    def supports_structured_output(self) -> bool:
        """
        Check if backend supports structured JSON output

        Returns:
            True if structured output is supported, False otherwise
        """
        pass

    @abstractmethod
    def get_context_limit(self) -> int:
        """
        Get the context window size for this backend

        Returns:
            Maximum context length in tokens
        """
        pass

    @abstractmethod
    def get_model_limits(self) -> Tuple[int, int]:
        """
        Get model input/output limits

        Returns:
            Tuple of (context_limit, max_output_tokens)
        """
        pass
