"""
Backend Factory
Creates appropriate LLM backend based on configuration
"""

from typing import Optional
from .base import LLMBackend
from .ollama import OllamaBackend
from .mistral import MistralBackend
from .cerebras import CerebrasBackend
from .cerebras_structured import StructuredCerebrasBackend


class BackendFactory:
    """Factory for creating LLM backends"""

    @staticmethod
    def create(
        backend_type: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        use_structured_output: bool = False
    ) -> LLMBackend:
        """
        Create LLM backend based on type

        Args:
            backend_type: Type of backend ('ollama', 'mistral', 'cerebras')
            model_name: Model name (optional, uses default if None)
            model_path: Model path (for Mistral only)
            use_structured_output: Use structured output if available (for Cerebras)

        Returns:
            LLMBackend instance

        Raises:
            ValueError: If backend_type is not recognized

        Examples:
            # Ollama backend
            backend = BackendFactory.create('ollama', model_name='gemma3:12b')

            # Mistral backend
            backend = BackendFactory.create('mistral')

            # Cerebras backend (standard)
            backend = BackendFactory.create('cerebras', model_name='qwen-3-235b-a22b-instruct-2507')

            # Cerebras backend (structured output)
            backend = BackendFactory.create(
                'cerebras',
                model_name='qwen-3-235b-a22b-thinking-2507',
                use_structured_output=True
            )
        """
        backend_type = backend_type.lower()

        if backend_type == 'ollama':
            if model_name is None:
                model_name = 'gemma3:12b'
            return OllamaBackend(model_name=model_name)

        elif backend_type == 'mistral':
            return MistralBackend(model_path=model_path)

        elif backend_type == 'cerebras':
            if use_structured_output:
                if model_name is None:
                    model_name = 'qwen-3-235b-a22b-thinking-2507'
                return StructuredCerebrasBackend(model_name=model_name)
            else:
                if model_name is None:
                    model_name = 'qwen-3-235b-a22b-instruct-2507'
                return CerebrasBackend(model_name=model_name)

        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Supported types: 'ollama', 'mistral', 'cerebras'"
            )

    @staticmethod
    def list_backends() -> list:
        """
        List all available backend types

        Returns:
            List of backend type strings
        """
        return ['ollama', 'mistral', 'cerebras']
