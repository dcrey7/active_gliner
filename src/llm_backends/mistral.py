"""
Mistral Backend Implementation
Extracted from mistral_labeler.py
"""

from typing import Tuple
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from .base import LLMBackend
from config.llm_config import MISTRAL_CONFIG


class MistralBackend(LLMBackend):
    """Mistral Inference backend"""

    def __init__(self, model_path: str = None):
        """
        Initialize Mistral backend

        Args:
            model_path: Path to Mistral model folder (default: ~/mistral_models/7B-Instruct-v0.3)
        """
        if model_path is None:
            self.model_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
        else:
            self.model_path = Path(model_path)

        model_name = "MISTRAL 7B v0.3 (mistral inf)"
        super().__init__(model_name)

        # Load tokenizer and model
        tokenizer_path = self.model_path / "tokenizer.model.v3"
        self.tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
        self.model = Transformer.from_folder(self.model_path)

        self.config = MISTRAL_CONFIG.copy()

    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """
        Generate response using Mistral inference

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Create chat completion request
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )

        # Encode the prompt and count input tokens
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        input_token_count = len(tokens)

        # Generate response
        out_tokens, _ = generate(
            [tokens],
            self.model,
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature'],
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )

        # Count output tokens
        output_token_count = len(out_tokens[0])

        # Decode the response
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return result, input_token_count, output_token_count

    def supports_structured_output(self) -> bool:
        """Mistral Inference doesn't support structured output"""
        return False

    def get_context_limit(self) -> int:
        """Mistral 7B context limit"""
        return self.config['context_limit']

    def get_model_limits(self) -> Tuple[int, int]:
        """Get Mistral model limits"""
        return (self.config['context_limit'], self.config['max_tokens'])
