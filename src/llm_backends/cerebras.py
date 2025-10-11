"""
Cerebras Backend Implementation
Extracted from api_labeler.py
"""

import os
import time
from typing import Tuple
from dotenv import load_dotenv

import cerebras.cloud.sdk
from cerebras.cloud.sdk import Cerebras

from .base import LLMBackend
from config.llm_config import CEREBRAS_CONFIG

# Load environment variables
load_dotenv()


class CerebrasBackend(LLMBackend):
    """Cerebras API backend (standard prompting)"""

    def __init__(self, model_name: str = "qwen-3-235b-a22b-instruct-2507"):
        """
        Initialize Cerebras backend

        Args:
            model_name: Cerebras model name
        """
        super().__init__(model_name)

        # Get API key
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable not set")

        # Initialize client
        self.client = Cerebras(
            api_key=api_key,
            max_retries=2,
            timeout=90.0
        )

        self.config = CEREBRAS_CONFIG.copy()

        # Rate limiting tracking
        self.requests_per_minute = 0
        self.minute_start_time = time.time()
        self.tokens_per_minute = 0

    def _is_hard_quota_error(self, error: Exception) -> bool:
        """
        Check if error is a hard quota limit (daily/hourly)

        Args:
            error: Exception from Cerebras API

        Returns:
            True if error indicates hard quota exceeded (daily/hourly limit)
        """
        error_str = str(error).lower()
        hard_quota_indicators = [
            "token_quota_exceeded", "tokens per day limit exceeded",
            "tokens per hour limit exceeded", "daily limit exceeded",
            "hourly limit exceeded", "requests per day limit exceeded",
            "requests per hour limit exceeded", "quota exceeded"
        ]
        return any(indicator in error_str for indicator in hard_quota_indicators)

    def _wait_for_rate_limit(self, estimated_tokens: int = 500):
        """
        Intelligent rate limiting based on current usage
        Extracted from api_labeler.py

        Args:
            estimated_tokens: Estimated tokens for next request
        """
        current_time = time.time()

        # Reset counters every minute
        if current_time - self.minute_start_time >= 60:
            self.requests_per_minute = 0
            self.tokens_per_minute = 0
            self.minute_start_time = current_time

        # Check if we need to wait for request limit
        if self.requests_per_minute >= self.config['max_requests_per_minute']:
            wait_time = 60 - (current_time - self.minute_start_time) + 1
            if wait_time > 0:
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.tokens_per_minute = 0
                self.minute_start_time = time.time()

        # Check if we need to wait for token limit
        if self.tokens_per_minute + estimated_tokens >= self.config['max_tokens_per_minute']:
            wait_time = 60 - (current_time - self.minute_start_time) + 1
            if wait_time > 0:
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.tokens_per_minute = 0
                self.minute_start_time = time.time()

        # Add small buffer between requests
        time.sleep(2.1)

    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """
        Generate response using Cerebras API

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            cerebras.cloud.sdk.RateLimitError: If hard quota exceeded (caller should handle gracefully)
        """
        # Estimate tokens (rough approximation: 4 chars per token)
        estimated_tokens = len(prompt) // 4

        # Wait for rate limits
        self._wait_for_rate_limit(estimated_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_completion_tokens=self.config['max_completion_tokens'],
                top_p=self.config['top_p']
            )

            # Update rate limiting counters
            self.requests_per_minute += 1

            # Extract token counts from response
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            self.tokens_per_minute += input_tokens + output_tokens

            response_text = response.choices[0].message.content

            return response_text, input_tokens, output_tokens

        except cerebras.cloud.sdk.RateLimitError as e:
            # Check if it's a hard quota error (daily/hourly limit)
            if self._is_hard_quota_error(e):
                # Re-raise for graceful handling by caller
                raise
            else:
                # Temporary rate limit - wait and retry
                wait_time = min(60, 2 ** 3)
                time.sleep(wait_time)
                raise

        except cerebras.cloud.sdk.APITimeoutError as e:
            # Re-raise to let caller handle retry
            raise

        except cerebras.cloud.sdk.APIConnectionError as e:
            time.sleep(5)
            raise

        except cerebras.cloud.sdk.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10)
            raise

    def supports_structured_output(self) -> bool:
        """Cerebras supports structured output (but not used in this backend)"""
        return False

    def get_context_limit(self) -> int:
        """Cerebras context limit"""
        return self.config['context_limit']

    def get_model_limits(self) -> Tuple[int, int]:
        """Get Cerebras model limits"""
        return (self.config['context_limit'], self.config['max_completion_tokens'])
