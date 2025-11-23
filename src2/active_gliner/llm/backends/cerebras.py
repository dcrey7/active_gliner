import os
import time
from typing import Tuple, Dict, Optional, List
import cerebras.cloud.sdk
from cerebras.cloud.sdk import Cerebras

from .base import BackendBase
from ...config.llm_config import CEREBRAS_DEFAULT
from ..exceptions import LLMConfigError, LLMRuntimeError, HardQuotaError


class CerebrasBackend(BackendBase):
    """
    Cerebras Cloud backend with rate limiting and quota tracking

    Rate limiting:
    - Per-minute limits: Sleep and retry
    - Hourly/daily limits: Raise HardQuotaError
    """

    def _get_default_config(self) -> Dict:
        return CEREBRAS_DEFAULT.copy()

    def _setup(self):
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise LLMConfigError("CEREBRAS_API_KEY not found in environment")

        self.client = Cerebras(api_key=api_key, timeout=self.config['timeout'])

        # Track usage across time windows
        self._usage_tracker = {
            'minute': {'requests': [], 'tokens': []},
            'hour': {'requests': [], 'tokens': []},
            'day': {'requests': [], 'tokens': []}
        }

    def _cleanup_old_entries(self):
        """Remove entries outside time windows"""
        now = time.time()

        # Keep only last minute
        self._usage_tracker['minute']['requests'] = [
            t for t in self._usage_tracker['minute']['requests'] if now - t < 60
        ]
        self._usage_tracker['minute']['tokens'] = [
            (tokens, t) for tokens, t in self._usage_tracker['minute']['tokens'] if now - t < 60
        ]

        # Keep only last hour
        self._usage_tracker['hour']['requests'] = [
            t for t in self._usage_tracker['hour']['requests'] if now - t < 3600
        ]
        self._usage_tracker['hour']['tokens'] = [
            (tokens, t) for tokens, t in self._usage_tracker['hour']['tokens'] if now - t < 3600
        ]

        # Keep only last day
        self._usage_tracker['day']['requests'] = [
            t for t in self._usage_tracker['day']['requests'] if now - t < 86400
        ]
        self._usage_tracker['day']['tokens'] = [
            (tokens, t) for tokens, t in self._usage_tracker['day']['tokens'] if now - t < 86400
        ]

    def _check_and_wait_for_minute_limits(self):
        """Check per-minute limits and sleep if needed"""
        now = time.time()
        self._cleanup_old_entries()

        # Check request limit
        minute_requests = len(self._usage_tracker['minute']['requests'])
        if minute_requests >= self.config['max_requests_per_minute']:
            oldest_request = min(self._usage_tracker['minute']['requests'])
            wait_time = 60 - (now - oldest_request) + 1
            if wait_time > 0:
                print(f"Request limit reached ({minute_requests}/{self.config['max_requests_per_minute']}). Sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
                self._cleanup_old_entries()

        # Check token limit
        minute_tokens = sum(tokens for tokens, _ in self._usage_tracker['minute']['tokens'])
        if minute_tokens >= self.config['max_tokens_per_minute']:
            oldest_token_time = min(t for _, t in self._usage_tracker['minute']['tokens'])
            wait_time = 60 - (now - oldest_token_time) + 1
            if wait_time > 0:
                print(f"Token limit reached ({minute_tokens}/{self.config['max_tokens_per_minute']}). Sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
                self._cleanup_old_entries()

    def _track_usage(self, input_tokens: int, output_tokens: int):
        """Track token usage across all time windows"""
        now = time.time()
        total_tokens = input_tokens + output_tokens

        self._usage_tracker['minute']['requests'].append(now)
        self._usage_tracker['minute']['tokens'].append((total_tokens, now))

        self._usage_tracker['hour']['requests'].append(now)
        self._usage_tracker['hour']['tokens'].append((total_tokens, now))

        self._usage_tracker['day']['requests'].append(now)
        self._usage_tracker['day']['tokens'].append((total_tokens, now))

    def _check_quota_warnings(self):
        """Warn if approaching hourly/daily limits (<10% remaining)"""
        self._cleanup_old_entries()

        # Check hourly quota
        hourly_tokens = sum(tokens for tokens, _ in self._usage_tracker['hour']['tokens'])
        hourly_limit = self.config['max_tokens_per_hour']
        if hourly_tokens > hourly_limit * 0.9:
            remaining = hourly_limit - hourly_tokens
            remaining_pct = (remaining / hourly_limit) * 100
            print(f"WARNING: {remaining} tokens remaining in hourly quota ({remaining_pct:.1f}%)")

        # Check daily quota
        daily_tokens = sum(tokens for tokens, _ in self._usage_tracker['day']['tokens'])
        daily_limit = self.config['max_tokens_per_day']
        if daily_tokens > daily_limit * 0.9:
            remaining = daily_limit - daily_tokens
            remaining_pct = (remaining / daily_limit) * 100
            print(f"WARNING: {remaining} tokens remaining in daily quota ({remaining_pct:.1f}%)")

    def generate(self, prompt: str, schema: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Generate with Cerebras Cloud API

        Args:
            prompt: Input prompt
            schema: Optional JSON schema for structured output

        Returns:
            (content, stats_dict)

        Raises:
            HardQuotaError: If daily/hourly quota exceeded
        """
        # Check per-minute limits and sleep if needed
        self._check_and_wait_for_minute_limits()

        max_retries = self.config['max_retries']
        backoff_base = self.config['retry_backoff_base']

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                kwargs = {
                    "model": self.config['model_name'],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config['temperature'],
                    "max_completion_tokens": self.config['max_completion_tokens'],
                    "top_p": self.config['top_p']
                }

                if schema:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ner_extraction",
                            "strict": True,
                            "schema": schema
                        }
                    }

                response = self.client.chat.completions.create(**kwargs)

                # Track actual token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self._track_usage(input_tokens, output_tokens)

                # Check quota warnings
                self._check_quota_warnings()

                # Calculate stats
                latency_ms = (time.time() - start_time) * 1000
                cost_usd = self._calculate_cost(input_tokens, output_tokens)

                stats = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost_usd': cost_usd,
                    'latency_ms': latency_ms,
                    'attempts': attempt + 1
                }

                self.stats.add_success(input_tokens, output_tokens, cost_usd, latency_ms, attempt + 1)

                return response.choices[0].message.content, stats

            except cerebras.cloud.sdk.RateLimitError as e:
                error_str = str(e).lower()

                # Check if hard quota (hourly/daily)
                if any(x in error_str for x in ['daily', 'day', 'hourly', 'hour', 'quota']):
                    self.stats.add_failure('hard_quota', str(e))
                    raise HardQuotaError(f"Hourly/daily quota exceeded: {e}")

                # Per-minute rate limit
                if 'minute' in error_str or 'per minute' in error_str:
                    print(f"Per-minute rate limit hit. Sleeping 60s")
                    time.sleep(60)
                    self._cleanup_old_entries()
                    continue

                # Other rate limit errors
                self.stats.add_failure('rate_limit', str(e))
                if attempt == max_retries:
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
                time.sleep(backoff_base ** attempt)

            except cerebras.cloud.sdk.APITimeoutError as e:
                self.stats.add_failure('timeout', str(e))
                if attempt == max_retries:
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
                time.sleep(backoff_base ** attempt)

            except cerebras.cloud.sdk.APIConnectionError as e:
                self.stats.add_failure('connection', str(e))
                if attempt == max_retries:
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
                time.sleep(backoff_base ** attempt)

            except Exception as e:
                error_str = str(e).lower()

                # Handle temporary errors
                if 'too many requests' in error_str or 'high traffic' in error_str:
                    print(f"Temporary error: {error_str}. Returning empty response")
                    self.stats.add_failure('temporary', str(e))
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}

                self.stats.add_failure('other', str(e))
                if attempt == max_retries:
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
                time.sleep(backoff_base ** attempt)

        return "", {'error': 'Max retries exceeded', 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
