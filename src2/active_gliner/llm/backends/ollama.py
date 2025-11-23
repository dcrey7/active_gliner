import os
import time
import ollama
from typing import Tuple, Dict, Optional

from .base import BackendBase
from ...config.llm_config import OLLAMA_DEFAULT


class OllamaBackend(BackendBase):

    def _get_default_config(self) -> Dict:
        return OLLAMA_DEFAULT.copy()

    def _setup(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    def generate(self, prompt: str, schema: Optional[Dict] = None) -> Tuple[str, Dict]:
        max_retries = self.config['max_retries']
        backoff_base = self.config['retry_backoff_base']

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                options = {
                    'temperature': self.config['temperature'],
                    'num_predict': self.config['num_predict'],
                    'top_k': self.config['top_k'],
                    'top_p': self.config['top_p']
                }

                if schema:
                    response = ollama.generate(
                        model=self.config['model_name'],
                        prompt=prompt,
                        format=schema,
                        options=options
                    )
                else:
                    response = ollama.generate(
                        model=self.config['model_name'],
                        prompt=prompt,
                        options=options
                    )

                content = response.get('response', '').strip()

                input_tokens = response.get('prompt_eval_count', 0)
                output_tokens = response.get('eval_count', 0)
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

                return content, stats

            except ollama.ResponseError as e:
                print(f"Ollama error (attempt {attempt+1}): {e.error}")
                self.stats.add_failure('ollama_error', str(e))

                if attempt == max_retries:
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
                time.sleep(backoff_base ** attempt)

            except Exception as e:
                print(f"Unexpected error (attempt {attempt+1}): {e}")
                self.stats.add_failure('other', str(e))

                if attempt == max_retries:
                    return "", {'error': str(e), 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
                time.sleep(backoff_base ** attempt)

        return "", {'error': 'Max retries exceeded', 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'latency_ms': 0}
