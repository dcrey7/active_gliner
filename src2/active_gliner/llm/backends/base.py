from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import time


class BackendBase(ABC):
    """Backend blueprint - each backend implements its own retry logic"""

    def __init__(self, model_name: str = None, **overrides):
        # Load defaults from config
        self.config = self._get_default_config()

        # Apply overrides
        if model_name:
            self.config['model_name'] = model_name
        self.config.update(overrides)

        # Initialize stats
        from ..stats import BackendStats
        self.stats = BackendStats()

        # Setup
        try:
            self._setup()
        except Exception as e:
            self.stats.setup_errors += 1
            from ..exceptions import LLMConfigError
            raise LLMConfigError(f"Backend setup failed: {e}")

    @abstractmethod
    def _get_default_config(self) -> Dict:
        """Load config from config module"""
        pass

    @abstractmethod
    def _setup(self):
        """Initialize client, check API key (NO network calls!)"""
        pass

    @abstractmethod
    def generate(self, prompt: str, schema: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Generate with backend-specific retry

        Returns: (content, stats_dict)
        Raises: HardQuotaError only
        """
        pass

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost from config values"""
        cost_input = (input_tokens / 1_000_000) * self.config['cost_per_million_input_tokens']
        cost_output = (output_tokens / 1_000_000) * self.config['cost_per_million_output_tokens']
        return cost_input + cost_output

    def info(self) -> Dict:
        return {
            'backend': self.__class__.__name__,
            'model': self.config['model_name'],
            'config': self.config,
            'stats': self.stats.summary()
        }
