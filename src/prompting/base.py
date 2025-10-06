"""
Base Prompt Builder Interface
"""

from abc import ABC, abstractmethod
from typing import List


class PromptBuilder(ABC):
    """Abstract base class for prompt builders"""

    @abstractmethod
    def build(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """
        Build NER labeling prompt

        Args:
            tokenized_text: Text tokens to label
            entity_types: Entity types to identify

        Returns:
            Formatted prompt string
        """
        pass
