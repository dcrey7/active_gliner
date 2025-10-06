"""
Structured Prompt Builder
For backends with structured output support (Cerebras)
Simpler prompts since JSON schema is enforced by API
"""

from typing import List
from .base import PromptBuilder


class StructuredPromptBuilder(PromptBuilder):
    """Structured NER labeling prompt for LLMs with schema validation"""

    def build(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """
        Build structured NER labeling prompt

        Args:
            tokenized_text: Text tokens to label
            entity_types: Entity types to identify

        Returns:
            Formatted prompt string (simpler since schema is enforced)
        """
        text = " ".join(tokenized_text)

        prompt = f"""CRITICAL: You are an expert at Named Entity Recognition. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**Entity Types to Use (ONLY these types):**
"""

        # Add entity types dynamically
        for entity_type in entity_types:
            prompt += f"- {entity_type}: Entities of type {entity_type}\n"

        prompt += f"""
**Text to Label:**
{text}

**CRITICAL Requirements:**
- MUST use entities from these types ONLY: {', '.join(entity_types)}
- Identify ALL relevant entities in the text
- Use clear, exact entity names as they appear in text
- Do not modify or paraphrase entity names
- Include entities even if you're not 100% certain
"""

        return prompt
