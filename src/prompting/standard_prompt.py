"""
Standard Prompt Builder
Extracted from gemma_labeler.py, mistral_labeler.py, api_labeler.py
For backends that don't support structured output
"""

from typing import List
from .base import PromptBuilder


class StandardPromptBuilder(PromptBuilder):
    """Standard NER labeling prompt for normal LLMs"""

    def build(self, tokenized_text: List[str], entity_types: List[str]) -> str:
        """
        Build standard NER labeling prompt

        Args:
            tokenized_text: Text tokens to label
            entity_types: Entity types to identify

        Returns:
            Formatted prompt string
        """
        text = " ".join(tokenized_text)

        prompt = f"""CRITICAL: You are an expert at Name Entity Recognition information extractor. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled with the specified entity types
- Use ONLY the provided entity types

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

**MANDATORY Output Format:**
{{
  "text": "{text}",
  "entities": [
    {{"entity": "exact entity name", "types": ["entity type"]}},
    ...
  ]
}}

CRITICAL: Generate ONLY the JSON format above. Start immediately with the JSON object.
"""

        return prompt
