from typing import List, Optional, Tuple, Dict
from .schemas import NERResponse


def StandardPrompt(text: str, entities: List[str], examples: Optional[List[Dict]] = None) -> str:
    """Standard prompt - exact content from src/prompting/standard_prompt.py"""

    prompt = f"""CRITICAL: You are an expert at Name Entity Recognition information extractor. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled with the specified entity types
- Use ONLY the provided entity types

**Entity Types to Use (ONLY these types):**
"""

    # Add entity types
    for entity_type in entities:
        prompt += f"- {entity_type}: Entities of type {entity_type}\n"

    # Add few-shot examples if provided
    if examples:
        prompt += "\n**Examples:**\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Text: {ex['text']}\n"
            prompt += f"Output: {ex['entities']}\n"
        prompt += "\n"

    prompt += f"""
**Text to Label:**
{text}

**CRITICAL Requirements:**
- MUST use entities from these types ONLY: {', '.join(entities)}
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


def StructuredPrompt(text: str, entities: List[str], examples: Optional[List[Dict]] = None) -> Tuple[str, Dict]:
    """Structured prompt - exact content from src/prompting/structured_prompt.py + schema"""

    prompt = f"""CRITICAL: You are an expert at Named Entity Recognition. Label the given text with named entities.

**Objective:**
Identify and extract named entities from the provided text using the specified entity types.

**Entity Types to Use (ONLY these types):**
"""

    # Add entity types
    for entity_type in entities:
        prompt += f"- {entity_type}: Entities of type {entity_type}\n"

    # Add few-shot examples if provided
    if examples:
        prompt += "\n**Examples:**\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Text: {ex['text']}\n"
            prompt += f"Output: {ex['entities']}\n"
        prompt += "\n"

    prompt += f"""
**Text to Label:**
{text}

**CRITICAL Requirements:**
- MUST use entities from these types ONLY: {', '.join(entities)}
- Identify ALL relevant entities in the text
- Use clear, exact entity names as they appear in text
- Do not modify or paraphrase entity names
- Include entities even if you're not 100% certain
"""

    # Generate schema from Pydantic model
    schema = NERResponse.get_json_schema()

    return prompt, schema
