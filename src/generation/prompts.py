"""
Prompt generation for synthetic data - extracted from your create_baseline_synthetic_prompt and create_targeted_prompt_with_analysis functions
Preserves exact prompt templates and logic - FIXED function signatures
"""

import json
from typing import List, Dict, Any, Optional


def create_baseline_synthetic_prompt(entity_types: List[str], domain_focus: str, 
                                   language: str, country: str, **kwargs) -> str:
    """
    Create baseline prompt for synthetic data when no corrected examples available - domain-agnostic version
    
    Args:
        entity_types: List of entity types to focus on
        domain_focus: Domain description (from config or analysis)
        language: Generation language
        country: Country for variation
        **kwargs: Additional attributes for variation
        
    Returns:
        Formatted prompt string
    """
    # Filter attributes
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}
    
    # Build base prompt - made domain-agnostic
    prompt = f"""CRITICAL: This is a PRODUCTION system for generating training data.

**Objective:**
Generate realistic text passages in the domain of "{domain_focus}" that include clearly identified named entities. 

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled in the 'entities' list
- Follow the exact format shown in the examples below

**Entity Types to Focus On:**
"""
    
    # DYNAMIC ENTITY TYPES
    for entity_type in entity_types:
        prompt += f"- {entity_type}: Entities of type {entity_type}\n"
    
    # Add baseline examples - made generic
    prompt += f"""
**BASELINE EXAMPLES:**
Here are examples showing the expected format and entity types for {domain_focus}:

Example 1:
{{
  "text": "Sample text containing various named entities that are relevant to the domain.",
  "entities": [
    {{"entity": "example entity 1", "types": ["{entity_types[0] if entity_types else 'entity'}"]}},
    {{"entity": "example entity 2", "types": ["{entity_types[1] if len(entity_types) > 1 else 'entity'}"]}}
  ]
}}

Example 2:
{{
  "text": "Another sample text with different entities for variety and diversity.",
  "entities": [
    {{"entity": "different entity", "types": ["{entity_types[0] if entity_types else 'entity'}"]}},
    {{"entity": "another entity", "types": ["{entity_types[-1] if entity_types else 'entity'}"]}}
  ]
}}

"""
    
    # Add generation instructions
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])
    
    prompt += f"""

**MANDATORY Task:**
Generate a NEW text passage in the domain of "{domain_focus}" similar to the examples above but with different content.
Use the following attributes for variation: language={language}, country={country}, {attributes_string}

**CRITICAL Variation Requirements:**
- MUST include entities from these types: {', '.join(entity_types)}
- Create diverse expressions and formats for each entity type
- Use clear, explicit language for entity identification
- Provide sufficient context for each entity
- Make entities easily distinguishable in the text
- Content should be relevant to: {domain_focus}

**MANDATORY Output Format:**
<start language="{language}" country="{country}" {attributes_string}>
{{
  "text": "your generated text here",
  "entities": [
    {{"entity": "entity name", "types": ["entity type"]}},
    ...
  ]
}}
<end>

CRITICAL: Generate ONLY ONE example in the specified JSON format.

<start language="{language}" country="{country}" {attributes_string}>
"""
    
    return prompt

def create_targeted_prompt_with_analysis(low_confidence_examples: List[Dict], entity_types: List[str], 
                                       domain_focus: str, language: str, country: str,
                                       final_summary: Optional[Dict] = None, **kwargs) -> str:
    """
    Create prompt with analysis integration and dynamic entity types - domain-agnostic version
    
    Args:
        low_confidence_examples: List of low confidence examples as templates
        entity_types: List of entity types
        domain_focus: Domain description (from config or analysis)
        language: Generation language
        country: Country for variation
        final_summary: Optional final summary from analysis
        **kwargs: Additional attributes for variation
        
    Returns:
        Formatted prompt string
    """
    # Filter attributes
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}
    
    # Build base prompt - made domain-agnostic
    prompt = f"""CRITICAL: This is a PRODUCTION system for generating training data.

**Objective:**
Generate realistic text passages in the domain of "{domain_focus}" that include clearly identified named entities. Focus on creating diverse examples based on domain analysis and provided templates.

**MANDATORY Format Requirements:**
- Output MUST be in JSON format with "text" and "entities" fields
- Each entity MUST be accurately labeled in the 'entities' list
- Follow the exact format shown in the examples below

**Entity Types to Focus On:**
"""
    
    # DYNAMIC ENTITY TYPES - using the entity_types variable
    for entity_type in entity_types:
        prompt += f"- {entity_type}: Entities of type {entity_type}\n"
    
    # ADD DOMAIN ANALYSIS INSIGHTS - exact copy from your original
    if final_summary:
        prompt += f"""
**DOMAIN ANALYSIS INSIGHTS:**
Based on analysis of low-confidence examples, here are key insights to incorporate:

Domain Summary: {final_summary.get('domain_summary', 'N/A')}

Entity-Specific Insights:
"""
        for entity_type in entity_types:
            entity_data = final_summary.get('entity_summaries', {}).get(entity_type, {})
            if entity_data:
                prompt += f"""
{entity_type.upper()}:
- Position Issues: {entity_data.get('position_summary', 'None identified')}
- What Works Well: {entity_data.get('good_examples_summary', 'N/A')}
- What Fails: {entity_data.get('bad_examples_summary', 'N/A')}
- Needed Variations: {entity_data.get('variations_summary', 'N/A')}
"""
    
    prompt += f"""
**TEMPLATE EXAMPLES:**
Here are some real examples showing the expected format and entity types for {domain_focus}:

"""
    
    # Add low confidence examples as templates - exact copy from your original
    for i, example in enumerate(low_confidence_examples):
        text = " ".join(example['tokenized_text'])
        entities = []
        
        # Convert NER format to JSON entities
        for start, end, label in example['ner']:
            entity_text = " ".join(example['tokenized_text'][start:end+1])
            entities.append({
                "entity": entity_text,
                "types": [label]
            })
        
        prompt += f"""
Example {i+1}:
{{
  "text": "{text}",
  "entities": {json.dumps(entities, indent=2)}
}}
"""
    
    # Add generation instructions
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])
    
    prompt += f"""

**MANDATORY Task:**
Generate a NEW text passage in the domain of "{domain_focus}" similar to the examples above but with different content.
Use the following attributes for variation: language={language}, country={country}, {attributes_string}
"""
    
    # Add analysis-based instructions if available - exact copy from your original
    if final_summary:
        prompt += f"""
IMPORTANT: Incorporate the domain analysis insights above to:
- Address the position/boundary issues identified
- Use patterns from "what works well" examples
- Avoid patterns from "what fails" examples  
- Include the needed variations identified in the analysis
"""
    
    prompt += f"""

**CRITICAL Variation Requirements:**
- MUST include entities from these types: {', '.join(entity_types)}
- Create diverse expressions and formats for each entity type
- Use clear, explicit language for entity identification
- Provide sufficient context for each entity
- Make entities easily distinguishable in the text
- Content should be relevant to: {domain_focus}

**MANDATORY Output Format:**
<start language="{language}" country="{country}" {attributes_string}>
{{
  "text": "your generated text here",
  "entities": [
    {{"entity": "entity name", "types": ["entity type"]}},
    ...
  ]
}}
<end>

CRITICAL: Generate ONLY ONE example in the specified JSON format.

<start language="{language}" country="{country}" {attributes_string}>
"""
    
    return prompt
