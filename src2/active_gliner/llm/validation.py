import json
from typing import List, Dict, Tuple, Optional

from .schemas import NERResponse
from .stats import ValidationStats


def extract_json(response_text: str) -> Dict:
    """Extract JSON from LLM response (handles markdown, extra text)"""
    response_text = response_text.strip()

    # Remove markdown formatting if present
    if '```json' in response_text:
        start_idx = response_text.find('```json') + 7
        end_idx = response_text.find('```', start_idx)
        if end_idx != -1:
            response_text = response_text[start_idx:end_idx].strip()

    # Extract JSON by finding matching braces
    if '{' in response_text and '}' in response_text:
        start_idx = response_text.find('{')
        response_text = response_text[start_idx:]

        # Find matching closing brace
        brace_count = 0
        end_idx = -1
        for idx, char in enumerate(response_text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = idx + 1
                    break

        if end_idx != -1:
            response_text = response_text[:end_idx]
        else:
            # Fallback: use rfind to get last closing brace
            end_idx = response_text.rfind('}') + 1
            response_text = response_text[:end_idx]

    return json.loads(response_text)


def validate_ner_response(
    response_text: str,
    expected_entities: List[str],
    original_text: str,
    stats: Optional[ValidationStats] = None
) -> Tuple[bool, Optional[Dict], List[str]]:
    """
    Validate and clean LLM NER response

    Input: response_text="{'text': '...', 'entities': [...]}", expected_entities=['actor', 'genre'], original_text="Brad Pitt..."
    Output: (True, {'text': '...', 'entities': [{'entity': 'Brad Pitt', 'types': ['actor']}]}, [])
    """
    errors = []
    expected_entities_set = set(expected_entities)

    # Check empty response
    if not response_text or not response_text.strip():
        errors.append("Empty response from LLM")
        if stats:
            stats.add_invalid('empty')
        return False, None, errors

    # Parse JSON
    try:
        data = extract_json(response_text)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        if stats:
            stats.add_invalid('parse', {'error': str(e)})
        return False, None, errors

    # Step 2: Validate structure with Pydantic
    try:
        ner_response = NERResponse(**data)
        data = ner_response.dict()
    except Exception as e:
        errors.append(f"Structure validation error: {e}")
        if stats:
            stats.add_invalid('structure', {'error': str(e)})
        return False, None, errors

    # Step 3: Validate NER-specific rules
    cleaned_entities = []
    invalid_count = 0

    for entity in data['entities']:
        entity_text = entity['entity']
        entity_types = entity['types']

        # Check entity text not empty
        if not entity_text or not entity_text.strip():
            errors.append(f"Empty entity text")
            invalid_count += 1
            if stats:
                stats.add_invalid('empty')
            continue

        # Check entity exists in original text
        if entity_text not in original_text:
            errors.append(f"Entity '{entity_text}' not found in original text")
            invalid_count += 1
            if stats:
                stats.add_invalid('bounds', {'entity': entity_text})
            continue

        # Check entity types not empty
        if not entity_types:
            errors.append(f"Entity '{entity_text}' has no types")
            invalid_count += 1
            if stats:
                stats.add_invalid('empty')
            continue

        # Check all types are valid
        invalid_types = [t for t in entity_types if t not in expected_entities_set]
        if invalid_types:
            errors.append(f"Entity '{entity_text}' has invalid types: {invalid_types}")
            invalid_count += 1
            if stats:
                for inv_type in invalid_types:
                    stats.add_invalid('type', {'invalid_type': inv_type, 'entity': entity_text})
            continue

        # Check entity not too long (> 100 chars likely error)
        if len(entity_text) > 500000:
            errors.append(f"Entity too long ({len(entity_text)} chars): '{entity_text[:50]}...'")
            invalid_count += 1
            if stats:
                stats.add_invalid('long_span', {'entity': entity_text, 'length': len(entity_text)})
            continue

        # Entity is valid
        cleaned_entities.append(entity)

    # Update data with cleaned entities
    data['entities'] = cleaned_entities

    # Check if any entities remain after cleaning
    if len(cleaned_entities) == 0:
        errors.append("No valid entities after cleaning")
        if stats:
            stats.add_invalid('empty')
        return False, None, errors

    # Valid response with cleaned entities
    if stats:
        stats.add_valid(len(cleaned_entities))
        if invalid_count > 0:
            stats.total_entities_removed += invalid_count

    return True, data, errors
