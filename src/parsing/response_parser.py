"""
Response Parser
Extracts JSON from LLM responses (handles markdown wrapping, extra text, etc.)
Extracted from existing labeler parsing logic
"""

import json
from typing import Dict, Any


class ResponseParser:
    """Parser for extracting JSON from LLM responses"""

    @staticmethod
    def extract_json(response_text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response

        Handles:
        - Markdown-wrapped JSON (```json ... ```)
        - JSON buried in extra text
        - Malformed braces

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If JSON cannot be extracted or parsed
        """
        # Clean up response
        response_text = response_text.strip()

        # Remove markdown formatting if present
        if '```json' in response_text:
            start_idx = response_text.find('```json') + 7
            end_idx = response_text.find('```', start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()

        # Extract JSON by finding matching braces
        if '{' in response_text and '}' in response_text:
            # Find the first opening brace
            start_idx = response_text.find('{')
            response_text = response_text[start_idx:]

            # Try to find the matching closing brace
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

        # Parse JSON
        return json.loads(response_text)
