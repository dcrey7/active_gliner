"""
NER Data Validator with Detailed Reporting
Validates NER data and provides detailed reports of what was removed and why
"""

from typing import List, Dict, Tuple
from .validation_report import ValidationReport
from utils.logging import get_logger


class NERValidator:
    """
    NER data validator with detailed reporting

    Features:
    - Validates entity indices, types, format
    - Tracks exactly what was removed and why
    - Generates human-readable reports
    """

    def __init__(self, entity_types: List[str], logger=None):
        """
        Initialize validator

        Args:
            entity_types: Valid entity types
            logger: Logger instance (optional)
        """
        self.entity_types = set(entity_types)
        self.logger = logger or get_logger("NERValidator")

    def validate(
        self,
        ner_data: List[Dict],
        strict: bool = True
    ) -> Tuple[List[Dict], ValidationReport]:
        """
        Validate NER data with detailed reporting

        Args:
            ner_data: List of NER examples to validate
            strict: If True, remove invalid examples; if False, keep with empty entities

        Returns:
            Tuple of (cleaned_data, validation_report)
        """
        report = ValidationReport()
        report.total_examples = len(ner_data)

        cleaned_data = []

        for i, example in enumerate(ner_data):
            try:
                tokenized_text = example.get('tokenized_text', [])
                ner = example.get('ner', [])
                text_len = len(tokenized_text)

                # Check format
                if not isinstance(tokenized_text, list):
                    report.add_invalid_format(i, "tokenized_text is not a list")
                    if strict:
                        continue
                    else:
                        cleaned_data.append({'tokenized_text': [], 'ner': []})
                        continue

                if not isinstance(ner, list):
                    report.add_invalid_format(i, "ner is not a list")
                    if strict:
                        continue
                    else:
                        cleaned_data.append({'tokenized_text': tokenized_text, 'ner': []})
                        continue

                cleaned_entities = []
                text = " ".join(tokenized_text)

                # Validate each entity
                for entity in ner:
                    # Check entity format
                    if not isinstance(entity, (list, tuple)) or len(entity) != 3:
                        report.add_invalid_format(i, f"Entity format invalid: {entity}", text)
                        continue

                    start, end, entity_type = entity

                    # Check index types
                    if not isinstance(start, int) or not isinstance(end, int):
                        report.add_invalid_format(i, f"Non-integer indices: {entity}", text)
                        continue

                    # Check index order
                    if start > end:
                        report.add_invalid_order(i, entity, text)
                        continue

                    # Check index bounds
                    if start < 0 or end >= text_len:
                        report.add_out_of_bounds(i, entity, text_len, text)
                        continue

                    # Check for extremely long spans (likely errors)
                    if (end - start) > 15:
                        report.add_invalid_format(i, f"Span too long ({end - start} tokens): {entity}", text)
                        continue

                    # Check entity type validity
                    if entity_type not in self.entity_types:
                        report.add_invalid_type(i, entity_type, entity, text)
                        continue

                    # If we get here, entity is valid
                    cleaned_entities.append([start, end, entity_type])

                # Add example to cleaned data
                if strict:
                    # In strict mode, only keep examples with valid entities
                    if len(cleaned_entities) > 0:
                        cleaned_data.append({
                            "tokenized_text": tokenized_text,
                            "ner": cleaned_entities
                        })
                        report.valid_examples += 1
                    else:
                        report.empty_entities_removed += 1
                else:
                    # In non-strict mode, keep all examples (preserve indices)
                    cleaned_data.append({
                        "tokenized_text": tokenized_text,
                        "ner": cleaned_entities
                    })
                    if len(cleaned_entities) > 0:
                        report.valid_examples += 1
                    else:
                        report.empty_entities_removed += 1

            except Exception as e:
                self.logger.warning(f"Error validating example {i}: {e}")
                report.add_invalid_format(i, f"Exception: {str(e)}")
                if not strict:
                    # Preserve index even on error
                    cleaned_data.append({
                        "tokenized_text": example.get('tokenized_text', []),
                        "ner": []
                    })

        # Calculate removed count
        if strict:
            report.removed_examples = report.total_examples - len(cleaned_data)
        else:
            report.removed_examples = 0  # We preserve all indices

        return cleaned_data, report

    def validate_and_log(
        self,
        ner_data: List[Dict],
        strict: bool = True,
        log_report: bool = True
    ) -> List[Dict]:
        """
        Validate and automatically log the report

        Args:
            ner_data: List of NER examples to validate
            strict: If True, remove invalid examples; if False, keep with empty entities
            log_report: Whether to log the validation report

        Returns:
            Cleaned data
        """
        cleaned_data, report = self.validate(ner_data, strict=strict)

        if log_report:
            self.logger.info(report.summary())

        return cleaned_data
