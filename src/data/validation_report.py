"""
Validation Report Data Class
Tracks detailed information about validation results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class ValidationReport:
    """Report of validation results with detailed removal tracking"""

    total_examples: int = 0
    valid_examples: int = 0
    removed_examples: int = 0

    # Removal reasons with example details
    out_of_bounds: List[Dict[str, any]] = field(default_factory=list)
    invalid_order: List[Dict[str, any]] = field(default_factory=list)
    invalid_types: List[Dict[str, any]] = field(default_factory=list)
    invalid_format: List[Dict[str, any]] = field(default_factory=list)
    empty_entities_removed: int = 0

    # Invalid types found
    invalid_types_found: Set[str] = field(default_factory=set)

    def add_out_of_bounds(self, example_idx: int, entity: tuple, text_length: int, text: str = ""):
        """Track out of bounds entity"""
        self.out_of_bounds.append({
            'example_idx': example_idx,
            'entity': entity,
            'text_length': text_length,
            'text': text[:50] + "..." if len(text) > 50 else text
        })

    def add_invalid_order(self, example_idx: int, entity: tuple, text: str = ""):
        """Track invalid order entity"""
        self.invalid_order.append({
            'example_idx': example_idx,
            'entity': entity,
            'text': text[:50] + "..." if len(text) > 50 else text
        })

    def add_invalid_type(self, example_idx: int, entity_type: str, entity: tuple, text: str = ""):
        """Track invalid type entity"""
        self.invalid_types.append({
            'example_idx': example_idx,
            'entity_type': entity_type,
            'entity': entity,
            'text': text[:50] + "..." if len(text) > 50 else text
        })
        self.invalid_types_found.add(entity_type)

    def add_invalid_format(self, example_idx: int, reason: str, text: str = ""):
        """Track invalid format"""
        self.invalid_format.append({
            'example_idx': example_idx,
            'reason': reason,
            'text': text[:50] + "..." if len(text) > 50 else text
        })

    def summary(self) -> str:
        """
        Generate human-readable summary

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("="*60)
        lines.append("VALIDATION REPORT")
        lines.append("="*60)
        lines.append(f"Total examples processed: {self.total_examples}")
        lines.append(f"Valid examples: {self.valid_examples}")
        lines.append(f"Removed examples: {self.removed_examples}")

        if self.removed_examples > 0:
            lines.append("")
            lines.append("Removal Details:")

            if self.out_of_bounds:
                lines.append(f"  • Out of bounds indices: {len(self.out_of_bounds)} entities")
                for item in self.out_of_bounds[:3]:  # Show first 3
                    lines.append(f"    - Example {item['example_idx']}: Entity {item['entity']} but text length is {item['text_length']}")
                if len(self.out_of_bounds) > 3:
                    lines.append(f"    ... and {len(self.out_of_bounds) - 3} more")

            if self.invalid_order:
                lines.append(f"  • Invalid index order: {len(self.invalid_order)} entities")
                for item in self.invalid_order[:3]:
                    lines.append(f"    - Example {item['example_idx']}: {item['entity']} (start > end)")
                if len(self.invalid_order) > 3:
                    lines.append(f"    ... and {len(self.invalid_order) - 3} more")

            if self.invalid_types:
                lines.append(f"  • Invalid entity types: {len(self.invalid_types)} entities")
                lines.append(f"    Invalid types found: {sorted(self.invalid_types_found)}")
                for item in self.invalid_types[:3]:
                    lines.append(f"    - Example {item['example_idx']}: Found '{item['entity_type']}' (not in allowed types)")
                if len(self.invalid_types) > 3:
                    lines.append(f"    ... and {len(self.invalid_types) - 3} more")

            if self.invalid_format:
                lines.append(f"  • Invalid format: {len(self.invalid_format)} examples")
                for item in self.invalid_format[:3]:
                    lines.append(f"    - Example {item['example_idx']}: {item['reason']}")
                if len(self.invalid_format) > 3:
                    lines.append(f"    ... and {len(self.invalid_format) - 3} more")

            if self.empty_entities_removed > 0:
                lines.append(f"  • Empty entities removed: {self.empty_entities_removed} examples")

        lines.append("="*60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for saving"""
        return {
            'total_examples': self.total_examples,
            'valid_examples': self.valid_examples,
            'removed_examples': self.removed_examples,
            'out_of_bounds_count': len(self.out_of_bounds),
            'invalid_order_count': len(self.invalid_order),
            'invalid_types_count': len(self.invalid_types),
            'invalid_format_count': len(self.invalid_format),
            'empty_entities_removed': self.empty_entities_removed,
            'invalid_types_found': sorted(self.invalid_types_found),
            'details': {
                'out_of_bounds': self.out_of_bounds,
                'invalid_order': self.invalid_order,
                'invalid_types': self.invalid_types,
                'invalid_format': self.invalid_format
            }
        }
