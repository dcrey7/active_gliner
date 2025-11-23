from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BackendStats:
    """Track backend-level statistics"""

    # Requests
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Tokens & cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Timing
    total_latency_ms: float = 0.0

    # Errors
    setup_errors: int = 0
    retry_attempts: int = 0
    connection_errors: int = 0
    timeout_errors: int = 0
    rate_limit_errors: int = 0
    hard_quota_errors: int = 0
    other_errors: int = 0
    error_messages: List[str] = field(default_factory=list)

    def add_success(self, input_tokens: int, output_tokens: int, cost_usd: float, latency_ms: float, attempts: int):
        self.total_requests += 1
        self.successful_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.total_latency_ms += latency_ms
        if attempts > 1:
            self.retry_attempts += (attempts - 1)

    def add_failure(self, error_type: str, error_msg: str):
        self.total_requests += 1
        self.failed_requests += 1

        if error_type == 'connection':
            self.connection_errors += 1
        elif error_type == 'timeout':
            self.timeout_errors += 1
        elif error_type == 'rate_limit':
            self.rate_limit_errors += 1
        elif error_type == 'hard_quota':
            self.hard_quota_errors += 1
        else:
            self.other_errors += 1

        if len(self.error_messages) < 20:
            self.error_messages.append(f"{error_type}: {error_msg}")

    def summary(self) -> Dict:
        return {
            'total_requests': self.total_requests,
            'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost_usd': round(self.total_cost_usd, 4),
            'avg_latency_ms': round(self.total_latency_ms / self.successful_requests, 2) if self.successful_requests > 0 else 0,
            'retry_rate': self.retry_attempts / self.total_requests if self.total_requests > 0 else 0,
            'errors': {
                'connection': self.connection_errors,
                'timeout': self.timeout_errors,
                'rate_limit': self.rate_limit_errors,
                'hard_quota': self.hard_quota_errors,
                'other': self.other_errors
            }
        }


@dataclass
class ValidationStats:
    """Track validation-level statistics"""

    total_validated: int = 0
    valid_responses: int = 0
    invalid_responses: int = 0

    # Error types
    parse_errors: int = 0
    structure_errors: int = 0
    type_errors: int = 0
    bounds_errors: int = 0
    empty_entities: int = 0
    long_span_errors: int = 0

    # Entity tracking
    total_entities_extracted: int = 0
    total_entities_valid: int = 0
    total_entities_removed: int = 0
    invalid_types_found: set = field(default_factory=set)
    error_examples: List[Dict] = field(default_factory=list)

    def add_valid(self, entity_count: int):
        self.total_validated += 1
        self.valid_responses += 1
        self.total_entities_extracted += entity_count
        self.total_entities_valid += entity_count

    def add_invalid(self, error_type: str, details: Dict = None):
        self.total_validated += 1
        self.invalid_responses += 1

        if error_type == 'parse':
            self.parse_errors += 1
        elif error_type == 'structure':
            self.structure_errors += 1
        elif error_type == 'type':
            self.type_errors += 1
            if details and 'invalid_type' in details:
                self.invalid_types_found.add(details['invalid_type'])
        elif error_type == 'bounds':
            self.bounds_errors += 1
        elif error_type == 'empty':
            self.empty_entities += 1
        elif error_type == 'long_span':
            self.long_span_errors += 1

        if details and len(self.error_examples) < 10:
            self.error_examples.append({'error_type': error_type, 'details': details})

    def summary(self) -> Dict:
        return {
            'total_validated': self.total_validated,
            'validation_rate': round(self.valid_responses / self.total_validated, 4) if self.total_validated > 0 else 0,
            'total_entities_extracted': self.total_entities_extracted,
            'entity_validity_rate': round(self.total_entities_valid / self.total_entities_extracted, 4) if self.total_entities_extracted > 0 else 0,
            'errors': {
                'parse': self.parse_errors,
                'structure': self.structure_errors,
                'type': self.type_errors,
                'bounds': self.bounds_errors,
                'empty': self.empty_entities,
                'long_span': self.long_span_errors
            },
            'invalid_types_found': sorted(list(self.invalid_types_found))
        }
