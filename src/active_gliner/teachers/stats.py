from dataclasses import asdict, dataclass, field


@dataclass
class TeacherStats:
    """Count validity per record; token costs cover only this job's API calls."""

    elapsed_s: float = 0.0
    workers: int = 1
    gpu_shared: bool = False
    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    duplicate: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s_total: float = 0.0
    errors_by_kind: dict[str, int] = field(default_factory=dict)
    invalid_records: int = 0
    valid_rate: float = 1.0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
