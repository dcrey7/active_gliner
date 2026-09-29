"""Sentence confidence rules from design section 2."""

from collections.abc import Iterable


def min_confidence(confs: Iterable[float]) -> float:
    """NER and relations: minimum predicted confidence; no prediction gives zero."""
    return min(confs, default=0.0)


def classification_confidence(probs: dict[str, float], threshold: float) -> float:
    """Classification: minimum distance to the frozen decision threshold."""
    return min((abs(p - threshold) for p in probs.values()), default=0.0)


def slot_confidence(pred_confs: Iterable[float], max_below: float | None) -> float:
    """Slots: minimum prediction, else one minus best candidate, else confident absence."""
    fallback = 1.0 if max_below is None else 1.0 - max_below
    return min(pred_confs, default=fallback)
