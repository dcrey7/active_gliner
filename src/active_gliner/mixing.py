"""Assign simulated gold access within a fixed selected set."""

import random

from active_gliner.selection import select


def nested_ids(selected_ids: list[str], count: int, seed: int) -> set[str]:
    """The first `count` ids of one seeded shuffle, so a larger set extends a smaller one."""
    order = sorted(selected_ids)
    random.Random(seed).shuffle(order)
    return set(order[:count])


def gold_ids(
    selected_ids: list[str], confidences: dict, fraction: float, assignment: str, seed: int
) -> set[str]:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    if assignment not in {"routed", "random", "nested"}:
        raise ValueError("assignment must be routed, random or nested")
    if assignment == "nested":
        return nested_ids(selected_ids, round(fraction * len(selected_ids)), seed)
    scores = (
        [confidences[i] for i in selected_ids]
        if assignment == "routed"
        else [0.0] * len(selected_ids)
    )
    return set(
        select(
            selected_ids,
            scores,
            round(fraction * len(selected_ids)),
            "min" if assignment == "routed" else "random",
            seed,
        )
    )
