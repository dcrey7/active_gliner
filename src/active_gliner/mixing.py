"""Assign simulated gold access within a fixed selected set."""

from active_gliner.selection import select


def gold_ids(
    selected_ids: list[str], confidences: dict, fraction: float, assignment: str, seed: int
) -> set[str]:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    if assignment not in {"routed", "random"}:
        raise ValueError("assignment must be routed or random")
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
