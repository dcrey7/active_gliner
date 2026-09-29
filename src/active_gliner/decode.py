"""Greedy flat decoding across labels."""


def greedy_flat(spans: list[dict]) -> list[dict]:
    """Keep non-overlapping spans by confidence, with deterministic ties."""
    ordered = sorted(spans, key=lambda s: (-s["confidence"], s["start"], s["end"], s["label"]))
    accepted = []
    for span in ordered:
        if not any(
            span["start"] < other["end"] and other["start"] < span["end"] for other in accepted
        ):
            accepted.append(span)
    return sorted(accepted, key=lambda s: (s["start"], s["end"], s["label"]))
