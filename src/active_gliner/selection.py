"""Thesis confidence scores and seeded pool selection."""

import math
import random
from collections.abc import Iterable


def score_min(confs: Iterable[float]) -> float:
    return min(confs, default=0.0)


def score_avg(confs: Iterable[float]) -> float:
    values = list(confs)
    return sum(values) / len(values) if values else 0.0


def score_mse(confs: Iterable[float]) -> float:
    values = list(confs)
    return sum((1.0 - c) ** 2 for c in values) / len(values) if values else 1.0


def score_mnlp(confs: Iterable[float]) -> float:
    values = list(confs)
    if not values:
        return math.inf
    return -sum(math.log(max(c, 1e-10)) for c in values) / len(values)


# Thesis selectors (src2 selection/strategy.py): lower key = less sure, picked first.
THESIS_KEYS = {
    "avg": score_avg,
    "mse": lambda confs: -score_mse(confs),
    "mnlp": lambda confs: -score_mnlp(confs),
}


def item_confidences(pred: dict) -> list[float]:
    """Confidences of the predicted items: spans, relations or classification labels."""
    if "spans" in pred:
        return [s["confidence"] for s in pred["spans"]]
    if "relations" in pred:
        return [r["probability"] for r in pred["relations"]]
    return [pred["probabilities"][label] for label in pred["labels"]]


def thesis_keys(strategy: str, items: list[list[float]]) -> list[float]:
    return [THESIS_KEYS[strategy](confs) for confs in items]


def select(ids: list[str], confidences: list[float], n: int, strategy: str, seed: int) -> list[str]:
    """Select by ascending confidence with seeded ties, or sample randomly."""
    if len(ids) != len(confidences):
        raise ValueError("ids and confidences must have equal lengths")
    if not 0 <= n <= len(ids):
        raise ValueError("n must be between zero and the number of ids")
    rng = random.Random(seed)
    if strategy == "random":
        return rng.sample(ids, n)
    # Thesis selectors arrive as keys where lower means less sure, like min.
    if strategy != "min" and strategy not in THESIS_KEYS:
        raise ValueError(f"Unknown strategy: {strategy}")
    ranked = list(zip(ids, confidences, strict=True))
    rng.shuffle(ranked)
    ranked.sort(key=lambda item: item[1])
    return [item_id for item_id, _ in ranked[:n]]


def rank_empty_last(confidences: list[float], has_prediction: list[bool]) -> list[float]:
    return [
        c if present else 1.0 + 1e-12
        for c, present in zip(confidences, has_prediction, strict=True)
    ]


def select_diverse(ids: list[str], embeddings, n: int, seed: int) -> list[str]:
    import numpy as np
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) != len(ids):
        raise ValueError("embeddings must have one row per id")
    if len(set(ids)) != len(ids) or not 0 <= n <= len(ids):
        raise ValueError("Need distinct ids and n between zero and the pool size")
    if n == 0:
        return []
    # Multi-threaded k-means sums in a varying order, so the same embeddings gave
    # different clusters in different processes. One thread makes it repeatable.
    with threadpool_limits(limits=1):
        clusters = KMeans(n_clusters=n, random_state=seed, n_init=1).fit(values)
    picked = []
    for cluster, center in enumerate(clusters.cluster_centers_):
        candidates = np.array(
            [i for i in np.flatnonzero(clusters.labels_ == cluster) if i not in picked],
            dtype=int,
        )
        # Identical embeddings can leave clusters empty. Fill from unused records.
        if not len(candidates):
            candidates = np.array([i for i in range(len(ids)) if i not in picked])
        distances = ((values[candidates] - center) ** 2).sum(axis=1)
        picked.append(int(candidates[np.argmin(distances)]))
    return [ids[i] for i in picked]
