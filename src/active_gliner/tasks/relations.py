"""Multi-label relation decisions over all supplied mention pairs."""

import math
import random

from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier

from active_gliner.data.records import Record
from active_gliner.evaluate.metrics import span_prf

from . import TrainingExample


def pairs(record: Record) -> list[tuple[int, int]]:
    size = len(record.gold["entities"])
    return [(i, j) for i in range(size) for j in range(size) if i != j]


def marked_text(record: Record, i: int, j: int) -> str:
    events = {}
    for index, marker in ((i, "H"), (j, "T")):
        mention = record.gold["entities"][index]
        events.setdefault(mention["start"], []).append((1, f"[{marker}:{mention['label']}] "))
        events.setdefault(mention["end"], []).append((0, f" [/{marker}]"))
    parts, offset = [], 0
    for position, markers in sorted(events.items()):
        parts.append(record.text[offset:position])
        parts.extend(text for _, text in sorted(markers))
        offset = position
    parts.append(record.text[offset:])
    return "".join(parts)


def to_training_example(record: Record, labels: dict[str, str]) -> list[dict]:
    gold = gold_items(record)
    return [
        TrainingExample(
            input=marked_text(record, i, j),
            output={
                "classifications": [
                    {
                        "task": "relations",
                        "labels": list(labels.values()),
                        "true_label": [
                            natural for raw, natural in labels.items() if (i, j, raw) in gold
                        ],
                        "multi_label": True,
                    }
                ]
            },
        )
        for i, j in pairs(record)
    ]


def subsample_negatives(examples: list[dict], ratio: float, seed: int) -> tuple[list[dict], dict]:
    """Keep positives and floor(ratio * positives) uniformly sampled negatives."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("The relation negative ratio must be finite and positive")
    negatives = [
        i
        for i, example in enumerate(examples)
        if not example["output"]["classifications"][0]["true_label"]
    ]
    positive_count = len(examples) - len(negatives)
    kept_count = min(len(negatives), math.floor(ratio * positive_count))
    dropped = set(negatives) - set(random.Random(seed).sample(negatives, kept_count))
    return [example for i, example in enumerate(examples) if i not in dropped], {
        "positive_pairs": positive_count,
        "negative_pairs_available": len(negatives),
        "negative_pairs": kept_count,
        "relation_neg_ratio": ratio,
        "seed": seed,
    }


def predict(
    model,
    records: list[Record],
    labels: dict[str, str],
    threshold: float = 0.5,
    batch_size: int = 32,
    **kwargs,
) -> list[dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidates = [pairs(record) for record in records]
    texts = [
        marked_text(record, i, j)
        for record, ordered in zip(records, candidates, strict=True)
        for i, j in ordered
    ]
    custom = getattr(model, "relation_head", "classifier") != "classifier"
    if custom:
        import torch

        from .relation_heads import pair_logits

        rows = [(record, i, j) for record in records for i, j in pairs(record)]
        probabilities_by_pair = []
        model.eval()
        with torch.inference_mode():
            if model.relation_head == "native":
                chunks = [[(record, i, j) for i, j in pairs(record)] for record in records]
            else:
                chunks = [
                    rows[start : start + batch_size] for start in range(0, len(rows), batch_size)
                ]
            for chunk in chunks:
                if chunk:
                    probabilities_by_pair.extend(
                        pair_logits(model, chunk, labels).float().sigmoid().cpu().tolist()
                    )
        custom_scores = iter(probabilities_by_pair)
    schema = ClassificationSchema().multi("relations", list(labels.values()))
    scores = iter(
        Classifier(model).batch_score(
            texts, schema, config=ClassificationConfig(batch_size=batch_size)
        )
        if texts and not custom
        else []
    )
    predictions = []
    for record, ordered in zip(records, candidates, strict=True):
        decisions, positive = [], []
        for i, j in ordered:
            if custom:
                probabilities = dict(zip(labels, next(custom_scores), strict=True))
            else:
                scored = next(scores)
                probabilities = {
                    raw: scored.probability("relations", natural) for raw, natural in labels.items()
                }
            selected = [raw for raw, p in probabilities.items() if p >= threshold]
            decisions.append(
                {"head": i, "tail": j, "probabilities": probabilities, "labels": selected}
            )
            positive.extend(
                {"head": i, "tail": j, "type": raw, "probability": probabilities[raw]}
                for raw in selected
            )
        predictions.append(
            {
                "id": record.id,
                "entities": record.gold["entities"],
                "pairs": decisions,
                "relations": positive,
                "confidence": min(
                    (abs(p - 0.5) for pair in decisions for p in pair["probabilities"].values()),
                    default=0.5,
                ),
            }
        )
    return predictions


def gold_items(record: Record) -> set[tuple]:
    return {(r["head_id"], r["tail_id"], r["type"]) for r in record.gold["relations"]}


def pred_items(pred: dict) -> set[tuple]:
    return {(r["head"], r["tail"], r["type"]) for r in pred["relations"]}


def score(records: list[Record], preds: list[dict], labels: dict[str, str]) -> dict:
    return span_prf([gold_items(r) for r in records], [pred_items(p) for p in preds])
