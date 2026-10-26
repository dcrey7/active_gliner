"""Decisions: one input, exactly one answer from a fixed option list.

Design: docs/research/2026-10-18-1345-decision-tasks-design.md. The student uses the
single-label classification head (softmax over the options). Gold and predictions are
one-item sets, so micro-F1 over decisions equals accuracy. An invalid teacher answer is
an empty set: it trains nothing and counts as a miss.
"""

from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier

from active_gliner.data.records import Record
from active_gliner.evaluate.metrics import classification_prf

from . import TrainingExample

TASK = "decision"


def to_training_example(record: Record, labels: dict[str, str]) -> dict | None:
    if not record.gold["labels"]:
        return None
    return TrainingExample(
        {
            "input": record.text,
            "output": {
                "classifications": [
                    {
                        "task": TASK,
                        "labels": list(labels.values()),
                        "true_label": [labels[record.gold["labels"][0]]],
                        "multi_label": False,
                    }
                ]
            },
        }
    )


def predict(
    model, records: list[Record], labels: dict[str, str], batch_size: int = 32, **kwargs
) -> list[dict]:
    schema = ClassificationSchema().single(TASK, list(labels.values()))
    scores = Classifier(model).batch_score(
        [r.text for r in records], schema, config=ClassificationConfig(batch_size=batch_size)
    )
    preds = []
    for record, scored in zip(records, scores, strict=True):
        probs = {raw: scored.probability(TASK, natural) for raw, natural in labels.items()}
        best = max(probs, key=probs.get)
        preds.append(
            {
                "id": record.id,
                "probabilities": probs,
                "labels": [best],
                # Least confident first: the lowest top-option probability.
                "confidence": probs[best],
            }
        )
    return preds


def gold_items(record: Record) -> set[str]:
    return set(record.gold["labels"])


def pred_items(pred: dict) -> set[str]:
    return set(pred["labels"])


def score(records: list[Record], preds: list[dict], labels: dict[str, str]) -> dict:
    return classification_prf(
        [gold_items(r) for r in records], [pred_items(p) for p in preds], list(labels)
    )
