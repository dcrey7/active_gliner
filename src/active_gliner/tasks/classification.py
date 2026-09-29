"""Hallmarks classification with independent decisions for every label."""

from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier

from active_gliner.confidence import classification_confidence
from active_gliner.data.records import Record
from active_gliner.evaluate.metrics import classification_prf

from . import TrainingExample


def to_training_example(record: Record, labels: dict[str, str]) -> dict:
    # GLiNER2 accepts true_label=[] as an all-negative multi-label example.
    return TrainingExample(
        {
            "input": record.text,
            "output": {
                "classifications": [
                    {
                        "task": "hallmarks",
                        "labels": list(labels.values()),
                        "true_label": [labels[label] for label in record.gold["labels"]],
                        "multi_label": True,
                    }
                ]
            },
        }
    )


def predict(
    model,
    records: list[Record],
    labels: dict[str, str],
    threshold: float = 0.5,
    batch_size: int = 32,
    **kwargs,
) -> list[dict]:
    schema = ClassificationSchema().multi("hallmarks", list(labels.values()))
    scores = Classifier(model).batch_score(
        [r.text for r in records], schema, config=ClassificationConfig(batch_size=batch_size)
    )
    preds = []
    for record, scored in zip(records, scores, strict=True):
        probs = {raw: scored.probability("hallmarks", natural) for raw, natural in labels.items()}
        preds.append(
            {
                "id": record.id,
                "probabilities": probs,
                "labels": [raw for raw, p in probs.items() if p >= threshold],
                "confidence": classification_confidence(probs, threshold),
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
