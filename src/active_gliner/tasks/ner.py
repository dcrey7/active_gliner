"""Flat named entity extraction."""

from active_gliner.confidence import min_confidence
from active_gliner.data.records import Record
from active_gliner.decode import greedy_flat
from active_gliner.evaluate.metrics import span_prf

from . import TrainingExample
from ._spans import entity_predictions, training_values


def to_training_example(record: Record, labels: dict[str, str]) -> dict:
    return TrainingExample(input=record.text, output={"entities": training_values(record, labels)})


def predict(
    model,
    records: list[Record],
    labels: dict[str, str],
    threshold: float = 0.5,
    batch_size: int = 32,
    **kwargs,
) -> list[dict]:
    outputs = entity_predictions(model, records, labels, threshold, batch_size)
    preds = []
    for record, spans in zip(records, outputs, strict=True):
        spans = greedy_flat(spans)
        preds.append(
            {
                "id": record.id,
                "spans": spans,
                "confidence": min_confidence(s["confidence"] for s in spans),
            }
        )
    return preds


def gold_items(record: Record) -> set[tuple]:
    return pred_items(record.gold)


def pred_items(pred: dict) -> set[tuple]:
    return {(s["start"], s["end"], s["label"]) for s in pred["spans"]}


def score(records: list[Record], preds: list[dict], labels: dict[str, str]) -> dict:
    return span_prf([gold_items(r) for r in records], [pred_items(p) for p in preds])
