"""Aggregate slot extraction, with an optional entity representation."""

from dataclasses import replace

from gliner2.training.data import InputExample, Structure

from active_gliner.confidence import slot_confidence
from active_gliner.data.records import Record
from active_gliner.decode import greedy_flat
from active_gliner.evaluate.metrics import canonical_value, record_accuracy, span_prf

from . import TrainingExample
from ._spans import extract, training_values


def _check_mode(slot_mode: str) -> None:
    if slot_mode not in {"structure", "entity"}:
        raise ValueError(f"Unknown slot mode: {slot_mode}")


def to_training_example(
    record: Record, labels: dict[str, str], slot_mode: str = "structure"
) -> dict:
    _check_mode(slot_mode)
    values = training_values(record, labels)
    if slot_mode == "entity":
        return TrainingExample(input=record.text, output={"entities": values})
    example = InputExample(record.text, structures=[Structure("slots", mode=None, **values)])
    result = example.to_dict()
    # from_dict defaults missing metadata to natural record mode, not aggregate mode.
    result["output"]["record_metadata"] = {"slots": {"mode": None}}
    return TrainingExample(result)


def predict(
    model,
    records: list[Record],
    labels: dict[str, str],
    threshold: float = 0.5,
    batch_size: int = 32,
    **kwargs,
) -> list[dict]:
    slot_mode = kwargs.get("slot_mode", "structure")
    _check_mode(slot_mode)
    structure = slot_mode == "structure"
    schema = model.create_schema()
    if structure:
        fields = schema.structure("slots", mode=None)
        for natural in labels.values():
            fields.field(natural, dtype="list")
    else:
        schema.entities(list(labels.values()))
    outputs = extract(model, records, labels, schema, threshold, batch_size, structure=structure)
    settings = model.boundary_settings
    try:
        if not structure:
            model.boundary_settings = replace(settings, abstention_threshold=1.0)
        candidates = extract(model, records, labels, schema, 0.0, batch_size, structure=structure)
    finally:
        model.boundary_settings = settings
    preds = []
    for record, spans, proposed in zip(records, outputs, candidates, strict=True):
        spans = greedy_flat(spans)
        max_below = max(
            (s["confidence"] for s in proposed if s["confidence"] < threshold), default=None
        )
        preds.append(
            {
                "id": record.id,
                "spans": spans,
                "max_below": max_below,
                "confidence": slot_confidence((s["confidence"] for s in spans), max_below),
            }
        )
    return preds


def gold_items(record: Record) -> set[tuple]:
    return pred_items(record.gold)


def pred_items(pred: dict) -> set[tuple]:
    return {(canonical_value(s["text"]), s["label"]) for s in pred["spans"]}


def score(records: list[Record], preds: list[dict], labels: dict[str, str]) -> dict:
    gold, predicted = [gold_items(r) for r in records], [pred_items(p) for p in preds]
    return {**span_prf(gold, predicted), "record_accuracy": record_accuracy(gold, predicted)}
