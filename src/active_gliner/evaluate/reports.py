"""CSV output for per-label metrics and calibration."""

import csv
from pathlib import Path

from .errors import items


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def per_type_rows(task, metrics: dict, predictions) -> list[dict]:
    confidences = {}
    for prediction in predictions:
        for key, _item, confidence in items(task, prediction):
            label = key if isinstance(key, str) else key[-1]
            confidences.setdefault(label, []).append(confidence)
    rows = []
    for label, scores in sorted(metrics["per_label"].items()):
        values = confidences.get(label, [])
        rows.append(
            {
                "label": label,
                **scores,
                "mean_confidence": sum(values) / len(values) if values else None,
            }
        )
    return rows
