"""Exact errors with span and pair diagnostics."""


def items(task, prediction: dict) -> list[tuple]:
    if "pairs" in prediction:
        return [
            (
                (pair["head"], pair["tail"], label),
                {"head": pair["head"], "tail": pair["tail"], "type": label},
                p,
            )
            for pair in prediction["pairs"]
            for label, p in pair["probabilities"].items()
        ]
    if "probabilities" in prediction:
        return [(label, {"label": label}, p) for label, p in prediction["probabilities"].items()]
    key = "relations" if "relations" in prediction else "spans"
    return [
        (
            next(iter(task.pred_items({key: [item]}))),
            item,
            item["probability"] if key == "relations" else item["confidence"],
        )
        for item in prediction[key]
    ]


def _kind(item: dict, gold: dict) -> str:
    if "label" in item and "start" in item:
        for expected in gold.get("spans", []):
            same_span = (item["start"], item["end"]) == (expected["start"], expected["end"])
            if same_span and item["label"] != expected["label"]:
                return "wrong_label"
        for expected in gold.get("spans", []):
            overlap = max(item["start"], expected["start"]) < min(item["end"], expected["end"])
            if overlap and item["label"] == expected["label"]:
                return "wrong_boundary"
    if "head" in item:
        for expected in gold.get("relations", []):
            same_pair = (item["head"], item["tail"]) == (expected["head_id"], expected["tail_id"])
            if same_pair and item["type"] != expected["type"]:
                return "wrong_label"
    return "false_positive"


def error_rows(task, records, predictions) -> list[dict]:
    rows = []
    for record, prediction in zip(records, predictions, strict=True):
        gold, actual = task.gold_items(record), task.pred_items(prediction)
        context = {"id": record.id, "text": record.text}
        emitted = (
            {"relations": prediction["relations"]} if "relations" in prediction else prediction
        )
        for key, item, confidence in items(task, emitted):
            if key in actual - gold:
                rows.append(
                    {
                        **context,
                        "kind": _kind(item, record.gold),
                        "item": item,
                        "confidence": confidence,
                    }
                )
        # Keep every missing gold item, including those paired with a diagnostic above.
        gold_key = "relations" if "relations" in record.gold else "spans"
        for item in record.gold.get(gold_key, []):
            key = (
                (item["head_id"], item["tail_id"], item["type"])
                if gold_key == "relations"
                else next(iter(task.pred_items({gold_key: [item]})))
            )
            if key in gold - actual:
                rows.append({**context, "kind": "false_negative", "item": item})
        for label in record.gold.get("labels", []):
            if label in gold - actual:
                rows.append({**context, "kind": "false_negative", "item": {"label": label}})
    return rows
