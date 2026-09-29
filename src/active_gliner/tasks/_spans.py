"""Shared entity and aggregate structure extraction."""

from active_gliner.data.records import Record


def training_values(record: Record, labels: dict[str, str]) -> dict[str, list[str]]:
    values = {natural: [] for natural in labels.values()}
    for span in record.gold["spans"]:
        values[labels[span["label"]]].append(
            ExactValue(span["text"], record.text, span["start"], span["end"], span["label"])
        )
    return values


def extract(
    model,
    records: list[Record],
    labels: dict[str, str],
    schema,
    threshold: float,
    batch_size: int,
    *,
    structure: bool = False,
) -> list[list[dict]]:
    outputs = model.batch_extract(
        [r.text for r in records],
        schema,
        threshold=threshold,
        batch_size=batch_size,
        include_confidence=True,
        include_spans=True,
        overlap_policy="allow",
    )
    raw = {natural: label for label, natural in labels.items()}
    return [
        [
            dict(span, label=raw[label])
            for fields in (output.get("slots", []) if structure else [output.get("entities", {})])
            for label, spans in fields.items()
            for span in spans
        ]
        for output in outputs
    ]


def entity_predictions(
    model, records: list[Record], labels: dict[str, str], threshold: float, batch_size: int
) -> list[list[dict]]:
    schema = model.create_schema().entities(list(labels.values()))
    return extract(model, records, labels, schema, threshold, batch_size)


class ExactValue(str):
    """An in-memory training value with its original character offsets."""

    def __new__(cls, value, source=None, start=None, end=None, label=None):
        instance = super().__new__(cls, value)
        instance.source = source
        instance.start = start
        instance.end = end
        instance.label = label
        return instance


def exact_token_span(value: ExactValue, splitter) -> tuple[int, int]:
    """Map an exact character span to inclusive word positions; never widen it."""
    tokens = list(splitter(value.source))
    starts = {start: i for i, (_, start, _) in enumerate(tokens)}
    ends = {end: i for i, (_, _, end) in enumerate(tokens)}
    if (
        value.start not in starts
        or value.end not in ends
        or value.source[value.start : value.end] != value
    ):
        raise ValueError(f"Training span does not align to word tokens: {value!r}")
    return starts[value.start], ends[value.end]


def training_targets_match(record: Record, example: dict) -> bool:
    """Check exact supervision, including repeated values and raw label identity."""
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    output = example["output"]
    groups = [output.get("entities", {})]
    groups.extend(
        fields for structure in output.get("json_structures", []) for fields in structure.values()
    )
    actual = [value for group in groups for values in group.values() for value in values]
    if any(not isinstance(value, ExactValue) for value in actual):
        return False
    try:
        for value in actual:
            exact_token_span(value, WhitespaceTokenSplitter())
    except ValueError:
        return False
    return example["input"] == record.text and {
        (v.start, v.end, v.label, str(v), v.source) for v in actual
    } == {(s["start"], s["end"], s["label"], s["text"], record.text) for s in record.gold["spans"]}
