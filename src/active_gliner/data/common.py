import json
import os
from collections.abc import Iterator
from pathlib import Path


def raw_dir(name: str) -> Path:
    return (
        Path(os.environ.get("ACTIVE_GLINER_RAW", "~/.cache/active_gliner/raw")).expanduser() / name
    )


def one_file(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Missing raw file: {root / pattern}")
    if len(paths) != 1:
        raise ValueError(f"Ambiguous raw files: {paths}")
    return paths[0]


def json_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def token_offsets(tokens: list[str]) -> tuple[str, list[int], list[int]]:
    starts, ends = [], []
    offset = 0
    for token in tokens:
        starts.append(offset)
        ends.append(offset + len(token))
        offset += len(token) + 1
    return " ".join(tokens), starts, ends


def bio_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[dict]]:
    if len(tokens) != len(tags):
        raise ValueError("Token and tag counts differ")
    text, starts, ends = token_offsets(tokens)
    spans = []
    active = None
    for i, tag in enumerate(tags):
        if tag == "O":
            active = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix not in {"B", "I", "E", "S"}:
            raise ValueError(f"Unknown BIOES tag: {tag}")
        # An orphan I/E starts a mention, preserving the source annotation.
        if prefix in {"B", "S"} or active is None or active["label"] != label:
            active = {"start": starts[i], "end": ends[i], "label": label}
            spans.append(active)
        active["end"] = ends[i]
        active["text"] = text[active["start"] : active["end"]]
        if prefix in {"E", "S"}:
            active = None
    return text, spans


def conll_rows(path: Path) -> Iterator[tuple[int, str, list[dict]]]:
    tokens, tags = [], []
    document = -1
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            columns = line.split()
            if not columns or columns[0] == "-DOCSTART-":
                if tokens:
                    text, spans = bio_spans(tokens, tags)
                    yield document, text, spans
                    tokens, tags = [], []
                if columns:
                    document += 1
            else:
                tokens.append(columns[0])
                tags.append(columns[-1])
    if tokens:
        text, spans = bio_spans(tokens, tags)
        yield document, text, spans
