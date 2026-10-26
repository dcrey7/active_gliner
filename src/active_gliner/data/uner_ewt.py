"""Universal NER English-EWT (Mayhew et al. 2024, CC BY-SA 4.0): web text with PER, ORG, LOC.

The open general-domain NER set next to CleanCoNLL, whose Reuters text is not open.
Each row is "index, token, tag, annotator, annotator"; the tag is the third column.
"""

from collections.abc import Iterator
from pathlib import Path

from .common import bio_spans, one_file
from .records import Record


def files(root: Path) -> list[Path]:
    return [one_file(root, f"en_ewt-ud-{s}.iob2") for s in ("train", "dev", "test")]


def rows(path: Path) -> Iterator[tuple[str, str, list[dict]]]:
    document, tokens, tags = None, [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.startswith("# newdoc id = "):
                document = line.removeprefix("# newdoc id = ")
            elif line.startswith("#"):
                continue
            elif line.strip():
                columns = line.split("\t")
                tokens.append(columns[1])
                tags.append(columns[2])
            elif tokens:
                yield document, *bio_spans(tokens, tags)
                tokens, tags = [], []
    if tokens:
        yield document, *bio_spans(tokens, tags)


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    splits = {}
    for split, path in zip(("train", "dev", "test"), files(root), strict=True):
        splits[split] = [
            Record(f"uner_ewt:{locale}:{split}-{i}", doc, text, "ner", locale, split, {"spans": spans})
            for i, (doc, text, spans) in enumerate(rows(path))
        ]
    return splits
