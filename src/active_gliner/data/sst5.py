"""SST-5: the sentiment of a movie-review sentence on five ordered levels."""

from pathlib import Path

from .common import json_rows
from .records import Record

SPLITS = ("train", "dev", "test")


def files(root: Path) -> list[Path]:
    return [root / f"{s}.jsonl" for s in SPLITS]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    splits = {}
    for split, path in zip(SPLITS, files(root), strict=True):
        splits[split] = [
            Record(
                f"sst5:{locale}:{split}-{i}",
                None,
                row["text"],
                "decision",
                locale,
                split,
                {"labels": [row["label_text"]]},
            )
            for i, row in enumerate(json_rows(path))
        ]
    return splits
