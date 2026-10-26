"""CLINC150 (plus): pick the intent of a short request from 150, or out of scope."""

import json
from pathlib import Path

from .records import Record

SPLITS = ("train", "validation", "test")


def files(root: Path) -> list[Path]:
    return [root / "plus" / f"{s}-00000-of-00001.parquet" for s in SPLITS]


def intent_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    return meta["info"]["features"]["intent"]["names"]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    import pyarrow.parquet as pq

    names = intent_names(files(root)[0])
    splits = {}
    for split, path in zip(SPLITS, files(root), strict=True):
        ours = "dev" if split == "validation" else split
        splits[ours] = [
            Record(
                f"clinc150:{locale}:{ours}-{i}",
                None,
                row["text"],
                "decision",
                locale,
                ours,
                {"labels": [names[row["intent"]]]},
            )
            for i, row in enumerate(pq.read_table(path).to_pylist())
        ]
    return splits
