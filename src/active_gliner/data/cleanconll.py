from pathlib import Path

from .common import conll_rows, one_file
from .records import Record


def files(root: Path) -> list[Path]:
    return [one_file(root, f"*/data/cleanconll/cleanconll.{s}") for s in ("train", "dev", "test")]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    splits = {}
    for split, path in zip(("train", "dev", "test"), files(root), strict=True):
        splits[split] = [
            Record(
                f"cleanconll:{locale}:{split}-{i}",
                f"{split}-{doc}",
                text,
                "ner",
                locale,
                split,
                {"spans": spans},
            )
            for i, (doc, text, spans) in enumerate(conll_rows(path))
        ]
    return splits
