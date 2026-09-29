from pathlib import Path

from .common import conll_rows
from .records import Record


def files(root: Path) -> list[Path]:
    return [root / f"MIT_movies_fixed_{s}.tsv" for s in ("train", "test")]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    return {
        split: [
            Record(
                f"mit_movie:{locale}:{split}-{i}",
                None,
                text,
                "ner",
                locale,
                split,
                {"spans": spans},
            )
            for i, (_, text, spans) in enumerate(conll_rows(path))
        ]
        for split, path in zip(("train", "test"), files(root), strict=True)
    }
