"""BoolQ: a yes/no question about a passage. The HF test split has no labels, so the HF
validation split is our test set; make_splits holds out dev from train."""

from pathlib import Path

from .records import Record


def files(root: Path) -> list[Path]:
    return [root / "data" / f"{s}-00000-of-00001.parquet" for s in ("train", "validation")]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    import pyarrow.parquet as pq

    splits = {}
    for split, path in zip(("train", "test"), files(root), strict=True):
        splits[split] = [
            Record(
                f"boolq:{locale}:{split}-{i}",
                None,
                f"question: {row['question']}?\npassage: {row['passage']}",
                "decision",
                locale,
                split,
                {"labels": ["yes" if row["answer"] else "no"]},
            )
            for i, row in enumerate(pq.read_table(path).to_pylist())
        ]
    return splits
