from pathlib import Path

from .records import Record


def files(root: Path) -> list[Path]:
    folder = root / "hf_parquet/hallmarks_of_cancer_bigbio_text"
    return [folder / s / "0000.parquet" for s in ("train", "validation", "test")]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    import pyarrow.parquet as pq

    splits = {}
    for split, path in zip(("train", "validation", "test"), files(root), strict=True):
        splits["dev" if split == "validation" else split] = [
            Record(
                f"hallmarks:{locale}:{row['document_id']}",
                row["document_id"].rsplit("_", 1)[0],
                row["text"],
                "classification",
                locale,
                split,
                {"labels": [label for label in row["labels"] if label != "none"]},
            )
            for row in pq.read_table(path).to_pylist()
        ]
    return splits
