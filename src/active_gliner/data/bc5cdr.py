import json
from pathlib import Path
from zipfile import ZipFile

from .common import bio_spans, json_rows
from .records import Record


def files(root: Path) -> list[Path]:
    return [
        root / "tner_main/dataset" / f"{s}.json" for s in ("train", "valid", "test", "label")
    ] + [root / "bigbio_main/CDR_Data.zip"]


def document_ids(rows: list[dict], archive: Path, split: str) -> list[str]:
    source = {"train": "Training", "valid": "Development", "test": "Test"}[split]
    member = f"CDR_Data/CDR.Corpus.v010516/CDR_{source}Set.PubTator.txt"
    documents = {}
    with ZipFile(archive) as zip_file:
        for line in zip_file.read(member).decode("utf-8").splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[1] in {"t", "a"}:
                documents[parts[0]] = documents.get(parts[0], "") + "".join(parts[2].split())
    docs = list(documents.items())
    current, offset = 0, 0
    ids, missing = [], []
    for i, row in enumerate(rows):
        sentence = "".join("".join(row["tokens"]).split())
        for index in range(current, len(docs)):
            position = docs[index][1].find(sentence, offset if index == current else 0)
            if position >= 0:
                current, offset = index, position + len(sentence)
                ids.append(docs[index][0])
                break
        else:
            # Six released rows lose leading words; retain their neighbour's PMID.
            ids.append(docs[current][0])
            missing.append(i)
    for i in missing:
        if i == 0 or i + 1 == len(ids) or ids[i - 1] != ids[i + 1]:
            raise ValueError(f"BC5CDR {split} row {i}: ambiguous neighbour PMID")
    return ids


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    folder = root / "tner_main/dataset"
    labels = {value: key for key, value in json.loads((folder / "label.json").read_text()).items()}
    splits = {}
    for split in ("train", "valid", "test"):
        rows = list(json_rows(folder / f"{split}.json"))
        pmids = document_ids(rows, root / "bigbio_main/CDR_Data.zip", split)
        records = []
        for i, (row, pmid) in enumerate(zip(rows, pmids, strict=True)):
            text, spans = bio_spans(row["tokens"], [labels[tag] for tag in row["tags"]])
            records.append(
                Record(
                    f"bc5cdr:{locale}:{split}-{i}",
                    pmid,
                    text,
                    "ner",
                    locale,
                    split,
                    {"spans": spans},
                )
            )
        splits["dev" if split == "valid" else split] = records
    return splits
