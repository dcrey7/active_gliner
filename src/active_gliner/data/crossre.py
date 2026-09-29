from pathlib import Path

from .common import json_rows, one_file, token_offsets
from .records import Record

DOMAINS = ("ai", "literature", "music", "news", "politics", "science")


def files(root: Path) -> list[Path]:
    return [
        one_file(root, f"*/crossre_data/{d}-{s}.json")
        for s in ("train", "dev", "test")
        for d in DOMAINS
    ]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    splits = {s: [] for s in ("train", "dev", "test")}
    for path in files(root):
        split = path.stem.rsplit("-", 1)[1]
        for row in json_rows(path):
            text, starts, ends = token_offsets(row["sentence"])

            spans = {
                (a, b): {"start": starts[a], "end": ends[b], "text": text[starts[a] : ends[b]]}
                for a, b, _ in row["ner"]
            }

            mention_ids = {}
            for index, (a, b, _) in enumerate(row["ner"]):
                mention_ids.setdefault((starts[a], ends[b]), index)

            gold = {
                "entities": [dict(spans[a, b], label=label) for a, b, label in row["ner"]],
                "relations": [
                    {
                        "head": spans[r[0], r[1]], "tail": spans[r[2], r[3]],
                        "head_id": mention_ids[starts[r[0]], ends[r[1]]],
                        "tail_id": mention_ids[starts[r[2]], ends[r[3]]],
                        "type": r[4],
                    }
                    for r in row["relations"]
                ],
            }
            splits[split].append(
                Record(
                    f"crossre:{locale}:{row['doc_key']}",
                    None,
                    text,
                    "relations",
                    locale,
                    split,
                    gold,
                )
            )
    return splits
