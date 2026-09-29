import re
from pathlib import Path

from .common import json_rows
from .records import Record

SLOT = re.compile(r"\[(\S+?) : (.*?)\]")


def files(root: Path) -> list[Path]:
    return [root / "1.1/data" / f"{locale}.jsonl" for locale in ("en-US", "fr-FR")]


def load(root: Path, locale: str) -> dict[str, list[Record]]:
    splits = {s: [] for s in ("train", "dev", "test")}
    for row in json_rows(root / "1.1/data" / f"{locale}.jsonl"):
        annotated, text, cursor, spans = row["annot_utt"], "", 0, []
        for match in SLOT.finditer(annotated):
            text += annotated[cursor : match.start()]
            start = len(text)
            text += match[2]
            spans.append({"start": start, "end": len(text), "label": match[1], "text": match[2]})
            cursor = match.end()
        text += annotated[cursor:]
        if text != row["utt"]:
            raise ValueError(f"MASSIVE annotation does not match utt: {locale}:{row['id']}")
        split = row["partition"]
        splits[split].append(
            Record(
                f"massive:{locale}:{row['id']}",
                None,
                row["utt"],
                "slots",
                locale,
                split,
                {"spans": spans},
            )
        )
    return splits
