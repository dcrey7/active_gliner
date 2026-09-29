"""Small, flushed JSON writers."""

import json
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def watch(path: Path) -> None:
    print(f"{'Step':>8} {'Dev loss':>12} {'Dev F1':>12}")
    for row in read_jsonl(path / "eval_log.jsonl"):
        print(f"{row['step']:8d} {row['eval_loss']:12.4f} {row['dev_f1']:12.4f}")
