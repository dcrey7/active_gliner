import hashlib
import json
import random
from pathlib import Path

from .records import Record

SPLITS = ("pool", "dev", "test")


def make_splits(source: dict[str, list[Record]], name: str) -> dict[str, list[Record]]:
    candidates = source["train"]
    if name in {"mit_movie", "crossre"}:
        if name == "crossre":
            candidates = candidates + source["dev"]
        ids = sorted(r.id for r in candidates)
        random.Random(20260925).shuffle(ids)
        holdout = set(ids[: 978 if name == "mit_movie" else 300])
        return {
            "pool": [r for r in candidates if r.id not in holdout],
            "dev": [r for r in candidates if r.id in holdout],
            "test": source["test"],
        }
    return {"pool": candidates, "dev": source["dev"], "test": source["test"]}


def _hash(ids: dict) -> str:
    payload = json.dumps({s: ids[s] for s in SPLITS}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_hash(splits: dict[str, list[Record]]) -> str:
    return _hash({s: [r.id for r in splits[s]] for s in SPLITS})


def write_split_ids(
    splits: dict[str, list[Record]], name: str, locale: str, out_dir: str | Path = "data/splits"
) -> Path:
    path = Path(out_dir) / f"{name}-{locale}.json"
    payload = {
        "dataset": name,
        "locale": locale,
        "hash": split_hash(splits),
        **{s: [r.id for r in splits[s]] for s in SPLITS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        if read_split_ids(path) != payload:
            raise ValueError(f"Frozen split IDs differ: {path}") from None
    return path


def read_split_ids(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["hash"] != _hash(payload):
        raise ValueError(f"Invalid split hash: {path}")
    ids = [record_id for s in SPLITS for record_id in payload[s]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate split IDs: {path}")
    return payload
