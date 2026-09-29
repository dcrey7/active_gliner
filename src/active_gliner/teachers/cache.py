import json
from pathlib import Path
from threading import Lock


class LabelCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._values = {}
        if path.exists():
            # A kill can leave only the final append incomplete. Drop that fragment.
            with path.open("rb+") as stream:
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        stream.seek(-len(line), 1)
                        stream.truncate()
                        break
                    row = json.loads(line)
                    self._values[row["id"]] = row["value"]

    @classmethod
    def for_job(cls, root, teacher, task, dataset, locale, prompt_hash) -> "LabelCache":
        return cls(Path(root) / teacher / f"{dataset}-{locale}" / f"{task}-{prompt_hash}.jsonl")

    def get(self, id: str) -> dict | None:
        with self._lock:
            return self._values.get(id)

    def put(self, id: str, value: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": id, "value": value}, ensure_ascii=False) + "\n")
                stream.flush()
            self._values[id] = value
