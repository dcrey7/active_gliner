"""Download the decision datasets into the raw cache at a pinned revision.

Writes <raw>/<name>/source.json with the repo, the revision sha and the files, so every
run can record the exact data it read (research rule 7).
"""

import json
import sys

from huggingface_hub import HfApi, hf_hub_download

from active_gliner.data.common import raw_dir

SOURCES = {
    "boolq": (
        "google/boolq",
        ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
    ),
    "clinc150": (
        "clinc/clinc_oos",
        [f"plus/{split}-00000-of-00001.parquet" for split in ("train", "validation", "test")],
    ),
    "sst5": ("SetFit/sst5", ["train.jsonl", "dev.jsonl", "test.jsonl"]),
}


def fetch(name: str) -> None:
    repo, files = SOURCES[name]
    sha = HfApi().dataset_info(repo).sha
    root = raw_dir(name)
    root.mkdir(parents=True, exist_ok=True)
    for path in files:
        hf_hub_download(repo, path, repo_type="dataset", revision=sha, local_dir=root)
    source = {"repo": repo, "revision": sha, "files": files}
    (root / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(name, repo, sha)


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        fetch(name)
