"""Download Universal NER English-EWT (CC BY-SA 4.0) into the raw cache at a pinned commit.

The Hugging Face copy (universalner/universal_ner) is a loading script that reads the
unpinned GitHub master, so this script reads the GitHub files at a fixed commit instead.
Writes <raw>/uner_ewt/source.json with the repo, the commit and the files (research rule 7).
"""

import json
import urllib.request

from active_gliner.data.common import raw_dir

REPO = "UniversalNER/UNER_English-EWT"
COMMIT = "8ed072de2fd24022cc62458997fe96a8fe191ea4"
FILES = [f"en_ewt-ud-{split}.iob2" for split in ("train", "dev", "test")] + ["LICENSE.txt"]


def fetch() -> None:
    root = raw_dir("uner_ewt")
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}"
        with urllib.request.urlopen(url) as response:
            (root / name).write_bytes(response.read())
    source = {"repo": f"https://github.com/{REPO}", "revision": COMMIT, "files": FILES}
    (root / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print("uner_ewt", REPO, COMMIT)


if __name__ == "__main__":
    fetch()
