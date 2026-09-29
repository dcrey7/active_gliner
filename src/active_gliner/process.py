"""Run GPU work in a fresh interpreter and retain bounded failure output."""

import subprocess
import sys
import tempfile
from collections import deque


def run_child(module: str, args: list[str], *, stdin: str | None = None) -> tuple[int, str]:
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr:
        result = subprocess.run(
            [sys.executable, "-m", module, *args],
            input=stdin,
            text=True,
            stderr=stderr,
            check=False,
        )
        stderr.seek(0)
        tail = "".join(deque(stderr, maxlen=20))
    return result.returncode, tail
