"""Record model, data, and environment versions for a run."""

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch


def collect_pins(
    model_repo: str,
    model_sha: str,
    dataset_revisions: dict,
    extra: dict | None = None,
) -> dict:
    repo = Path(__file__).resolve().parents[2]
    git = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
        git = {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        # Git metadata can be absent in an installed package or source archive.
        pass

    packages = {name: version(name) for name in ("gliner2", "torch", "transformers", "peft")}
    packages["python"] = platform.python_version()
    available = torch.cuda.is_available()
    pins = {
        "model": {"repo": model_repo, "sha": model_sha},
        "packages": packages,
        "cuda": {
            "available": available,
            "version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name() if available else None,
        },
        "git": git,
        "datasets": dataset_revisions,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra is not None:
        pins["extra"] = extra
    return pins


def write_pins(path: str | Path, pins: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
