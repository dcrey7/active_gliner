"""Frozen acquisition artifacts and run protocol identities."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

PROTOCOL_MODULES = (
    "data/",
    "tasks/",
    "teachers/",
    "evaluate/",
    "decode.py",
    "confidence.py",
    "selection.py",
    "exact_targets.py",
    "model.py",
    "train.py",
    "run.py",
    "protocol.py",
    "mixing.py",
)


def is_protocol_source(path: str) -> bool:
    """Check a POSIX path relative to the active_gliner package."""
    return (
        path.endswith(".py")
        and path != "teachers/stats.py"
        and any(
            path.startswith(module) if module.endswith("/") else path == module
            for module in PROTOCOL_MODULES
        )
    )


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint() -> str:
    root = Path(__file__).parent
    return digest(
        {
            p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(root.rglob("*.py"))
            if is_protocol_source(p.relative_to(root).as_posix())
        }
    )


@contextmanager
def inference_precision():
    precision = torch.get_float32_matmul_precision()
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@contextmanager
def artifact_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def pool_score_path(cfg, labels, records) -> Path:
    from active_gliner.model import STUDENT_SHA

    key = digest(
        {
            "dataset": cfg.dataset,
            "locale": cfg.locale,
            "threshold": cfg.acq_threshold,
            "student": STUDENT_SHA,
            "code": code_fingerprint(),
            "labels": labels,
            "slot_mode": cfg.slot_mode,
            "relation_head": cfg.relation_head,
            "inputs": [(r.id, r.text, r.gold.get("entities", [])) for r in records],
        }
    )
    return Path(cfg.out_root) / "pool_scores" / f"{cfg.dataset}-{cfg.locale}-{key}.jsonl"


def embedding_path(cfg, records) -> Path:
    """One shared embedding file per pool, so paired diversity arms cluster identical values.

    Recomputing embeddings per run let GPU rounding move a sentence to another cluster,
    and paired arms then selected different pool IDs.
    """
    from active_gliner.model import STUDENT_SHA

    key = digest(
        {
            "dataset": cfg.dataset,
            "locale": cfg.locale,
            "student": STUDENT_SHA,
            "code": code_fingerprint(),
            "inputs": [(r.id, r.text) for r in records],
        }
    )
    return Path(cfg.out_root) / "pool_scores" / f"embeddings-{cfg.dataset}-{cfg.locale}-{key}.npy"


def protocol_fingerprint(cfg, pool_score_hash: str, prompt_hash: str) -> str:
    from active_gliner.model import STUDENT_SHA

    config = cfg.model_dump(exclude={"out_root", "block", "recipe_file"})
    recipe_hash = file_hash(Path(cfg.recipe_file)) if cfg.recipe_file else None
    return digest(
        {
            "config": config,
            "recipe": recipe_hash,
            "prompt": prompt_hash,
            "student": STUDENT_SHA,
            "pool_scores": pool_score_hash,
            "code": code_fingerprint(),
        }
    )


def current_protocol(cfg) -> str | None:
    from active_gliner import data, tasks
    from active_gliner.teachers import prompts

    splits = data.load(cfg.dataset, cfg.locale)
    labels = tasks.labels(cfg.dataset, splits)
    records = splits["pool"][: cfg.pool_limit]
    path = pool_score_path(cfg, labels, records)
    if not path.exists():
        return None
    version = prompts.version_for(cfg.dataset)
    prompt_hash = prompts.prompt_hash(
        records[0].task, labels, prompts.prompt_for(cfg.dataset, version)
    )
    return protocol_fingerprint(cfg, file_hash(path), prompt_hash)


def assert_selection(cfg, score_hash: str, selected_ids: list[str]) -> str:
    """Persist one selection identity for paired arms under the same protocol."""
    key = digest(
        {
            "dataset": cfg.dataset,
            "locale": cfg.locale,
            "selector": cfg.selector,
            "n": cfg.n,
            "seed": cfg.seed,
            "scores": score_hash,
            "no_prediction": cfg.no_prediction,
            # Identical pool scores under new code must not match a selection made by old code.
            "code": code_fingerprint(),
        }
    )
    path = Path(cfg.out_root) / "pool_scores" / f"selection-{key}.json"
    fingerprint = digest(selected_ids)
    with artifact_lock(path):
        if path.exists():
            if json.loads(path.read_text()) != fingerprint:
                raise ValueError("Paired arms selected different pool IDs")
        else:
            path.write_text(json.dumps(fingerprint))
    return fingerprint
