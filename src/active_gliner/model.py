"""Load a fresh student from a pinned snapshot."""

import gc
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download

STUDENT_REPO = "fastino/gliner2.5-multi-v1"
STUDENT_SHA = "12fc40399dae672ce840c5e3c50a92340bff3c8c"


def free_gpu() -> None:
    """Release unwound exception frames and collect unused GPU objects.

    Call after dropping local model references or leaving the function that owns them.
    """
    import torch

    # Optuna retains exceptions while handling failed and pruned trials.
    pending = [sys.exception()]
    seen = set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending.extend((error.__cause__, error.__context__))
    gc.collect()
    torch.cuda.empty_cache()


def snapshot_path(repo: str, sha: str, local_files_only: bool = False) -> Path:
    # AutoExtractor(revision=) pins only config.json, so pin the whole snapshot.
    return Path(snapshot_download(repo, revision=sha, local_files_only=local_files_only))


def load_student(
    repo: str = STUDENT_REPO,
    sha: str = STUDENT_SHA,
    device: str = "cuda",
    local_files_only: bool = False,
    word_splitter: str | Callable | None = None,
) -> AutoExtractor:
    path = snapshot_path(repo, sha, local_files_only=local_files_only)
    # LoRA changes the model in place, so every run needs a fresh base.
    model = AutoExtractor.from_pretrained(str(path), map_location=device)
    if word_splitter is not None:
        model.set_word_splitter(word_splitter)
    return model


def sentence_embeddings(model, texts: list[str], batch_size: int):
    """Mean-pool sentence subwords, excluding padding and special tokens."""
    import numpy as np
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoder = model.encoder
    tokenizer = model.processor.tokenizer
    device = next(encoder.parameters()).device
    vectors = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=encoder.config.max_position_embeddings,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                ).to(device)
                special = batch.pop("special_tokens_mask")
                mask = (batch["attention_mask"].bool() & ~special.bool()).unsqueeze(-1)
                hidden = encoder(**batch).last_hidden_state
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                vectors.append(pooled.float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(vectors) if vectors else np.empty((0, encoder.config.hidden_size))


def load_adapter(student, path: str | Path):
    """Restore the head schema before PEFT restores adapters and marker weights."""
    import json

    from peft import PeftModel

    from active_gliner.tasks.relation_heads import configure

    metadata = Path(path) / "relation_head.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        configure(student, saved["head"], saved["labels"])
    return PeftModel.from_pretrained(student, path)
