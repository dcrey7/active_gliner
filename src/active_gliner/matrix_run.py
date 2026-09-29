"""Execute one matrix config received from the parent."""

import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from active_gliner import model
from active_gliner.run import RunConfig, experiment_dir, run_experiment

# A memory gate at start is not enough: a MASSIVE run next door keeps growing, and
# Xid 109 faults kill runs. Both are retried in a fresh process.
RETRY_ENV = "ACTIVE_GLINER_GPU_ATTEMPT"
CONFIG_ENV = "ACTIVE_GLINER_RUN_CONFIG"
MAX_GPU_RETRIES = 10
GPU_RETRY_WAIT_S = 120.0
RETRYABLE_SIGNS = (
    "out of memory",
    "alloc_failed",
    "cudaerrormemoryallocation",
    "cuda_error_launch_failed",
    "unspecified launch failure",
)

# Peak GPU memory seen per run on the 3090 (nvidia-smi, 27 Sep 2026), plus a margin.
# Lanes share one GPU; a run that starts without room fails with out-of-memory.
NEED_GIB = {"ner": 7, "classification": 5, "relations": 7, "slots": 13}
WHOLE_POOL_RELATIONS_GIB = 15
EXCLUSIVE_TASKS = {"slots"}
TASK_OF = {
    "cleanconll": "ner",
    "bc5cdr": "ner",
    "mit_movie": "ner",
    "crossre": "relations",
    "hallmarks": "classification",
    "massive": "slots",
}


def needed_gib(cfg: RunConfig) -> float:
    task = TASK_OF.get(cfg.dataset, "slots")
    if task == "relations" and cfg.selector == "all":
        return WHOLE_POOL_RELATIONS_GIB
    return NEED_GIB[task]


def wait_for_gpu_memory(cfg: RunConfig, poll_s=20.0, max_wait_s=7200.0, free_gib=None) -> float:
    """Wait until the GPU has room for this run; give up waiting after max_wait_s.

    Waiting changes no result: the run starts later with the same config and seed.
    """
    if free_gib is None:
        import torch

        if not torch.cuda.is_available():
            return 0.0

        def free_gib():
            return torch.cuda.mem_get_info()[0] / 2**30

    need = needed_gib(cfg)
    started = time.monotonic()
    while free_gib() < need and time.monotonic() - started < max_wait_s:
        time.sleep(poll_s)
    return time.monotonic() - started


@contextmanager
def heavy_task_lock(cfg: RunConfig, root="runs"):
    """One slot run at a time: two at once often ended in Xid 109 GPU faults (27 Sep)."""
    if TASK_OF.get(cfg.dataset) not in EXCLUSIVE_TASKS:
        yield
        return
    import fcntl

    path = Path(root) / "locks" / f"{TASK_OF[cfg.dataset]}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def is_retryable_gpu_error(error: BaseException) -> bool:
    """Memory shortage, or a run killed by an Xid 109 GPU fault (all seen on 27 and 28 Sep).

    A fault leaves a sticky CUDA error in this process only; a new process starts clean.
    """
    text = "".join(traceback.format_exception(error)).lower()
    return any(sign in text for sign in RETRYABLE_SIGNS)


def free_gpu_safely() -> None:
    # After a GPU fault, freeing memory can raise too; that must not block the retry.
    try:
        model.free_gpu()
    except Exception as error:
        print(f"free_gpu failed: {error}", file=sys.stderr)


def retry_in_new_process(raw: str, cfg: RunConfig, attempt: int, wait_s: float) -> None:
    """Replace this process with a fresh one for the same run: clean folder, clean CUDA.

    exec keeps the process ID, so the parent lane still waits on it, and the old CUDA
    context is freed (a child process would keep this one alive while it waits).
    The run is deterministic from its config and seed, so a clean retry gives the
    result the first attempt would have given.
    """
    shutil.rmtree(experiment_dir(cfg), ignore_errors=True)
    time.sleep(wait_s)
    sys.stdout.flush()
    sys.stderr.flush()
    env = {**os.environ, RETRY_ENV: str(attempt + 1), CONFIG_ENV: raw}
    os.execve(sys.executable, [sys.executable, "-m", "active_gliner.matrix_run"], env)


def main() -> None:
    # A retry gets its config from the environment: stdin was read by the first attempt.
    raw = os.environ.get(CONFIG_ENV) or sys.stdin.read()
    cfg = RunConfig.model_validate_json(raw)
    attempt = int(os.environ.get(RETRY_ENV, "0"))
    try:
        with heavy_task_lock(cfg, cfg.out_root):
            wait_for_gpu_memory(cfg)
            run_experiment(cfg)
    except Exception as error:
        if not is_retryable_gpu_error(error) or attempt >= MAX_GPU_RETRIES:
            raise
        print(f"GPU error; retry {attempt + 1} of {MAX_GPU_RETRIES}: {error}", file=sys.stderr)
        free_gpu_safely()
        retry_in_new_process(raw, cfg, attempt, GPU_RETRY_WAIT_S)
    finally:
        free_gpu_safely()


if __name__ == "__main__":
    main()
