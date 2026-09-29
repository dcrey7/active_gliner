"""The memory gate: a run waits for GPU room instead of failing with out-of-memory."""

from active_gliner import matrix, matrix_run, protocol


def _cfg(dataset, selector="min"):
    return next(r for r in matrix.build() if r.dataset == dataset and r.selector == selector)


def test_needed_memory_per_task():
    assert matrix_run.needed_gib(_cfg("massive")) == matrix_run.NEED_GIB["slots"]
    assert matrix_run.needed_gib(_cfg("crossre")) == matrix_run.NEED_GIB["relations"]
    assert matrix_run.needed_gib(_cfg("crossre", "all")) == matrix_run.WHOLE_POOL_RELATIONS_GIB
    assert matrix_run.needed_gib(_cfg("cleanconll")) == matrix_run.NEED_GIB["ner"]


def test_waits_until_memory_is_free():
    readings = iter([2.0, 4.0, 20.0])
    waited = matrix_run.wait_for_gpu_memory(
        _cfg("massive"), poll_s=0, free_gib=lambda: next(readings)
    )
    assert waited >= 0
    assert next(readings, None) is None  # all three readings used


def test_gives_up_waiting_after_the_limit():
    calls = []

    def never_free():
        calls.append(1)
        return 0.0

    matrix_run.wait_for_gpu_memory(_cfg("massive"), poll_s=0, max_wait_s=0, free_gib=never_free)
    assert calls == [1]


def test_only_one_slot_run_holds_the_gpu(tmp_path):
    import fcntl

    import pytest

    lock = tmp_path / "locks" / "slots.lock"
    with matrix_run.heavy_task_lock(_cfg("massive"), tmp_path):
        with lock.open("a") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free again afterwards


def test_other_tasks_take_no_lock(tmp_path):
    with matrix_run.heavy_task_lock(_cfg("cleanconll"), tmp_path):
        pass
    assert not (tmp_path / "locks").exists()


def test_out_of_memory_is_recognised():
    assert matrix_run.is_retryable_gpu_error(RuntimeError("CUDA error: out of memory"))
    assert matrix_run.is_retryable_gpu_error(MemoryError("CUDA Out Of Memory. Tried to allocate"))
    assert matrix_run.is_retryable_gpu_error(
        RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`")
    )
    assert matrix_run.is_retryable_gpu_error(
        RuntimeError("CUDA error: unspecified launch failure")  # killed by an Xid 109 fault
    )
    assert not matrix_run.is_retryable_gpu_error(
        ValueError("Paired arms selected different pool IDs")
    )


def test_a_failing_cleanup_does_not_block_the_retry(monkeypatch):
    retries = _main_with(monkeypatch, RuntimeError("CUDA_ERROR_LAUNCH_FAILED"), attempt=0)

    def broken():
        raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setattr(matrix_run.model, "free_gpu", broken)
    matrix_run.main()
    assert retries == [0]


def test_retry_replaces_the_process_with_a_clean_folder(tmp_path, monkeypatch):
    cfg = _cfg("crossre").model_copy(update={"out_root": str(tmp_path)})
    folder = matrix_run.experiment_dir(cfg)
    folder.mkdir(parents=True)
    (folder / "train_log.jsonl").write_text("{}\n")
    seen = {}

    def fake_execve(path, argv, env):
        seen.update(
            argv=argv[1:], attempt=env[matrix_run.RETRY_ENV], config=env[matrix_run.CONFIG_ENV]
        )

    monkeypatch.setattr(matrix_run.os, "execve", fake_execve)
    matrix_run.retry_in_new_process("RAW", cfg, attempt=2, wait_s=0)
    assert not folder.exists()
    assert seen == {"argv": ["-m", "active_gliner.matrix_run"], "attempt": "3", "config": "RAW"}


def test_a_retry_reads_its_config_from_the_environment(monkeypatch):
    import io

    cfg = _cfg("cleanconll")
    monkeypatch.setenv(matrix_run.CONFIG_ENV, cfg.model_dump_json())
    monkeypatch.setattr(matrix_run.sys, "stdin", io.StringIO(""))  # stdin is empty on exec
    monkeypatch.setattr(matrix_run, "wait_for_gpu_memory", lambda cfg: 0.0)
    monkeypatch.setattr(matrix_run.model, "free_gpu", lambda: None)
    ran = []
    monkeypatch.setattr(matrix_run, "run_experiment", lambda c: ran.append(c.dataset))
    matrix_run.main()
    assert ran == ["cleanconll"]


def _main_with(monkeypatch, error, attempt):
    import io

    cfg = _cfg("crossre")
    monkeypatch.setattr(matrix_run.sys, "stdin", io.StringIO(cfg.model_dump_json()))
    monkeypatch.delenv(matrix_run.CONFIG_ENV, raising=False)
    monkeypatch.setenv(matrix_run.RETRY_ENV, str(attempt))
    monkeypatch.setattr(matrix_run, "wait_for_gpu_memory", lambda cfg: 0.0)
    monkeypatch.setattr(matrix_run.model, "free_gpu", lambda: None)

    def fail(cfg):
        raise error

    monkeypatch.setattr(matrix_run, "run_experiment", fail)
    retries = []
    monkeypatch.setattr(matrix_run, "retry_in_new_process", lambda *a: retries.append(a[2]) or 0)
    return retries


def test_main_retries_out_of_memory(monkeypatch):
    retries = _main_with(monkeypatch, RuntimeError("CUDA out of memory"), attempt=0)
    matrix_run.main()  # the real retry replaces the process and never returns
    assert retries == [0]


def test_main_stops_retrying_at_the_limit(monkeypatch):
    import pytest

    _main_with(monkeypatch, RuntimeError("CUDA out of memory"), matrix_run.MAX_GPU_RETRIES)
    with pytest.raises(RuntimeError):
        matrix_run.main()


def test_main_does_not_retry_other_errors(monkeypatch):
    import pytest

    retries = _main_with(monkeypatch, ValueError("bad labels"), attempt=0)
    with pytest.raises(ValueError):
        matrix_run.main()
    assert retries == []


def test_the_gate_is_not_part_of_the_protocol():
    # Scheduling must not make finished runs stale.
    assert not protocol.is_protocol_source("matrix_run.py")


def test_out_of_memory_in_a_step_fails_the_run():
    # gliner2 would skip the batch and train on; the run must fail and be retried instead.
    import pytest
    import torch

    from active_gliner.train import RunTrainer

    error = torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        RunTrainer._record_failed_batch(object(), None, data_loader_step=3, error=error)
    assert matrix_run.is_retryable_gpu_error(error)
