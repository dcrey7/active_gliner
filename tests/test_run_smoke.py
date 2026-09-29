"""Step 4: one small end-to-end run writes every file in refactor plan section 5.

Marked `model`: a real 20-step LoRA run on a small MIT Movie subset.
"""

import csv
import json

import pytest
import torch

from active_gliner import data, model
from active_gliner.run import RunConfig, run_experiment

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    try:
        model.snapshot_path(model.STUDENT_REPO, model.STUDENT_SHA, local_files_only=True)
    except Exception:
        pytest.skip("student snapshot not cached")
    if not data.raw_available("mit_movie"):
        pytest.skip("raw data not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("needs the GPU")
    cfg = RunConfig(
        dataset="mit_movie",
        locale="en-US",
        selector="min",
        n=48,
        seed=1,
        labels_source="ground_truth",
        max_steps=20,
        eval_steps=10,
        batch_size=8,
        pool_limit=300,  # smoke only: score a 300-sentence slice of the pool
        dev_limit=32,
        test_limit=32,
        out_root=str(tmp_path_factory.mktemp("runs")),
    )
    return run_experiment(cfg)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_every_file_exists(run_dir):
    for name in [
        "config.yaml",
        "pins.json",
        "hashes.json",
        "selected_ids.json",
        "pool_scores.jsonl",
        "train_log.jsonl",
        "eval_log.jsonl",
        "metrics.json",
        "per_type.csv",
        "calibration.csv",
        "errors.jsonl",
        "predictions/dev.jsonl",
        "predictions/test.jsonl",
        "plots/training_curves.png",
        "plots/calibration.png",
        "report.md",
        "adapter/best/adapter_config.json",
    ]:
        assert (run_dir / name).exists(), name


def test_run_folder_name(run_dir):
    assert run_dir.name == "min-N48-seed1"
    assert run_dir.parent.name == "ground_truth"


def test_selection_is_from_pool_scores(run_dir):
    selected = json.loads((run_dir / "selected_ids.json").read_text())
    scores = _jsonl(run_dir / "pool_scores.jsonl")
    assert len(scores) == 300
    assert len(selected) == 48
    by_id = {s["id"]: s["confidence"] for s in scores}
    picked = sorted(by_id[i] for i in selected)
    rest = sorted(c for i, c in by_id.items() if i not in set(selected))
    assert picked[-1] <= rest[0]  # min selection: every pick is at most every non-pick


def test_logs_are_live_and_complete(run_dir):
    train = _jsonl(run_dir / "train_log.jsonl")
    assert len(train) >= 2
    for row in train:
        assert {"step", "loss", "learning_rate", "gpu_gb", "cpu_percent", "time_s"} <= set(row)
    evals = _jsonl(run_dir / "eval_log.jsonl")
    assert [e["step"] for e in evals] == [10, 20]
    for row in evals:
        assert {"eval_loss", "dev_f1"} <= set(row)


def test_metrics_and_best_checkpoint(run_dir):
    m = json.loads((run_dir / "metrics.json").read_text())
    evals = _jsonl(run_dir / "eval_log.jsonl")
    best = max(evals, key=lambda e: e["dev_f1"])
    assert m["best_step"] == best["step"]
    # Training logs dev F1 at 0.5; the final dev score uses the threshold chosen on dev.
    scores = {float(t): f1 for t, f1 in m["eval_threshold_dev_scores"].items()}
    assert set(scores) == {0.3, 0.4, 0.5, 0.6, 0.7}
    assert scores[0.5] == pytest.approx(best["dev_f1"], abs=1e-6)
    assert m["eval_threshold_dev_chosen"] in scores
    assert scores[m["eval_threshold_dev_chosen"]] == max(scores.values())
    assert m["dev"]["micro"]["f1"] == pytest.approx(
        scores[m["eval_threshold_dev_chosen"]], abs=1e-6
    )
    assert 0.0 <= m["test"]["micro"]["f1"] <= 1.0
    for key in ["wall_time_s", "train_time_s", "stop_reason", "seeds"]:
        assert key in m
    assert set(m["seeds"]) == {"selection", "data_order", "lora_init"}


def test_pins_and_hashes(run_dir):
    pins = json.loads((run_dir / "pins.json").read_text())
    assert pins["model"]["sha"] == model.STUDENT_SHA
    hashes = json.loads((run_dir / "hashes.json").read_text())
    assert {"split", "schema", "config"} <= set(hashes)


def test_per_type_and_calibration_tables(run_dir):
    rows = list(csv.DictReader((run_dir / "per_type.csv").open()))
    assert {"label", "precision", "recall", "f1", "tp", "fp", "fn", "support"} <= set(rows[0])
    bins = list(csv.DictReader((run_dir / "calibration.csv").open()))
    assert [b["bin"] for b in bins] == ["[0.00,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]
    assert {"count", "mean_confidence", "correctness"} <= set(bins[0])


def test_errors_have_kinds_and_confidence(run_dir):
    errors = _jsonl(run_dir / "errors.jsonl")
    kinds = {"false_positive", "false_negative", "wrong_label", "wrong_boundary"}
    for e in errors:
        assert e["kind"] in kinds
        assert "text" in e and "id" in e
        if e["kind"] != "false_negative":
            assert 0.0 <= e["confidence"] <= 1.0


def test_report_mentions_the_numbers(run_dir):
    report = (run_dir / "report.md").read_text()
    m = json.loads((run_dir / "metrics.json").read_text())
    assert f"{m['test']['micro']['f1']:.4f}" in report
    assert "training_curves.png" in report
