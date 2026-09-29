"""Step 6: the run matrix, the resumable runner, label mixing, diversity and tuning.

Counts come from design sections 5, 11, 12, 13 and 15 (309 main runs).
"""

from collections import Counter

import numpy as np
import pytest

from active_gliner import matrix, mixing, selection


def test_matrix_has_309_unique_runs():
    runs = matrix.build()
    assert len(runs) == 309
    names = [matrix.run_name(r) for r in runs]
    assert len(set(names)) == 309


def test_matrix_blocks_match_the_design():
    blocks = Counter(r.block for r in matrix.build())
    assert blocks == {
        "ner_core": 108,
        "ner_primary_extra_seeds": 24,
        "ner_routing": 18,
        "ner_qwen": 12,
        "ner_no_prediction_sensitivity": 6,
        "other_tasks": 72,
        "other_tasks_primary_extra_seeds": 24,
        "references": 12,
        "teacher_ladder_b": 15,
        "mixing": 6,
        "french": 12,
    }


def test_primary_cells_have_five_seeds():
    runs = matrix.build()
    cell = [
        r
        for r in runs
        if r.dataset == "cleanconll"
        and r.selector == "min"
        and r.n == 400
        and r.labels_source == "ground_truth"
        and r.gt_fraction is None
        and r.no_prediction == "zero"
    ]
    assert sorted(r.seed for r in cell) == [1, 2, 3, 4, 5]


def test_references_use_the_full_pool():
    refs = [r for r in matrix.build() if r.block == "references"]
    assert all(r.selector == "all" and r.seed == 1 for r in refs)
    assert Counter(r.labels_source for r in refs) == {"ground_truth": 6, "gemma-4-12b": 6}


def test_runner_skips_finished_runs(tmp_path):
    runs = matrix.build()[:3]
    done = matrix.run_dir(runs[0], out_root=tmp_path)
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    todo = matrix.pending(runs, out_root=tmp_path)
    assert runs[0] not in todo
    assert len(todo) == 2


def test_runner_filters_by_block():
    runs = matrix.build()
    only = matrix.select(runs, blocks=["mixing"])
    assert len(only) == 6


# ---------- ground-truth mixing (routing and the thesis mixing grid) ----------


def test_routed_gives_gold_to_the_most_uncertain():
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    conf = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.2, "e": 0.8, "f": 0.7, "g": 0.6, "h": 0.3}
    gold = mixing.gold_ids(ids, conf, fraction=0.25, assignment="routed", seed=1)
    assert gold == {"b", "d"}


def test_random_assignment_is_seeded_and_sized():
    ids = [f"id{i}" for i in range(400)]
    conf = {i: 0.5 for i in ids}
    a = mixing.gold_ids(ids, conf, fraction=0.25, assignment="random", seed=1)
    b = mixing.gold_ids(ids, conf, fraction=0.25, assignment="random", seed=1)
    assert a == b and len(a) == 100


# ---------- diversity (k-means on sentence embeddings) ----------


def test_diversity_picks_one_per_cluster_and_is_seeded():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    points = np.vstack([c + rng.normal(0, 0.3, size=(25, 2)) for c in centers])
    ids = [f"id{i}" for i in range(100)]
    picked = selection.select_diverse(ids, points, n=4, seed=3)
    clusters = {int(i[2:]) // 25 for i in picked}
    assert clusters == {0, 1, 2, 3}
    assert picked == selection.select_diverse(ids, points, n=4, seed=3)


def test_diversity_is_identical_across_processes():
    # Bug found 26 Sep: multi-threaded k-means picked different sentences in different
    # processes, so paired arms failed the selection check. Paired arms run in separate
    # processes, so the check must be across processes.
    import subprocess
    import sys

    code = (
        "import numpy as np; from active_gliner import selection;"
        "v = np.random.default_rng(1).normal(size=(3000, 64)).astype('float32');"
        "print('|'.join(selection.select_diverse([str(i) for i in range(3000)], v, 100, 3001)))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(3)
    }
    assert len(outputs) == 1


def test_diversity_runs_k_means_on_one_thread(monkeypatch):
    import threadpoolctl

    limits = []
    real = threadpoolctl.threadpool_limits

    def spy(*args, **kwargs):
        limits.append(kwargs.get("limits", args[0] if args else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", spy)
    points = np.random.default_rng(0).normal(size=(20, 2))
    selection.select_diverse([str(i) for i in range(20)], points, n=3, seed=1)
    assert limits == [1]


# ---------- no-prediction sensitivity ----------


def test_no_prediction_rank_last_rule():
    ids = ["empty", "low", "high"]
    conf = [0.0, 0.2, 0.9]
    has_pred = [False, True, True]
    zero = selection.select(ids, conf, n=2, strategy="min", seed=1)
    last = selection.select(
        ids, selection.rank_empty_last(conf, has_pred), n=2, strategy="min", seed=1
    )
    assert zero[0] == "empty"
    assert last == ["low", "high"]


# ---------- tuning ----------


@pytest.mark.model
def test_tuning_smoke(tmp_path):
    import torch

    from active_gliner import data, tune

    if not torch.cuda.is_available() or not data.raw_available("mit_movie"):
        pytest.skip("needs GPU and data")
    study = tune.run_search(
        dataset="mit_movie",
        n_trials=2,
        train_n=32,
        max_steps=10,
        eval_steps=5,
        dev_limit=32,
        out_root=tmp_path,
    )
    assert len(study["trials"]) == 2
    assert study["trials"][0]["vendor_inspired"] is True
    assert (tmp_path / "tuning" / "mit_movie" / "trials.jsonl").exists()
    assert "best_params" in study


def test_every_research_run_uses_exact_numerics():
    assert all(cfg.exact_numerics for cfg in matrix.build())
