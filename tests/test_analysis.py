"""Step 7: analysis. Pure functions on synthetic inputs, with hand-checked answers.

Rules: design section 4 and direction doc 3e (primary interaction, Holm families,
paired bootstrap, equal weight per dataset), 3f (costs), hero figures.
"""

import numpy as np
import pandas as pd
import pytest

from active_gliner.analysis import figures, numbers, stats, teacher


def test_holm_matches_hand_computation():
    # sorted p: 0.01 x4 = 0.04; 0.02 x3 = 0.06; 0.03 x2 = 0.06; 0.04 x1 -> 0.06 (monotone)
    adj = stats.holm([0.03, 0.01, 0.04, 0.02])
    assert adj == pytest.approx([0.06, 0.04, 0.06, 0.06])


def _cells():
    rows = []
    # two datasets, 3 seeds; min-random gap: gemma +2, gold +5 -> interaction -3 per seed
    for ds in ("a", "b"):
        for seed in (1, 2, 3):
            base = 80 + seed
            rows += [
                dict(dataset=ds, seed=seed, labels="ground_truth", selector="random", f1=base),
                dict(dataset=ds, seed=seed, labels="ground_truth", selector="min", f1=base + 5),
                dict(dataset=ds, seed=seed, labels="gemma-4-12b", selector="random", f1=base - 10),
                dict(dataset=ds, seed=seed, labels="gemma-4-12b", selector="min", f1=base - 8),
            ]
    return pd.DataFrame(rows)


def test_interaction_per_seed_and_pooled():
    res = stats.interaction(_cells(), teacher="gemma-4-12b")
    assert res["per_seed"] == pytest.approx([-3.0, -3.0, -3.0])
    assert res["mean"] == pytest.approx(-3.0)
    assert res["practical_mean"] == pytest.approx(2.0)  # (min - random) with the teacher


def test_equal_weight_per_dataset():
    df = _cells()
    # make dataset b have a bigger gold gap (+9): interaction b = -7, a = -3 -> pooled -5
    df.loc[(df.dataset == "b") & (df.labels == "ground_truth") & (df.selector == "min"), "f1"] += 4
    res = stats.interaction(df, teacher="gemma-4-12b")
    assert res["mean"] == pytest.approx(-5.0)
    assert res["per_dataset"] == pytest.approx({"a": -3.0, "b": -7.0})


def test_paired_bootstrap_micro_f1_difference():
    rng = np.random.default_rng(0)
    n = 400
    # per-sentence (tp, fp, fn) counts for two systems on the same test sentences
    a = np.stack([rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 2, n)], axis=1)
    b = a.copy()
    b[:, 0] += 1  # b has one more true positive per sentence
    res = stats.bootstrap_diff(b, a, n_boot=500, seed=1)
    assert res["diff"] > 0
    assert res["low"] > 0
    assert res["p_value"] < 0.01
    same = stats.bootstrap_diff(a, a, n_boot=200, seed=1)
    assert same["diff"] == 0 and same["low"] <= 0 <= same["high"]


def test_micro_f1_from_counts():
    counts = np.array([[2, 1, 0], [1, 0, 1]])  # tp 3, fp 1, fn 1
    assert stats.micro_f1(counts) == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))


def test_pareto_front():
    pts = pd.DataFrame({"name": ["a", "b", "c", "d"], "cost": [1, 2, 3, 4], "f1": [50, 70, 60, 80]})
    assert figures.pareto_front(pts, x="cost", y="f1") == ["a", "b", "d"]


def test_teacher_error_by_confidence_bin():
    # Q2: teacher F1 per student-confidence quartile, disjoint bins
    conf = np.linspace(0, 1, 8)
    counts = np.array([[0, 1, 1]] * 4 + [[1, 0, 0]] * 4)  # low half wrong, high half right
    table = teacher.f1_by_bin(conf, counts, n_bins=4)
    assert list(table["bin"]) == [0, 1, 2, 3]
    assert list(table["n"]) == [2, 2, 2, 2]
    assert table["f1"].iloc[0] == 0.0 and table["f1"].iloc[3] == 1.0


def test_error_overlap_between_teachers():
    a = {"s1", "s2", "s3"}
    b = {"s2", "s3", "s4"}
    assert teacher.jaccard(a, b) == pytest.approx(2 / 4)


def test_numbers_tex_macros(tmp_path):
    out = tmp_path / "numbers.tex"
    numbers.write_macros(out, {"PrimaryNER": -3.14159, "PoolCleanCoNLL": 13957})
    text = out.read_text()
    assert "\\newcommand{\\PrimaryNER}{\\ensuremath{-}3.14}" in text
    assert "\\newcommand{\\PoolCleanCoNLL}{13,957}" in text
    with pytest.raises(ValueError):
        numbers.write_macros(out, {"bad_name1": 1})  # LaTeX macro names are letters only


def test_zero_shot_rows_are_not_treated_as_a_teacher():
    # Zero-shot reference runs carry labels "none"; the report must not look up teacher labels.
    from active_gliner.analysis.report import _teacher_tables

    df = pd.DataFrame(
        [{"dataset": "cleanconll", "locale": "en-US", "labels": "none", "run_dir": "/missing"}]
    )
    skipped = []
    tables, bins = _teacher_tables(df, skipped)
    assert skipped == [] and bins == {}
    assert tables["scores"] == {}
