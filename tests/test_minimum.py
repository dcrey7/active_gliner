import numpy as np
import pandas as pd

from active_gliner.analysis import minimum

POOLS = {("cleanconll", "en-US"): 1000}


def _row(labels, f1, dev_f1, seed=1, gt_fraction=None, gt_assignment=None, **extra):
    row = dict(
        dataset="cleanconll",
        locale="en-US",
        labels=labels,
        selector="all",
        variant="long",
        seed=seed,
        f1=f1,
        dev_f1=dev_f1,
        gt_fraction=gt_fraction,
        gt_assignment=gt_assignment,
        run_dir=f"{labels}-{gt_fraction}-{seed}",
    )
    row.update(extra)
    return row


def test_minimum_rows_count_human_sentences_and_keep_both_ends():
    df = pd.DataFrame(
        [
            _row("gemma-4-12b", 75.0, 0.75),
            _row("gemma-4-12b", 78.0, 0.78, gt_fraction=0.1, gt_assignment="nested"),
            _row("ground_truth", 90.0, 0.90),
            # Left out: fixed-N mixing, short whole pool, other teachers.
            _row("gemma-4-12b", 70.0, 0.70, gt_fraction=0.5, gt_assignment="random"),
            _row("gemma-4-12b", 74.0, 0.74, variant=None),
            _row("gemma-4-e4b", 60.0, 0.60),
        ]
    )
    rows = minimum.minimum_rows(df, POOLS)
    assert sorted(zip(rows.human, rows.f1, strict=True)) == [(0, 75.0), (100, 78.0), (1000, 90.0)]
    macros = minimum.macros(minimum.summary(rows), {})
    assert macros == {
        "CleanCoNLLMinimumAtZero": 75.0,
        "CleanCoNLLMinimumAtHundred": 78.0,
        "CleanCoNLLMinimumAtWholePool": 90.0,
    }


def test_paired_gap_is_zero_for_identical_counts():
    counts = np.array([[1, 0, 0], [0, 1, 1], [2, 0, 1]])
    result = minimum.paired_gap([counts, counts], counts, n_boot=50)
    assert result["gap"] == 0.0 and result["low"] == 0.0 and result["high"] == 0.0


def test_choose_takes_smallest_dev_crossing_and_reads_test_only_there(monkeypatch):
    good = np.array([[1, 0, 0]] * 40)
    bad = np.array([[0, 1, 1]] * 40)
    counts = {"small": bad, "middle": good, "large": good}
    rows = pd.DataFrame(
        [
            dict(dataset="cleanconll", locale="en-US", human=h, run_dir=name)
            for h, name in ((0, "zero"), (100, "small"), (400, "middle"), (800, "large"))
        ]
        + [dict(dataset="cleanconll", locale="en-US", human=1000, run_dir="human")]
    )
    counts["zero"] = bad
    ids = [str(i) for i in range(40)]
    monkeypatch.setattr(minimum, "test_ids", lambda run_dir, split: ids)
    monkeypatch.setattr(minimum, "sentence_counts", lambda run_dir, split: counts[run_dir])
    reads = []

    def teacher(dataset, locale, split, sentence_ids):
        reads.append(split)
        return np.array([[1, 1, 0]] * 40)

    chosen = minimum.choose(rows, teacher, n_boot=50)
    result = chosen[("cleanconll", "en-US")]
    assert result["human"] == 400
    assert result["share"] == 0.4
    assert reads.count("test") == 1
    assert result["test"]["p_holm"] == result["test"]["p_value"]
