"""Second-teacher numbers use only the seeds that every label source shares (rule 10)."""

import pandas as pd
import pytest

from active_gliner.analysis import second_teacher


def _row(labels, selector, seed, f1, **kw):
    return {
        "dataset": "cleanconll",
        "locale": "en-US",
        "labels": labels,
        "selector": selector,
        "n": 400,
        "seed": seed,
        "f1": f1,
        "gt_fraction": None,
        "no_prediction": "zero",
        **kw,
    }


def _df():
    rows = []
    for seed in (1, 2):
        rows += [
            _row("ground_truth", "min", seed, 80.0),
            _row("ground_truth", "random", seed, 70.0),
            _row("gemma-4-12b", "min", seed, 60.0),
            _row("gemma-4-12b", "random", seed, 62.0),
            _row("gpt-6-luna", "min", seed, 64.0 + seed),
            _row("gpt-6-luna", "random", seed, 63.0),
        ]
    # Seed 3 has no Luna run, a mixing run and an off-budget run: all ignored.
    rows += [
        _row("ground_truth", "min", 3, 0.0),
        _row("ground_truth", "random", 3, 0.0),
        _row("gemma-4-12b", "min", 3, 0.0),
        _row("gemma-4-12b", "random", 3, 0.0),
        _row("gemma-4-12b", "min", 1, 99.0, gt_fraction=0.25),
        _row("gpt-6-luna", "min", 1, 99.0, n=100),
    ]
    return pd.DataFrame(rows)


TEACHERS = {
    "cleanconll/en-US": {
        "error_overlap": {"gemma-4-12b": {"gpt-6-luna": 0.7}, "gpt-6-luna": {"gemma-4-12b": 0.7}}
    },
    "scores": {
        "cleanconll/en-US/gemma-4-12b": {"f1": 76.0},
        "cleanconll/en-US/gpt-6-luna": {"f1": 79.0},
    },
}


def test_cells_keep_shared_seeds_at_the_main_budget():
    table = second_teacher.cells(_df())
    row = table.iloc[0]
    assert row.seeds == 2
    assert row.LunaMin == pytest.approx(65.5)
    assert row.LunaGap == pytest.approx(2.5)
    assert row.GemmaGap == pytest.approx(-2.0)
    assert row.GroundTruthGap == pytest.approx(10.0)


def test_macros_report_teacher_student_and_overlap():
    macros = second_teacher.macros(second_teacher.cells(_df()), TEACHERS)
    assert macros["TeacherCleanCoNLLLuna"] == 79.0
    assert macros["LunaTeacherMinusGemma"] == pytest.approx(3.0)
    assert macros["LunaStudentMinusGemma"] == pytest.approx(5.5)
    assert macros["CleanCoNLLFamilyOverlap"] == 0.7
    assert macros["LunaInteractionMeanLunaSeeds"] == pytest.approx(-7.5)
    assert macros["GemmaGapPositiveLunaSeeds"] == 0
    assert macros["LunaSeeds"] == 2


def test_latex_table_uses_minus_signs(tmp_path):
    path = second_teacher.latex_table(second_teacher.cells(_df()), TEACHERS, tmp_path / "t.tex")
    text = path.read_text()
    assert "CleanCoNLL & 76.0 & 79.0 & 0.70 & 60.0 & 65.5 & $-$2.0 & 2.5 & 10.0" in text
