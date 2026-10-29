"""The cross-task tolerance table reads the N = 1,000 ranked cells (rule 10)."""

import pandas as pd

from active_gliner.analysis import tolerance

PLACES = {("mit_movie", "en-US"): ("MIT Movie", "MITMovie")}


def _table(cells):
    return pd.DataFrame(
        [
            dict(dataset="mit_movie", locale="en-US", rule="ranked", share=s, budget=n, mean=m)
            for s, n, m in cells
        ]
    )


def test_rows_report_quality_loss_and_share_to_beat(tmp_path):
    table = _table(
        [
            (0, 1000, 78.0),
            (25, 1000, 82.0),
            (50, 1000, 84.9),
            (75, 1000, 86.6),
            (100, 1000, 87.2),
            (0, 400, 99.0),
        ]
    )
    rows = tolerance.rows(
        {"gemma-4-12b": table}, {"gemma-4-12b": {("mit_movie", "en-US"): 80.0}}, PLACES
    )
    row = rows.iloc[0]
    assert row.family == "NER" and row.share_to_beat == 25
    assert round(row.loss_quarter_llm, 1) == 0.6 and round(row.loss_all_llm, 1) == 9.2
    # 25% LLM loses 0.6, 50% loses 2.3 (over SAFE_LOSS), so the safe share is 25%.
    assert row.safe_llm_share == 25
    assert round(row.quality, 1) == round(100 * 80 / 87.2, 1)
    macros = tolerance.macros(rows)
    assert macros["MITMovieGemmaToleranceShare"] == 25
    assert macros["MITMovieGemmaToleranceSafeShare"] == 25
    text = tolerance.latex_table(rows, tmp_path / "t.tex").read_text()
    assert "NER & MIT Movie & Gemma & 92\\% & $0.6$ & $2.3$ & $5.2$ & $9.2$ & 25\\% & 25\\%" in text


def test_empty_table_writes_a_header_only(tmp_path):
    path = tolerance.latex_table(pd.DataFrame(), tmp_path / "t.tex")
    assert "\\bottomrule" in path.read_text()
