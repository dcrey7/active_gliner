"""Teacher error kinds: exact, wrong edges, wrong type, missed, invented, empty alarms."""

from active_gliner.analysis import teacher_errors
from active_gliner.data.records import Record


def _record(id, spans):
    return Record(id, None, "play a nirvana playlist", "slots", "en-US", "test", {"spans": spans})


def _span(start, end, label):
    return {"start": start, "end": end, "label": label}


def test_span_errors_sort_each_mistake():
    records = [
        _record("a", [_span(7, 23, "playlist_name")]),
        _record("b", []),
        _record("c", [_span(0, 4, "x")]),
    ]
    labels = {
        # wrong edges (same type, overlap) plus one span that overlaps the gold one too
        "a": {"spans": [_span(16, 23, "playlist_name"), _span(7, 14, "artist_name")]},
        # an empty sentence given a slot: invented and an empty alarm
        "b": {"spans": [_span(0, 4, "x")]},
        # exact match
        "c": {"spans": [_span(0, 4, "x")]},
    }
    found = teacher_errors.span_errors(records, labels)
    assert found["gold"] == 2
    assert found["exact"] == 50 and found["edges"] == 50 and found["missed"] == 0
    assert found["invented"] == 50  # one invented span against two ground-truth spans
    assert found["empty_alarm"] == 100


def test_span_errors_count_wrong_edges_and_type_as_both():
    records = [_record("a", [_span(7, 23, "playlist_name")]), _record("b", [_span(0, 4, "x")])]
    labels = {"a": {"spans": [_span(16, 23, "artist_name")]}, "b": {"spans": [_span(0, 4, "x")]}}
    found = teacher_errors.span_errors(records, labels)
    assert found["edges_type"] == 50 and found["edges"] == 0 and found["type"] == 0


def test_latex_table_has_a_both_column(tmp_path):
    import pandas as pd

    row = dict(title="T", teacher="Gemma", exact=1, edges=2, type=3, edges_type=4)
    row.update(missed=5, invented=6, empty_alarm=7)
    path = teacher_errors.latex_table(pd.DataFrame([row]), tmp_path / "t.tex")
    text = path.read_text()
    assert "& Both &" in text and "T & Gemma & 1 & 2 & 3 & 4 & 5 & 6 & 7" in text


def test_scope_split_keeps_out_of_scope_apart():
    rows = [
        {"label": "alarm", "tp": "9", "support": "10", "recall": "0.9"},
        {"label": "greeting", "tp": "7", "support": "10", "recall": "0.7"},
        {"label": "oos", "tp": "1", "support": "4", "recall": "0.25"},
    ]
    intents, oos = teacher_errors.scope_split(rows)
    assert intents == 80 and oos == 25


def test_label_errors_count_none_sentences_given_a_label():
    records = [
        Record("a", None, "t", "classification", "en-US", "test", {"labels": []}),
        Record("b", None, "t", "classification", "en-US", "test", {"labels": ["x"]}),
    ]
    labels = {"a": {"labels": ["x"]}, "b": {"labels": ["x"]}}
    found = teacher_errors.label_errors(records, labels)
    assert found["precision"] == 50 and found["recall"] == 100 and found["empty_alarm"] == 100
