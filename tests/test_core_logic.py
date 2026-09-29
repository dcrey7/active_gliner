"""Step 3a: scoring, confidence rules, overlap decoding and selection.

Pure logic, no model. Every expected number is worked out by hand.
Rules come from design section 2 and refactor plan section 2.
"""

import pytest

from active_gliner import confidence, decode, selection
from active_gliner.evaluate import metrics

# ---------- metrics ----------


def test_span_prf_strict_micro_macro_per_label():
    # items: (start, end, label); last element is the label
    gold = [[(0, 5, "PER"), (10, 15, "LOC")], [(0, 3, "ORG")]]
    pred = [[(0, 5, "PER"), (10, 14, "LOC")], [(0, 3, "ORG"), (5, 8, "PER")]]
    m = metrics.span_prf(gold, pred)
    # tp = 2 (PER, ORG), fp = 2 (LOC wrong boundary, extra PER), fn = 1 (LOC)
    assert (m["micro"]["tp"], m["micro"]["fp"], m["micro"]["fn"]) == (2, 2, 1)
    assert m["micro"]["precision"] == pytest.approx(0.5)
    assert m["micro"]["recall"] == pytest.approx(2 / 3)
    assert m["micro"]["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    # per label: PER p=1/2 r=1 f=2/3; LOC 0; ORG 1
    assert m["per_label"]["PER"]["f1"] == pytest.approx(2 / 3)
    assert m["per_label"]["LOC"]["f1"] == 0.0
    assert m["per_label"]["ORG"]["f1"] == 1.0
    assert m["per_label"]["LOC"]["support"] == 1
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)


def test_span_prf_duplicates_count_once():
    m = metrics.span_prf([[(0, 5, "PER")]], [[(0, 5, "PER"), (0, 5, "PER")]])
    assert (m["micro"]["tp"], m["micro"]["fp"]) == (1, 0)


def test_span_prf_empty_is_zero_not_error():
    m = metrics.span_prf([[]], [[]])
    assert m["micro"]["f1"] == 0.0
    assert m["macro_f1"] == 0.0


def test_relation_triples_use_direction():
    gold = [[(0, 4, 10, 15, "role")]]
    pred = [[(10, 15, 0, 4, "role")]]  # reversed direction is wrong
    assert metrics.span_prf(gold, pred)["micro"]["tp"] == 0


def test_classification_prf_counts_negatives_correctly():
    labels = ["a", "b", "c"]
    gold = [{"a"}, set(), {"b", "c"}]
    pred = [{"a", "b"}, set(), {"b"}]
    m = metrics.classification_prf(gold, pred, labels)
    # decisions: tp a, b(rec3) = 2; fp b(rec1) = 1; fn c(rec3) = 1
    assert (m["micro"]["tp"], m["micro"]["fp"], m["micro"]["fn"]) == (2, 1, 1)
    assert m["micro"]["f1"] == pytest.approx(2 / 3)
    # per label f1: a 1.0, b 2/3 (tp1 fp1), c 0.0
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3 + 0) / 3)


def test_canonical_slot_value_and_record_accuracy():
    assert metrics.canonical_value("  Taylor   Swift ") == "taylor swift"
    gold = [{("artist_name", "taylor swift")}, set()]
    pred = [{("artist_name", "taylor swift")}, {("time", "7 am")}]
    assert metrics.record_accuracy(gold, pred) == 0.5


# ---------- overlap decoding ----------


def test_greedy_flat_keeps_highest_and_compatible_spans():
    spans = [
        {"start": 0, "end": 10, "label": "A", "confidence": 0.6},
        {"start": 0, "end": 4, "label": "B", "confidence": 0.5},
        {"start": 6, "end": 10, "label": "C", "confidence": 0.5},
    ]
    # greedy by score keeps A only: B and C both overlap A
    kept = decode.greedy_flat(spans)
    assert [(s["start"], s["label"]) for s in kept] == [(0, "A")]


def test_greedy_flat_two_labels_on_one_span():
    spans = [
        {"start": 0, "end": 12, "label": "person", "confidence": 0.57},
        {"start": 0, "end": 12, "label": "miscellaneous", "confidence": 0.66},
    ]
    assert [s["label"] for s in decode.greedy_flat(spans)] == ["miscellaneous"]


def test_greedy_flat_ties_are_deterministic():
    spans = [
        {"start": 3, "end": 6, "label": "b", "confidence": 0.5},
        {"start": 3, "end": 6, "label": "a", "confidence": 0.5},
    ]
    # tie: earlier start, then shorter end, then label name
    assert decode.greedy_flat(spans)[0]["label"] == "a"


def test_greedy_flat_returns_sorted_by_start():
    spans = [
        {"start": 8, "end": 9, "label": "x", "confidence": 0.9},
        {"start": 0, "end": 2, "label": "x", "confidence": 0.7},
    ]
    assert [s["start"] for s in decode.greedy_flat(spans)] == [0, 8]


# ---------- sentence confidence (design section 2) ----------


def test_min_rule_and_no_prediction_is_zero():
    assert confidence.min_confidence([0.9, 0.4, 0.7]) == 0.4
    assert confidence.min_confidence([]) == 0.0  # NER and relations: no prediction = 0


def test_classification_nearest_boundary():
    probs = {"a": 0.9, "b": 0.45, "c": 0.02}
    # min |p - t| with t = 0.5: b gives 0.05
    assert confidence.classification_confidence(probs, threshold=0.5) == pytest.approx(0.05)


def test_slot_rule_non_empty_empty_and_no_candidate():
    assert confidence.slot_confidence([0.8, 0.6], max_below=0.3) == 0.6
    assert confidence.slot_confidence([], max_below=0.45) == pytest.approx(0.55)
    assert confidence.slot_confidence([], max_below=None) == 1.0


# ---------- selection ----------


def test_min_selection_ascending_with_seeded_ties():
    ids = ["a", "b", "c", "d"]
    conf = [0.9, 0.1, 0.5, 0.1]
    picked = selection.select(ids, conf, n=3, strategy="min", seed=1)
    assert set(picked[:2]) == {"b", "d"}
    assert picked[2] == "c"
    again = selection.select(ids, conf, n=3, strategy="min", seed=1)
    assert picked == again


def test_min_selection_tie_order_depends_on_seed():
    ids = [f"id{i}" for i in range(50)]
    conf = [0.3] * 50
    orders = {tuple(selection.select(ids, conf, n=10, strategy="min", seed=s)) for s in range(5)}
    assert len(orders) > 1


def test_random_selection_is_seeded_and_ignores_confidence():
    ids = [f"id{i}" for i in range(100)]
    a = selection.select(ids, [0.0] * 100, n=10, strategy="random", seed=3)
    b = selection.select(ids, [1.0] * 100, n=10, strategy="random", seed=3)
    assert a == b
    assert len(set(a)) == 10


def test_selection_rejects_bad_input():
    with pytest.raises(ValueError):
        selection.select(["a"], [0.1], n=2, strategy="min", seed=0)
    with pytest.raises(ValueError):
        selection.select(["a"], [0.1], n=1, strategy="nope", seed=0)


def test_thesis_scores_kept():
    # thesis scores over a sentence's entity confidences
    c = [0.9, 0.5]
    assert selection.score_min(c) == 0.5
    assert selection.score_avg(c) == pytest.approx(0.7)
    assert selection.score_mse(c) == pytest.approx(((0.1) ** 2 + (0.5) ** 2) / 2)
