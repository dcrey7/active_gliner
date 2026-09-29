"""Step 3b: the four task plug-ins against the real student model.

Marked `model`: loads gliner2.5-multi-v1 from the pinned snapshot.
Uses 16 real dev records per dataset.
"""

import pytest
import torch

from active_gliner import data, model, tasks

DATASETS = [
    ("cleanconll", "en-US"),
    ("bc5cdr", "en-US"),
    ("mit_movie", "en-US"),
    ("crossre", "en-US"),
    ("hallmarks", "en-US"),
    ("massive", "en-US"),
    ("massive", "fr-FR"),
]

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def student():
    try:
        model.snapshot_path(model.STUDENT_REPO, model.STUDENT_SHA, local_files_only=True)
    except Exception:
        pytest.skip("student snapshot not cached")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.load_student(device=device, local_files_only=True)


def _dev(name, locale, n=16):
    if not data.raw_available(name):
        pytest.skip(f"raw data for {name} not downloaded")
    splits = data.load(name, locale=locale)
    return splits["dev"][:n], splits


@pytest.mark.parametrize("name,locale", DATASETS)
def test_labels_are_natural_and_map_back(name, locale):
    _, splits = _dev(name, locale)
    assert tasks.for_dataset(name) is not None
    labels = tasks.labels(name, splits)
    assert labels, "no labels"
    for raw, natural in labels.items():
        assert natural == natural.strip() and natural
        assert tasks.to_raw_label(labels, natural) == raw


@pytest.mark.parametrize("name,locale", DATASETS)
def test_training_examples_pass_gliner2_validation(name, locale):
    from gliner2.training.data import TrainingDataset

    records, splits = _dev(name, locale, n=40)
    task = tasks.for_dataset(name)
    labels = tasks.labels(name, splits)
    examples = []
    for r in records:
        ex = task.to_training_example(r, labels)
        examples.extend(ex if isinstance(ex, list) else [ex])  # relations: one per pair
    for ex in examples:
        assert set(ex) == {"input", "output"}
    TrainingDataset(examples).validate()


@pytest.mark.parametrize("name,locale", DATASETS)
def test_predict_shapes_offsets_and_confidence(student, name, locale):
    records, splits = _dev(name, locale)
    task = tasks.for_dataset(name)
    labels = tasks.labels(name, splits)
    preds = task.predict(student, records, labels, threshold=0.5, batch_size=8)
    assert len(preds) == len(records)
    for r, p in zip(records, preds, strict=True):
        assert p["id"] == r.id
        assert 0.0 <= p["confidence"] <= 1.0
        for span in tasks.pred_spans(p):
            assert r.text[span["start"] : span["end"]] == span["text"]
            assert 0.0 <= span["confidence"] <= 1.0


@pytest.mark.parametrize("name", ["cleanconll", "bc5cdr", "mit_movie"])
def test_ner_predictions_do_not_overlap(student, name):
    records, splits = _dev(name, "en-US")
    task = tasks.for_dataset(name)
    preds = task.predict(student, records, tasks.labels(name, splits), threshold=0.3)
    for p in preds:
        spans = sorted(p["spans"], key=lambda s: s["start"])
        for a, b in zip(spans, spans[1:], strict=False):
            assert a["end"] <= b["start"]


def test_classification_scores_every_label(student):
    records, splits = _dev("hallmarks", "en-US")
    task = tasks.for_dataset("hallmarks")
    labels = tasks.labels("hallmarks", splits)
    preds = task.predict(student, records, labels, threshold=0.5)
    for p in preds:
        assert set(p["probabilities"]) == set(labels)
        # decision = labels at or above threshold; empty set allowed
        assert set(p["labels"]) == {k for k, v in p["probabilities"].items() if v >= 0.5}


def test_slot_candidates_are_consistent(student):
    records, splits = _dev("massive", "en-US", n=32)
    task = tasks.for_dataset("massive")
    labels = tasks.labels("massive", splits)
    preds = task.predict(student, records, labels, threshold=0.5)
    for p in preds:
        if p["spans"]:
            assert p["confidence"] == pytest.approx(min(s["confidence"] for s in p["spans"]))
        else:
            assert p["max_below"] is None or p["max_below"] < 0.5
            expected = 1.0 if p["max_below"] is None else 1.0 - p["max_below"]
            assert p["confidence"] == pytest.approx(expected)


def test_relations_score_every_ordered_pair(student):
    # Design section 15: given mentions, one multi-label decision per ordered pair.
    records, splits = _dev("crossre", "en-US")
    task = tasks.for_dataset("crossre")
    labels = tasks.labels("crossre", splits)
    preds = task.predict(student, records, labels, threshold=0.5)
    for r, p in zip(records, preds, strict=True):
        n = len(r.gold["entities"])
        assert len(p["pairs"]) == n * (n - 1)
        for pair in p["pairs"]:
            assert set(pair["probabilities"]) == set(labels)
            assert pair["head"] != pair["tail"]
        if n < 2:
            assert p["confidence"] == 0.5
        else:
            expected = min(abs(v - 0.5) for pr in p["pairs"] for v in pr["probabilities"].values())
            assert p["confidence"] == pytest.approx(expected)
        for rel in p["relations"]:
            assert rel["type"] in labels
            assert rel["probability"] >= 0.5


@pytest.mark.parametrize("name,locale", DATASETS)
def test_score_zero_shot_runs(student, name, locale):
    records, splits = _dev(name, locale, n=32)
    task = tasks.for_dataset(name)
    labels = tasks.labels(name, splits)
    preds = task.predict(student, records, labels, threshold=0.5)
    m = task.score(records, preds, labels)
    assert 0.0 <= m["micro"]["f1"] <= 1.0
    assert "per_label" in m and "macro_f1" in m
