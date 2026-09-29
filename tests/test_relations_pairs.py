"""Design section 15: CrossRE as relation prediction over given entity mentions."""

import pytest

from active_gliner import data, tasks
from active_gliner.data.records import Record
from active_gliner.tasks import relations


def _record():
    return Record(
        id="crossre:en-US:x",
        doc_id=None,
        text="Marie Curie worked in Paris .",
        task="relations",
        locale="en-US",
        source_split="train",
        gold={
            "entities": [
                {"start": 0, "end": 11, "text": "Marie Curie", "label": "person"},
                {"start": 22, "end": 27, "text": "Paris", "label": "location"},
            ],
            "relations": [
                {
                    "head": {"start": 0, "end": 11, "text": "Marie Curie"},
                    "tail": {"start": 22, "end": 27, "text": "Paris"},
                    "head_id": 0,
                    "tail_id": 1,
                    "type": "physical",
                }
            ],
        },
    )


LABELS = {"physical": "physical", "role": "role"}


def test_ordered_pairs_exclude_self():
    assert relations.pairs(_record()) == [(0, 1), (1, 0)]


def _pair_examples(positives, negatives):
    def example(labels):
        return {"output": {"classifications": [{"true_label": labels}]}}

    return [example(["physical"])] * positives + [example([]) for _ in range(negatives)]


def test_subsample_keeps_every_positive_and_ratio_negatives():
    examples = _pair_examples(3, 20)
    kept, counts = relations.subsample_negatives(examples, 2.0, seed=7)
    positives = [e for e in kept if e["output"]["classifications"][0]["true_label"]]
    assert len(positives) == 3
    assert len(kept) == 3 + 6
    assert counts["negative_pairs"] == 6 and counts["negative_pairs_available"] == 20


def test_subsample_is_seeded_and_keeps_order():
    examples = _pair_examples(2, 30)
    first, _ = relations.subsample_negatives(examples, 1.0, seed=1)
    again, _ = relations.subsample_negatives(examples, 1.0, seed=1)
    assert [id(e) for e in first] == [id(e) for e in again]
    positions = [examples.index(e) for e in first]
    assert positions == sorted(positions)


def test_subsample_caps_at_available_negatives():
    kept, counts = relations.subsample_negatives(_pair_examples(5, 2), 4.0, seed=0)
    assert len(kept) == 7 and counts["negative_pairs"] == 2


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_subsample_rejects_a_bad_ratio(ratio):
    with pytest.raises(ValueError):
        relations.subsample_negatives(_pair_examples(1, 1), ratio, seed=0)


def test_marked_text_keeps_full_sentence():
    marked = relations.marked_text(_record(), 0, 1)
    assert "[H:person] Marie Curie [/H]" in marked
    assert "[T:location] Paris [/T]" in marked
    assert "worked in" in marked


def test_training_examples_one_per_pair_with_negatives():
    examples = relations.to_training_example(_record(), LABELS)
    assert isinstance(examples, list) and len(examples) == 2
    trues = [ex["output"]["classifications"][0]["true_label"] for ex in examples]
    assert trues == [["physical"], []]


def test_gold_items_use_mention_ids():
    assert relations.gold_items(_record()) == {(0, 1, "physical")}


def test_score_excludes_true_negatives():
    pred = {
        "id": "crossre:en-US:x",
        "pairs": [],
        "relations": [{"head": 0, "tail": 1, "type": "physical", "probability": 0.9}],
        "confidence": 0.4,
    }
    m = relations.score([_record()], [pred], LABELS)
    assert m["micro"]["f1"] == 1.0


def test_loader_adds_mention_ids():
    if not data.raw_available("crossre"):
        pytest.skip("raw data not downloaded")
    splits = data.load("crossre")
    for r in splits["dev"][:200]:
        ents = r.gold["entities"]
        for rel in r.gold["relations"]:
            h, t = ents[rel["head_id"]], ents[rel["tail_id"]]
            assert (h["start"], h["end"]) == (rel["head"]["start"], rel["head"]["end"])
            assert (t["start"], t["end"]) == (rel["tail"]["start"], rel["tail"]["end"])
    assert tasks.for_dataset("crossre") is relations
