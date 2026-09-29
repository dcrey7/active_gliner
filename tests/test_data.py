"""Step 2 (O2): dataset loaders, frozen splits, and data checks.

Expected sizes come from docs/research/2026-09-25-1230-dataset-sources.md
and design section 14. Tests skip when the raw data is not downloaded.
"""

import pytest

from active_gliner import data

EXPECTED = {
    # (dataset, locale): (pool, dev, test)
    ("cleanconll", "en-US"): (13957, 3233, 3427),
    ("bc5cdr", "en-US"): (5228, 5330, 5865),
    ("mit_movie", "en-US"): (8797, 978, 2443),  # dev = seeded 10% of 9,775 train
    ("crossre", "en-US"): (2519, 300, 2446),  # pool = train + dev minus 300 holdout
    ("hallmarks", "en-US"): (12119, 1798, 3547),
    ("massive", "en-US"): (11514, 2033, 2974),
    ("massive", "fr-FR"): (11514, 2033, 2974),
}

TASK = {
    "cleanconll": "ner",
    "bc5cdr": "ner",
    "mit_movie": "ner",
    "crossre": "relations",
    "hallmarks": "classification",
    "massive": "slots",
}


def _load(name, locale):
    if not data.raw_available(name):
        pytest.skip(f"raw data for {name} not downloaded")
    return data.load(name, locale=locale)


@pytest.mark.parametrize("name,locale", list(EXPECTED))
def test_split_sizes(name, locale):
    splits = _load(name, locale)
    sizes = tuple(len(splits[s]) for s in ("pool", "dev", "test"))
    assert sizes == EXPECTED[(name, locale)]


@pytest.mark.parametrize("name,locale", list(EXPECTED))
def test_records_are_well_formed(name, locale):
    splits = _load(name, locale)
    ids = set()
    for records in splits.values():
        for r in records:
            assert r.task == TASK[name]
            assert r.text.strip()
            assert r.id not in ids, f"duplicate id {r.id}"
            ids.add(r.id)
            for span in data.gold_spans(r):
                assert 0 <= span["start"] < span["end"] <= len(r.text)
                assert r.text[span["start"] : span["end"]] == span["text"]


@pytest.mark.parametrize("name", ["cleanconll", "bc5cdr", "hallmarks"])
def test_no_document_crosses_splits(name):
    splits = _load(name, "en-US")
    docs = {s: {r.doc_id for r in recs} for s, recs in splits.items()}
    assert all(d is not None for ds in docs.values() for d in ds)
    assert not docs["pool"] & docs["dev"]
    assert not docs["pool"] & docs["test"]
    assert not docs["dev"] & docs["test"]


@pytest.mark.parametrize("name", ["mit_movie", "crossre"])
def test_carved_dev_split_is_deterministic(name):
    a = _load(name, "en-US")
    b = data.load(name, locale="en-US")
    assert [r.id for r in a["dev"]] == [r.id for r in b["dev"]]
    assert not {r.id for r in a["dev"]} & {r.id for r in a["pool"]}


def test_massive_locales_share_ids():
    en = _load("massive", "en-US")
    fr = _load("massive", "fr-FR")
    for split in ("pool", "dev", "test"):
        assert [r.id.split(":")[-1] for r in en[split]] == [r.id.split(":")[-1] for r in fr[split]]


def test_empty_gold_stays_in_pool():
    # Design rule: never filter the pool with ground truth.
    splits = _load("massive", "en-US")
    no_slot = sum(1 for r in splits["pool"] if not data.gold_spans(r))
    assert no_slot == 3755
    splits = _load("crossre", "en-US")
    assert any(not r.gold["relations"] for r in splits["pool"])


def test_hallmarks_labels():
    splits = _load("hallmarks", "en-US")
    labels = {lab for r in splits["pool"] for lab in r.gold["labels"]}
    assert len(labels) == 10
    assert "none" not in labels
    assert any(r.gold["labels"] == [] for r in splits["pool"])


def test_frozen_split_file_roundtrip(tmp_path):
    splits = _load("mit_movie", "en-US")
    path = data.write_split_ids(splits, "mit_movie", "en-US", out_dir=tmp_path)
    frozen = data.read_split_ids(path)
    assert frozen["dev"] == [r.id for r in splits["dev"]]
    assert frozen["hash"] == data.split_hash(splits)


def test_check_report_fields():
    if not data.raw_available("crossre"):
        pytest.skip("raw data for crossre not downloaded")
    rep = data.check("crossre", locale="en-US")
    for key in ["sizes", "label_counts", "empty_gold", "misaligned_dev_rate", "split_hash"]:
        assert key in rep
    assert "repeated_argument_rate" in rep
