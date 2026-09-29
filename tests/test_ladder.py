"""Design sections 11 and 16: teacher ladder sample and teacher configs."""

from collections import Counter

from active_gliner import matrix
from active_gliner.analysis import ladder
from active_gliner.teachers import TeacherConfig


def test_ladder_b_uses_gemma_e4b_not_qwen_2b():
    labels = Counter(r.labels_source for r in matrix.build() if r.block == "teacher_ladder_b")
    assert labels == {"gemma-4-e4b": 6, "gpt-oss-120b": 6, "qwen-3.8-27b": 3}
    assert all(r.labels_source != "qwen-3.8-2b" for r in matrix.build())


def test_teacher_configs_exist():
    for name in ("gemma-4-12b", "gemma-4-e4b", "qwen-3.8-27b", "gpt-oss-120b"):
        cfg = TeacherConfig.from_yaml(f"configs/teachers/{name}.yaml")
        assert cfg.name == name
    e4b = TeacherConfig.from_yaml("configs/teachers/gemma-4-e4b.yaml")
    assert e4b.pins["model_sha256"].startswith("676c3507")
    api = TeacherConfig.from_yaml("configs/teachers/qwen-3.8-27b.yaml")
    assert api.api_key_env == "CEREBRAS_API_KEY"
    assert api.base_url.startswith("https://api.cerebras.ai")


def _items(n_docs=100, per_doc=4):
    # (id, doc_id, confidence)
    out = []
    for d in range(n_docs):
        for s in range(per_doc):
            out.append((f"s{d}-{s}", f"d{d}", (d % 10) / 10 + s / 100))
    return out


def test_ladder_sample_is_document_grouped_stratified_and_seeded():
    items = _items()
    a = ladder.sample(items, target=100, n_strata=4, seed=1)
    b = ladder.sample(items, target=100, n_strata=4, seed=1)
    assert a == b
    ids = {row["id"] for row in a}
    # document-grouped: a sampled document brings all its sentences
    docs = {row["doc_id"] for row in a}
    for d in docs:
        assert all(f"{d.replace('d', 's')}-{s}" in ids for s in range(4))
    # each stratum stops at the first whole document that reaches 25 sentences (7 x 4 = 28)
    assert 100 <= len(ids) <= 112
    # every stratum is represented and inclusion probabilities are recorded
    assert len({row["stratum"] for row in a}) == 4
    assert all(0 < row["inclusion_prob"] <= 1 for row in a)


def test_ladder_sample_takes_everything_when_small():
    items = _items(n_docs=10)
    rows = ladder.sample(items, target=1000, n_strata=4, seed=1)
    assert len(rows) == 40
    assert all(row["inclusion_prob"] == 1.0 for row in rows)


def test_weighted_f1_counts_rare_strata_more():
    # Sentence 1 stands for 1 sentence, sentence 2 for 3 sentences.
    counts = [[1, 0, 0], [0, 1, 1]]
    assert ladder.weighted_f1(counts, [1.0, 1.0]) == 0.5
    # tp 1, fp 3, fn 3: F1 = 2 / (2 + 6)
    assert ladder.weighted_f1(counts, [1.0, 1 / 3]) == 0.25


def test_gap_interval_resamples_whole_documents_and_is_seeded():
    rows = [
        dict(id=f"s{i}", doc_id=f"d{i // 2}", stratum=i % 2, inclusion_prob=0.5) for i in range(8)
    ]
    better = [[1, 0, 0]] * 8
    worse = [[1, 1, 0]] * 4 + [[1, 0, 0]] * 4
    first = ladder.gap_interval(rows, better, worse, n_boot=200, seed=3)
    assert first == ladder.gap_interval(rows, better, worse, n_boot=200, seed=3)
    assert 0 <= first["low"] <= first["high"]
    same = ladder.gap_interval(rows, better, better, n_boot=50)
    assert same["low"] == same["high"] == 0.0
