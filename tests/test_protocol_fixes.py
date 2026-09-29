"""Tests for the round-27 fixes: exact targets, frozen protocol, thresholds, ladder, stats."""

from copy import deepcopy

import pandas as pd
import pytest
import torch

from active_gliner import matrix, protocol
from active_gliner.analysis import cost, ladder, stats
from active_gliner.data.checks import altered_target_rate
from active_gliner.data.records import Record
from active_gliner.run import choose_eval_threshold
from active_gliner.tasks import ner, slots
from active_gliner.tasks._spans import ExactValue, exact_token_span, training_targets_match
from active_gliner.teachers.validate import span_decision

TEXT = "arthritis and rheumatoid arthritis"


def _ner_record():
    span = {"start": 0, "end": 9, "text": "arthritis", "label": "D"}
    return Record("x", None, TEXT, "ner", "en-US", "pool", {"spans": [span]})


def _splitter():
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    return WhitespaceTokenSplitter()


# 1. Exact training spans


def test_exact_value_maps_to_its_own_occurrence_only():
    first = ExactValue("arthritis", TEXT, 0, 9, "D")
    second = ExactValue("arthritis", TEXT, 25, 34, "D")
    assert exact_token_span(first, _splitter()) == (0, 0)
    assert exact_token_span(second, _splitter()) == (3, 3)


def test_exact_value_rejects_a_span_inside_a_word():
    with pytest.raises(ValueError):
        exact_token_span(ExactValue("arth", TEXT, 0, 4, "D"), _splitter())


def test_ner_training_example_matches_supplied_spans():
    record = _ner_record()
    example = ner.to_training_example(record, {"D": "disease"})
    assert training_targets_match(record, example)


def test_training_targets_match_rejects_plain_strings_and_moved_spans():
    record = _ner_record()
    example = ner.to_training_example(record, {"D": "disease"})
    plain = deepcopy(example)
    plain["output"]["entities"]["disease"] = ["arthritis"]
    assert not training_targets_match(record, plain)
    moved = deepcopy(example)
    moved["output"]["entities"]["disease"] = [ExactValue("arthritis", TEXT, 25, 34, "D")]
    assert not training_targets_match(record, moved)


@pytest.mark.parametrize("mode", ["structure", "entity"])
def test_slot_training_example_matches_supplied_spans(mode):
    record = Record(
        "s",
        None,
        TEXT,
        "slots",
        "en-US",
        "pool",
        {"spans": [{"start": 0, "end": 9, "text": "arthritis", "label": "D"}], "intent": "x"},
    )
    example = slots.to_training_example(record, {"D": "disease"}, slot_mode=mode)
    assert training_targets_match(record, example)


def test_altered_target_rate_counts_the_nested_repeat():
    report = altered_target_rate({"pool": [_ner_record()]})
    assert report["altered"] == 1 and report["total"] == 1


def test_altered_target_rate_ignores_a_sentence_with_one_occurrence():
    span = {"start": 0, "end": 5, "text": "Paris", "label": "L"}
    record = Record("y", None, "Paris is big", "ner", "en-US", "pool", {"spans": [span]})
    assert altered_target_rate({"pool": [record]})["altered"] == 0


@pytest.mark.model
@pytest.mark.parametrize(
    "task,kwargs",
    [(ner, {}), (slots, {"slot_mode": "structure"}), (slots, {"slot_mode": "entity"})],
)
def test_collator_marks_only_the_supplied_occurrence(student, task, kwargs):
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    from active_gliner.exact_targets import install_exact_targets

    install_exact_targets(student)
    processor = student.processor
    record = _ner_record()
    if task is slots:
        record = Record("s", None, TEXT, "slots", "en-US", "pool", {**record.gold, "intent": "x"})
    example = task.to_training_example(record, {"D": "disease"}, **kwargs)
    dataset = ExtractorDataset([deepcopy(example)], validate=True)
    batch = ExtractorCollator(processor, is_training=False, architecture="boundary")([dataset[0]])
    assert batch.structure_labels[0][0][1] == [[[(0, 0)]]]


@pytest.fixture(scope="module")
def student():
    from active_gliner import model

    return model.load_student(device="cpu", local_files_only=True)


# 2. Frozen acquisition


def test_inference_precision_is_restored():
    before = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    with protocol.inference_precision():
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    after = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    assert after == before


def _cfg(tmp_path, **update):
    base = dict(
        dataset="cleanconll",
        locale="en-US",
        selector="min",
        n=100,
        seed=1,
        labels="ground_truth",
        out_root=str(tmp_path),
    )
    return matrix.RunConfig(**{**base, **update})


def test_paired_arms_must_select_the_same_ids(tmp_path):
    gt = _cfg(tmp_path)
    teacher = _cfg(tmp_path, labels="gemma-4-12b")
    first = protocol.assert_selection(gt, "scores", ["a", "b"])
    assert protocol.assert_selection(teacher, "scores", ["a", "b"]) == first
    with pytest.raises(ValueError):
        protocol.assert_selection(teacher, "scores", ["a", "c"])


def test_other_seed_is_a_separate_selection(tmp_path):
    protocol.assert_selection(_cfg(tmp_path), "scores", ["a"])
    protocol.assert_selection(_cfg(tmp_path, seed=2), "scores", ["b"])


def test_digest_ignores_key_order():
    assert protocol.digest({"a": 1, "b": 2}) == protocol.digest({"b": 2, "a": 1})


# 3. Dev threshold


def test_threshold_picks_best_dev_f1():
    scores = {0.3: 0.70, 0.4: 0.72, 0.5: 0.71, 0.6: 0.74, 0.7: 0.60}
    assert choose_eval_threshold(scores) == 0.6


def test_threshold_tie_prefers_closest_to_half_then_lower():
    assert choose_eval_threshold({0.3: 0.8, 0.4: 0.8, 0.5: 0.7, 0.6: 0.8, 0.7: 0.8}) == 0.4
    assert choose_eval_threshold({0.3: 0.8, 0.4: 0.7, 0.5: 0.7, 0.6: 0.7, 0.7: 0.8}) == 0.3


def test_threshold_refuses_another_grid():
    with pytest.raises(ValueError):
        choose_eval_threshold({0.1: 0.9, 0.5: 0.8})


# 4. Recipes and resume


def test_required_recipe_fails_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(matrix, "REPO_ROOT", tmp_path)
    assert matrix.recipe("cleanconll") == {}
    with pytest.raises(ValueError, match="allow-default-recipe"):
        matrix.recipe("cleanconll", required=True)


def test_recipe_records_its_hash(monkeypatch, tmp_path):
    (tmp_path / "configs/recipes").mkdir(parents=True)
    (tmp_path / "configs/recipes/ner.yaml").write_text("task_lr: 0.0002\n")
    monkeypatch.setattr(matrix, "REPO_ROOT", tmp_path)
    values = matrix.recipe("cleanconll", required=True)
    assert values["task_lr"] == 0.0002 and len(values["recipe_hash"]) == 64


def test_resolve_validates_recipe_types_so_fingerprints_match(monkeypatch, tmp_path):
    # Bug found 26 Sep: an int lora_alpha from YAML made a finished run look stale.
    (tmp_path / "configs/recipes").mkdir(parents=True)
    (tmp_path / "configs/recipes/ner.yaml").write_text("lora_alpha: 64\nlora_r: 32\n")
    monkeypatch.setattr(matrix, "REPO_ROOT", tmp_path)
    resolved = matrix.resolve(_cfg(tmp_path), tmp_path)
    assert isinstance(resolved.lora_alpha, float)
    again = matrix.RunConfig.model_validate_json(resolved.model_dump_json())
    assert protocol.digest(resolved.model_dump()) == protocol.digest(again.model_dump())


def test_crossre_runs_train_every_step():
    # Decided 26 Sep: CrossRE dev F1 dips between checks, so early stopping is off.
    runs = [
        matrix.resolve(cfg, required=True) for cfg in matrix.build() if cfg.dataset == "crossre"
    ]
    assert runs
    for cfg in runs:
        assert cfg.early_stopping_patience >= cfg.max_steps // cfg.eval_steps


def test_run_without_fingerprint_is_stale(tmp_path):
    cfg = _cfg(tmp_path)
    assert matrix.run_status(cfg, tmp_path) == "pending"
    path = matrix.run_dir(cfg, tmp_path) / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert matrix.run_status(cfg, tmp_path) == "stale"


# 6. Ladder inclusion probabilities


def test_ladder_probability_is_documents_drawn_over_documents_in_stratum():
    items = [(f"s{i}", f"d{i // 2}", 0.1) for i in range(40)]
    rows = ladder.sample(items, target=10, n_strata=1, seed=3)
    documents = {row["doc_id"] for row in rows}
    assert len(documents) == 5  # ceil(10 / 1 / 2 sentences per document)
    assert all(row["inclusion_prob"] == 5 / 20 for row in rows)
    assert all(sum(r["doc_id"] == d for r in rows) == 2 for d in documents)


def test_ladder_small_dataset_takes_everything():
    items = [("a", None, 0.1), ("b", None, 0.9)]
    rows = ladder.sample(items, target=10, n_strata=2)
    assert {r["id"] for r in rows} == {"a", "b"}
    assert all(r["inclusion_prob"] == 1.0 for r in rows)


# 7. Primary completeness


def _primary_rows(datasets, seeds=range(1, 6)):
    return pd.DataFrame(
        [
            dict(dataset=d, seed=s, labels=label, selector=sel, f1=0.5)
            for d in datasets
            for s in seeds
            for label in (stats.TEACHER, "ground_truth")
            for sel in ("min", "random")
        ]
    )


def test_no_missing_cells_when_complete():
    assert stats.missing_primary_cells(_primary_rows(["a", "b"]), ["a", "b"]) == []


def test_missing_seed_and_dataset_are_listed():
    df = _primary_rows(["a"], seeds=range(1, 5))
    missing = stats.missing_primary_cells(df, ["a", "b"])
    assert len(missing) == 4 + 20
    assert dict(dataset="a", seed=5, labels="ground_truth", selector="min") in missing


def test_nan_score_counts_as_missing():
    df = _primary_rows(["a"])
    df.loc[0, "f1"] = float("nan")
    assert len(stats.missing_primary_cells(df, ["a"])) == 1


# 8. Overlap accounting


def test_span_decision_separates_duplicates_from_conflicts():
    kept = [{"start": 0, "end": 5, "label": "A"}]
    assert span_decision({"start": 0, "end": 5, "label": "A"}, kept) == "duplicate"
    assert span_decision({"start": 0, "end": 5, "label": "B"}, kept) == "overlap_conflict"
    assert span_decision({"start": 3, "end": 8, "label": "A"}, kept) == "overlap_conflict"
    assert span_decision({"start": 5, "end": 8, "label": "A"}, kept) == "accept"


# 9. GPU cost


def test_local_gpu_hours_use_elapsed_time_not_summed_latency(tmp_path):
    (tmp_path / "teacher_stats.json").write_text(
        '{"elapsed_s": 3600, "latency_s_total": 28800, "workers": 8, "cache_hits": 0}'
    )
    row = cost.run_cost({"run_dir": str(tmp_path), "labels": "gemma-4-12b"})
    assert row["teacher_gpu_h"] == pytest.approx(1.0)
    assert row["request_latency_s"] == 28800


# Round 28: tuning failures and fingerprint scope


def test_failed_trials_are_listed_and_block_freeze():
    from active_gliner import tune

    rows = [{"number": 0, "state": "COMPLETE"}, {"number": 1, "state": "FAIL"}]
    assert tune.failed_trials(rows) == [1]
    error = tune.trial_error(RuntimeError("CUDA out of memory"))
    assert error == "RuntimeError: CUDA out of memory"
    with pytest.raises(ValueError, match="failed trials"):
        tune.freeze("cleanconll", {"trials": rows, "best_params": {"task_lr": 1e-4}})


def test_trials_remaining_counts_every_finished_state():
    from active_gliner import tune

    states = ["COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING"]
    assert tune.trials_remaining(states, 20) == 17
    assert tune.trials_remaining(states, 2) == 0


def test_retried_failure_no_longer_blocks_freeze():
    from active_gliner import tune

    rows = [
        {"number": 0, "state": "COMPLETE"},
        {"number": 1, "state": "FAIL"},
        {"number": 2, "state": "FAIL"},
        {"number": 3, "state": "PRUNED", "retry_of": 1},
    ]
    assert tune.failed_trials(rows) == [2]
    assert tune.trials_to_retry(rows) == [2]


def test_failed_retry_is_not_retried_again():
    from active_gliner import tune

    rows = [{"number": 1, "state": "FAIL"}, {"number": 2, "state": "FAIL", "retry_of": 1}]
    assert tune.trials_to_retry(rows) == []
    assert tune.failed_trials(rows) == [1, 2]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tasks/ner.py", True),
        ("run.py", True),
        ("teachers/client.py", True),
        ("teachers/stats.py", False),
        ("analysis/stats.py", False),
        ("cli.py", False),
        ("report.py", False),
        ("observe/plots.py", False),
    ],
)
def test_fingerprint_covers_only_protocol_code(path, expected):
    assert protocol.is_protocol_source(path) is expected


# Exact numerics (Claude, 26 Sep)


def test_exact_numerics_off_keeps_gliner2_defaults():
    from active_gliner.train import exact_numerics_settings

    assert exact_numerics_settings(False) == {}


def test_exact_numerics_on_turns_off_reduced_precision():
    from gliner2.training.trainer import TrainingConfig

    from active_gliner.train import exact_numerics_settings

    settings = exact_numerics_settings(True)
    config = TrainingConfig(output_dir="unused", **settings)
    assert config.deterministic is True
    assert config.fp16 is False and config.bf16 is False
    assert config.allow_tf32 is False
    assert config.float32_matmul_precision == "highest"


def test_apply_exact_numerics_is_strict_not_warn_only():
    # warn_only let memory-efficient attention keep a non-deterministic backward.
    import torch

    from active_gliner.train import apply_exact_numerics

    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    try:
        apply_exact_numerics(False)
        assert torch.are_deterministic_algorithms_enabled() == saved[0]
        apply_exact_numerics(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[4:6]
        torch.set_float32_matmul_precision(saved[6])


def test_relation_training_uses_fp32_under_exact_numerics():
    import inspect

    from active_gliner.tasks import relation_training

    source = inspect.getsource(relation_training.train_relations)
    assert "not cfg.exact_numerics" in source


# Resume after a new config field (Claude, 26 Sep)


def _settings(**config):
    from active_gliner.run import RunConfig

    base = RunConfig(dataset="crossre", n=200, seed=0, **config).model_dump()
    return {"config": base, "dev_limit": None, "relation_heads": ["native"]}


def test_resume_accepts_a_later_field_at_its_default():
    from active_gliner import tune

    stored = _settings()
    del stored["config"]["exact_numerics"]
    assert tune.same_settings(stored, _settings())


def test_resume_rejects_a_later_field_that_is_not_default():
    from active_gliner import tune

    stored = _settings()
    del stored["config"]["exact_numerics"]
    assert not tune.same_settings(stored, _settings(exact_numerics=True))


def test_resume_rejects_a_changed_value():
    from active_gliner import tune

    assert not tune.same_settings(_settings(task_lr=1e-4), _settings(task_lr=2e-4))


# Round 34 (Claude): CrossRE head restriction and pruner warmup


def test_crossre_search_can_be_limited_to_the_native_head():
    import optuna

    from active_gliner import tune

    study = optuna.create_study(direction="maximize")
    for _ in range(5):
        params = tune.suggest_params(study.ask(), "crossre", ("native",))
        assert params["relation_head"] == "native"
        assert params["lora_targets"] == ["encoder", "relation_scorer"]
        assert params["augmentation"] is False


def test_open_study_uses_the_pruner_warmup(tmp_path):
    from active_gliner import tune

    study = tune.open_study(tmp_path, "crossre", 0, pruner_warmup_steps=300)
    assert study.pruner._n_warmup_steps == 300


@pytest.mark.parametrize(
    "kwargs", [{"relation_heads": ()}, {"relation_heads": ("bert",)}, {"pruner_warmup_steps": -1}]
)
def test_run_search_rejects_bad_search_options(tmp_path, kwargs):
    from active_gliner import tune

    with pytest.raises(ValueError):
        tune.run_search("crossre", out_root=tmp_path, **kwargs)


# Shared diversity embeddings (Claude, 26 Sep)


def test_diversity_embeddings_are_computed_once_and_shared(tmp_path, monkeypatch):
    # Bug found 26 Sep: per-run GPU embeddings let paired arms select different IDs.
    import numpy as np

    from active_gliner import model, run

    records = [
        Record(
            id=f"r{i}",
            doc_id=None,
            text=f"s {i}",
            task="ner",
            locale="en-US",
            source_split="train",
            gold={},
        )
        for i in range(4)
    ]
    calls = []

    def fake_embeddings(student, texts, batch_size):
        calls.append(len(texts))
        return np.random.default_rng(len(calls)).normal(size=(len(texts), 3))

    monkeypatch.setattr(model, "sentence_embeddings", fake_embeddings)
    cfg = _cfg(tmp_path, selector="diversity")
    first = run.pool_embeddings(None, cfg, records)
    second = run.pool_embeddings(None, cfg, records)
    assert calls == [4]
    assert np.array_equal(first, second)
    assert protocol.embedding_path(cfg, records).exists()
