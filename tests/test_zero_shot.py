"""Zero-shot reference ladder (design section 13): the lower bound for every task."""

import pytest

from active_gliner import model, zero_shot
from active_gliner.cli import main


def test_ladder_has_four_models_and_28_pairs():
    assert list(zero_shot.LADDER) == [
        "gliner2.5-small-v1",
        "gliner2.5-base-v1",
        "gliner2.5-multi-v1",
        "gliner2-large-v1",
    ]
    assert len(zero_shot.LADDER) * len(zero_shot.PAIRS) == 28


def test_multi_rung_is_the_pinned_student():
    assert zero_shot.LADDER["gliner2.5-multi-v1"] == (model.STUDENT_REPO, model.STUDENT_SHA)


def test_every_rung_is_pinned_to_a_full_sha():
    for repo, sha in zero_shot.LADDER.values():
        assert repo.startswith("fastino/")
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_finished_pair_is_skipped_without_loading_a_model(tmp_path, monkeypatch):
    directory = zero_shot.run_dir("gliner2.5-small-v1", "cleanconll", "en-US", tmp_path)
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text("{}")

    def fail(*args, **kwargs):
        raise AssertionError("a finished pair must not load a model")

    monkeypatch.setattr(model, "load_student", fail)
    assert zero_shot.evaluate("gliner2.5-small-v1", "cleanconll", "en-US", tmp_path) == directory


def test_unsupported_pair_is_skipped_without_loading_a_model(tmp_path, monkeypatch):
    # GLiNER2 large has no native relation scorer, so its CrossRE pair is marked unsupported.
    directory = zero_shot.run_dir("gliner2-large-v1", "crossre", "en-US", tmp_path)
    directory.mkdir(parents=True)
    (directory / zero_shot.UNSUPPORTED).write_text("no native relation scorer\n")

    def fail(*args, **kwargs):
        raise AssertionError("an unsupported pair must not load a model")

    monkeypatch.setattr(model, "load_student", fail)
    assert zero_shot.evaluate("gliner2-large-v1", "crossre", "en-US", tmp_path) is None


def test_cli_rejects_an_unknown_ladder_model():
    with pytest.raises(SystemExit):
        main(["zero-shot", "--models", "gliner-nonexistent"])
