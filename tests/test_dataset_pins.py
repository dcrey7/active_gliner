"""Research rule 7: every run pins the exact raw data files it read."""

import pytest

from active_gliner import data

DATASETS = ["bc5cdr", "cleanconll", "crossre", "hallmarks", "massive", "mit_movie"]


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_pin_hashes_every_loader_file(name):
    if not data.raw_available(name):
        pytest.skip(f"raw data for {name} is not on this machine")
    pin = data.dataset_pin(name)
    files = pin["raw_files_sha256"]
    assert files, "a dataset pin must list at least one raw file"
    assert all(len(digest) == 64 for digest in files.values())
    assert pin == data.dataset_pin(name), "the pin must be stable across calls"


def test_dataset_pin_rejects_an_unknown_dataset():
    with pytest.raises(ValueError):
        data.dataset_pin("conll2003")
