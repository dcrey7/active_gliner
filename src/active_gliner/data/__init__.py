from . import bc5cdr, cleanconll, crossre, hallmarks, massive, mit_movie
from .common import raw_dir
from .records import Record, gold_spans
from .splits import make_splits, read_split_ids, split_hash, write_split_ids

LOADERS = {
    module.__name__.rsplit(".", 1)[-1]: module
    for module in (bc5cdr, cleanconll, crossre, hallmarks, massive, mit_movie)
}
__all__ = [
    "Record",
    "gold_spans",
    "load",
    "raw_available",
    "check",
    "dataset_pin",
    "read_split_ids",
    "split_hash",
    "write_split_ids",
]


def load(name: str, locale: str = "en-US") -> dict[str, list[Record]]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    if locale not in (("en-US", "fr-FR") if name == "massive" else ("en-US",)):
        raise ValueError(f"Unsupported locale for {name}: {locale}")
    return make_splits(LOADERS[name].load(raw_dir(name), locale), name)


def dataset_pin(name: str) -> dict:
    """Identify the exact raw data a run read: sha256 of every loader file."""
    import hashlib
    import json

    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    root = raw_dir(name)
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(LOADERS[name].files(root))
    }
    source = root / "source.json"
    return {
        "raw_files_sha256": files,
        "source": json.loads(source.read_text()) if source.exists() else None,
    }


def raw_available(name: str) -> bool:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    try:
        return all(path.is_file() for path in LOADERS[name].files(raw_dir(name)))
    except FileNotFoundError:
        return False


def check(name: str, locale: str = "en-US") -> dict:
    from .checks import check as check_data

    return check_data(name, locale)
