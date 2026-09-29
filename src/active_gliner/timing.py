"""Measure teacher labelling time on a free GPU, for the annotation-cost figures.

The cached labels cannot give this: most were made while training shared the GPU,
so their request times are too long. This job labels a fixed pool slice again, with
an empty cache, and records the wall time of the whole job.
"""

import json
import tempfile
import time
from pathlib import Path

from active_gliner import data, tasks
from active_gliner.teachers import prompts
from active_gliner.teachers.client import label_records
from active_gliner.teachers.config import REPO_ROOT, TeacherConfig

# The first N pool records in frozen order: the same slice for every teacher.
SENTENCES = 200


def timing_path(teacher: str, dataset: str, locale: str, out_root="runs") -> Path:
    return Path(out_root) / "timing" / f"teacher-{teacher}-{dataset}-{locale}.json"


def time_teacher(
    teacher: str, dataset: str, locale: str = "en-US", out_root="runs", transport=None
) -> Path:
    cfg = TeacherConfig.from_yaml(REPO_ROOT / "configs/teachers" / f"{teacher}.yaml")
    splits = data.load(dataset, locale)
    labels = tasks.labels(dataset, splits)
    records = splits["pool"][:SENTENCES]
    version = prompts.version_for(dataset)
    with tempfile.TemporaryDirectory() as empty_cache:
        started = time.perf_counter()
        result = label_records(
            records,
            records[0].task,
            labels,
            cfg,
            empty_cache,
            transport=transport,
            dataset=dataset,
            definitions=prompts.prompt_for(dataset, version),
            prompt_version=version,
        )
        elapsed = time.perf_counter() - started
    stats = result.stats
    if stats["cache_hits"]:
        raise ValueError("Timing must not read cached labels")
    path = timing_path(teacher, dataset, locale, out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "teacher": teacher,
                "dataset": dataset,
                "locale": locale,
                "sentences": len(records),
                "workers": cfg.workers,
                "elapsed_s": elapsed,
                "seconds_per_sentence": elapsed / len(records),
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "valid_rate": stats["valid_rate"],
                "timing_protocol": "free GPU, empty cache, whole-job wall time",
                "date": time.strftime("%Y-%m-%d %H:%M %Z"),
            },
            indent=2,
        )
        + "\n"
    )
    return path


def seconds_per_sentence(teacher: str, dataset: str, locale: str, out_root="runs") -> float | None:
    path = timing_path(teacher, dataset, locale, out_root)
    if not path.exists():
        return None
    return json.loads(path.read_text())["seconds_per_sentence"]
