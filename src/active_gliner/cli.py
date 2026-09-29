import argparse
import json
import os
import sys
from pathlib import Path

from active_gliner import __version__


def _load_env() -> None:
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)
    else:
        load_dotenv(path, override=False)


def main(argv: list[str] | None = None) -> int:
    _load_env()
    # cuBLAS reads this when it starts; deterministic runs need it, other runs ignore it.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(prog="active-gliner")
    parser.add_argument("--version", action="version", version=f"active-gliner {__version__}")
    commands = parser.add_subparsers(dest="command")
    data_parser = commands.add_parser("data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    check_parser = data_commands.add_parser("check")
    check_parser.add_argument("name")
    check_parser.add_argument("--locale", default="en-US")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument(
        "--selector", choices=["min", "random", "diversity", "all"], default="min"
    )
    run_parser.add_argument("--n", type=int, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--locale", default="en-US")
    run_parser.add_argument("--max-steps", type=int, default=1000)
    run_parser.add_argument("--eval-steps", type=int, default=100)
    run_parser.add_argument("--out-root", default="runs")
    run_parser.add_argument("--labels-source", default="ground_truth")
    run_parser.add_argument("--gt-fraction", type=float)
    run_parser.add_argument("--gt-assignment", choices=["routed", "random"])
    run_parser.add_argument("--no-prediction", choices=["zero", "last"], default="zero")
    matrix_parser = commands.add_parser("matrix")
    matrix_commands = matrix_parser.add_subparsers(dest="matrix_command", required=True)
    for name in ("list", "run"):
        sub = matrix_commands.add_parser(name)
        sub.add_argument("--block", action="append")
        sub.add_argument("--out-root", default="runs")
        if name == "run":
            sub.add_argument("--dataset", action="append")
            sub.add_argument("--labels", action="append")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--allow-default-recipe", action="store_true")
            sub.add_argument("--rerun-stale", action="store_true")
    tune_parser = commands.add_parser("tune")
    tune_parser.add_argument("--dataset", required=True)
    tune_parser.add_argument("--trials", type=int, default=20)
    tune_parser.add_argument("--max-steps", type=int, default=200)
    tune_parser.add_argument("--eval-steps", type=int, default=50)
    tune_parser.add_argument("--freeze", action="store_true")
    tune_parser.add_argument("--retry-failed", action="store_true")
    tune_parser.add_argument(
        "--relation-heads",
        default="classifier,native,marker",
        help="CrossRE only: comma-separated heads to search (research runs use native)",
    )
    tune_parser.add_argument("--pruner-warmup-steps", type=int, default=0)
    tune_parser.add_argument("--out-root", default="runs")
    label_parser = commands.add_parser("label")
    label_parser.add_argument("--dataset", required=True)
    label_parser.add_argument("--locale", default="en-US")
    label_parser.add_argument("--teacher", required=True)
    label_parser.add_argument("--split", choices=["pool", "dev", "test", "all"], default="pool")
    label_parser.add_argument("--limit", type=int)
    for name in ("ladder-sample", "ladder-label"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--locale", default="en-US")
        if name == "ladder-label":
            sub.add_argument("--teacher", required=True)
    zero_parser = commands.add_parser("zero-shot")
    zero_parser.add_argument("--models", help="comma-separated ladder names; default all")
    timing_parser = commands.add_parser("time-teacher")
    timing_parser.add_argument("--teacher", required=True)
    timing_parser.add_argument("--dataset", required=True)
    timing_parser.add_argument("--locale", default="en-US")
    eval_parser = commands.add_parser("teacher-eval")
    eval_parser.add_argument("--teacher", required=True)
    eval_parser.add_argument("--prompt", choices=["v1", "v2"], required=True)
    eval_parser.add_argument("--n", type=int, default=200)
    analyse_parser = commands.add_parser("analyse")
    analyse_parser.add_argument("--runs", default="runs")
    analyse_parser.add_argument("--out", default="paper")
    watch_parser = commands.add_parser("watch")
    watch_parser.add_argument("run_dir", type=Path)
    for name in ("fit", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--texts", required=True, type=Path)
        sub.add_argument(
            "--task", required=True, choices=["ner", "slots", "classification", "relations"]
        )
        sub.add_argument("--labels", required=True)
        if name == "predict":
            sub.add_argument("--adapter", required=True, type=Path)
            continue
        sub.add_argument("--teacher-url", required=True)
        sub.add_argument("--teacher-model", required=True)
        sub.add_argument("--n", type=int, default=200)
        sub.add_argument("--select", choices=["min", "random"], default="min")
        sub.add_argument("--seed", type=int, default=1)
        sub.add_argument("--dev", type=Path)
        sub.add_argument("--api-key-env")
        sub.add_argument("--out", dest="out_dir", default="active-gliner-out", type=Path)
        sub.add_argument("--max-steps", type=int, default=500)
        sub.add_argument("--eval-steps", type=int, default=50)
        sub.add_argument("--threshold", type=float, default=0.5)
        sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"fit", "predict"}:
        from active_gliner import byod

        values = vars(args).copy()
        values.pop("command")
        try:
            if args.command == "predict":
                for prediction in byod.predict(**values):
                    print(json.dumps(prediction, ensure_ascii=False))
            elif values.pop("dry_run"):
                print(
                    json.dumps(
                        byod.plan(
                            **{
                                key: values[key]
                                for key in ("texts", "task", "labels", "n", "select", "seed")
                            }
                        ),
                        indent=2,
                    )
                )
            else:
                print(byod.fit(**values))
        except ValueError as exc:
            parser.error(str(exc))
        return 0

    if args.command in {"ladder-sample", "ladder-label"}:
        from active_gliner.analysis import ladder

        if args.command == "ladder-sample":
            print(ladder.sample_dataset(args.dataset, args.locale))
        else:
            print(json.dumps(ladder.label_sample(args.dataset, args.teacher, args.locale)))
        return 0
    if args.command == "zero-shot":
        from active_gliner import zero_shot

        names = args.models.split(",") if args.models else None
        if names and set(names) - set(zero_shot.LADDER):
            parser.error(f"Unknown ladder model; choose from {', '.join(zero_shot.LADDER)}")
        for row in zero_shot.evaluate_all(names):
            print(json.dumps(row))
        return 0
    if args.command == "time-teacher":
        from active_gliner import timing

        print(timing.time_teacher(args.teacher, args.dataset, args.locale))
        return 0
    if args.command == "analyse":
        from active_gliner.analysis.report import analyse

        result = analyse(args.runs, args.out)
        print(f"Analysed {len(result['runs'])} finished runs; results in {args.out}")
        for reason in result["skipped"]:
            print(f"Skipped: {reason}")
        return 0
    if args.command == "matrix":
        from collections import Counter

        from active_gliner import matrix

        runs = matrix.select(
            matrix.build(),
            blocks=args.block,
            datasets=getattr(args, "dataset", None),
            labels_sources=getattr(args, "labels", None),
        )
        if args.matrix_command == "list":
            counts = Counter(r.block for r in runs)
            statuses = Counter((r.block, matrix.run_status(r, args.out_root)) for r in runs)
            for block, count in counts.items():
                details = ", ".join(
                    f"{statuses[block, status]} {status}" for status in ("pending", "done", "stale")
                )
                print(f"{block}: {count} runs, {details}")
            print(f"total: {sum(counts.values())} runs")
        else:
            if args.limit is not None and args.limit < 0:
                parser.error("--limit must be nonnegative")
            matrix.execute(
                runs,
                args.out_root,
                args.limit,
                allow_default_recipe=args.allow_default_recipe,
                rerun_stale=args.rerun_stale,
            )
        return 0
    if args.command == "tune":
        from active_gliner import tune

        result = tune.run_search(
            args.dataset,
            n_trials=args.trials,
            retry_failed=args.retry_failed,
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            out_root=args.out_root,
            relation_heads=tuple(args.relation_heads.split(",")),
            pruner_warmup_steps=args.pruner_warmup_steps,
        )
        failed = tune.failed_trials(result["trials"])
        if failed:
            print(json.dumps(result, indent=2))
            print(f"Tuning failed; cannot freeze. Failed trials: {failed}", file=sys.stderr)
            return 1
        if args.freeze:
            print(tune.freeze(args.dataset, result))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "teacher-eval":
        from active_gliner.teachers.evaluate import evaluate

        if args.n <= 0:
            parser.error("--n must be positive")
        evaluate(args.teacher, args.prompt, args.n)
        return 0
    if args.command == "label":
        from active_gliner import data, tasks
        from active_gliner.teachers import TeacherConfig, label_records, prompts
        from active_gliner.teachers.config import REPO_ROOT

        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be positive")
        cfg = TeacherConfig.from_yaml(REPO_ROOT / "configs/teachers" / f"{args.teacher}.yaml")
        splits = data.load(args.dataset, args.locale)
        labels = tasks.labels(args.dataset, splits)
        prompt_version = prompts.version_for(args.dataset)
        definitions = prompts.prompt_for(args.dataset, prompt_version)
        selected_splits = ("pool", "dev", "test") if args.split == "all" else (args.split,)
        failed = False
        for split in selected_splits:
            records = splits[split][: args.limit]
            if not records:
                parser.error(f"The selected split is empty: {split}")
            result = label_records(
                records,
                records[0].task,
                labels,
                cfg,
                REPO_ROOT / "labels",
                dataset=args.dataset,
                definitions=definitions,
                prompt_version=prompt_version,
            )
            print(json.dumps({"split": split, **result.stats}), flush=True)
            failed |= bool(result.stats["errors_by_kind"].get("request_failed", 0))
        return 3 if failed else 0
    if args.command == "run":
        from active_gliner.run import RunConfig, run_experiment

        values = vars(args).copy()
        values.pop("command")
        print(run_experiment(RunConfig(**values)))
        return 0
    if args.command == "watch":
        from active_gliner.observe.logs import watch

        watch(args.run_dir)
        return 0
    if args.command == "data":
        from active_gliner import data

        report = data.check(args.name, args.locale)
        data.write_split_ids(data.load(args.name, args.locale), args.name, args.locale)
        path = Path("data/reports") / f"{args.name}-{args.locale}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        path.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0
    parser.print_help()
    return 0
