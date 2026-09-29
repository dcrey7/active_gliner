"""Plots from saved logs, using actual optimizer steps."""

from pathlib import Path

from .logs import read_jsonl


def plot_training(run_dir: Path, best_step: int, total_steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = read_jsonl(run_dir / "train_log.jsonl")
    evals = read_jsonl(run_dir / "eval_log.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    steps = [row["step"] for row in train]
    losses = [r["loss"] for r in train]
    window = max(1, len(steps) // 20)
    smoothed = []
    running_loss = 0.0
    for index, loss in enumerate(losses):
        running_loss += loss
        if index >= window:
            running_loss -= losses[index - window]
        smoothed.append(running_loss / min(index + 1, window))
    axes[0].plot(steps, losses, color="tab:blue", linewidth=0.7, alpha=0.3, label="Train")
    axes[0].plot(steps, smoothed, color="tab:blue", linewidth=2, label="Train (smoothed)")
    axes[0].plot([r["step"] for r in evals], [r["eval_loss"] for r in evals], "o-", label="Dev")
    f1 = axes[0].twinx()
    f1.plot([r["step"] for r in evals], [r["dev_f1"] for r in evals], color="green", label="Dev F1")
    f1.set(ylabel="Dev F1", ylim=(0, 1))
    best = next(r for r in evals if r["step"] == best_step)
    f1.plot(best_step, best["dev_f1"], "g*", markersize=15, label="Best dev F1")
    axes[0].set(xlabel="Optimizer step", ylabel="Loss", title="Training progress")
    handles, labels = axes[0].get_legend_handles_labels()
    f1_handles, f1_labels = f1.get_legend_handles_labels()
    axes[0].legend(handles + f1_handles, labels + f1_labels)
    axes[1].plot(steps, [r["learning_rate"] for r in train])
    axes[1].set(xlabel="Optimizer step", ylabel="Learning rate", title="LR schedule")
    minutes = [r["time_s"] / 60 for r in train]
    gpu_gb = [r["gpu_gb"] for r in train]
    axes[2].plot(minutes, gpu_gb, color="purple", label="GPU GB")
    axes[2].set(
        xlabel="Time (minutes)",
        ylabel="GPU GB",
        title="Resource use",
        ylim=(0, max(1.0, max(gpu_gb, default=0) * 1.2)),
    )
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    cpu = axes[2].twinx()
    cpu.plot(minutes, [r["cpu_percent"] for r in train], color="orange", label="CPU %")
    cpu.set_ylabel("CPU %")
    handles, labels = axes[2].get_legend_handles_labels()
    cpu_handles, cpu_labels = cpu.get_legend_handles_labels()
    axes[2].legend(handles + cpu_handles, labels + cpu_labels)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "plots/training_curves.png", dpi=180)
    plt.close(fig)
    (run_dir / "plots/training_summary.txt").write_text(
        f"Total optimizer steps: {total_steps}\nBest step: {best_step}\n", encoding="utf-8"
    )


def plot_calibration(run_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    populated = [row for row in rows if row["count"]]
    ax.bar(
        [r["mean_confidence"] for r in populated], [r["correctness"] for r in populated], width=0.18
    )
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical correctness")
    fig.tight_layout()
    fig.savefig(run_dir / "plots/calibration.png", dpi=180)
    plt.close(fig)
