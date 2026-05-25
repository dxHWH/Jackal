import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = [
    "eval/test_win_rate",
    "eval_window/mean_win_rate",
    "train_stats/battle_won_mean",
]


def read_scalars(run_dir):
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    rows = []
    for tag in TAGS:
        if tag not in available:
            continue
        for item in acc.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": int(item.step),
                    "t_env_m": float(item.step) / 1_000_000.0,
                    "value": float(item.value),
                    "wall_time": float(item.wall_time),
                }
            )
    rows.sort(key=lambda row: (row["tag"], row["step"], row["wall_time"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot current Spindle EDT-QMIX win-rate comparison.")
    parser.add_argument(
        "--tb-root",
        default="artifacts/tensorboard/marl2/by_map/spindle",
        help="TensorBoard root containing Spindle experiment folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/plots/marl2/current_spindle_edtrnn_gated_vs_nomap",
        help="Output directory for plot, CSV, and summary.",
    )
    args = parser.parse_args()

    tb_root = Path(args.tb_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "EDT gated map": tb_root
        / "jackal_autoaim_5v5_etdqmix_hetero_spindle_mapfeat7_gated_limit1600_dt004_10m_v3",
        "EDT no map": tb_root
        / "jackal_autoaim_5v5_etdqmix_hetero_spindle_nomap_limit1600_dt004_10m_v3",
    }

    series = {}
    all_rows = []
    for label, run_dir in runs.items():
        rows = read_scalars(run_dir)
        series[label] = rows
        for row in rows:
            all_rows.append({"run": label, **row})

    csv_path = out_dir / "win_rate_rows.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "t_env_m", "value", "wall_time"])
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"EDT gated map": "#1f77b4", "EDT no map": "#d62728"}
    for label, rows in series.items():
        for tag, ax, style, alpha in [
            ("eval/test_win_rate", axes[0], "-", 0.95),
            ("eval_window/mean_win_rate", axes[0], "--", 0.85),
            ("train_stats/battle_won_mean", axes[1], "-", 0.9),
        ]:
            pts = [row for row in rows if row["tag"] == tag]
            if not pts:
                continue
            xs = [row["t_env_m"] for row in pts]
            ys = [row["value"] for row in pts]
            name = label if tag != "eval_window/mean_win_rate" else f"{label} window"
            ax.plot(xs, ys, style, label=name, color=colors[label], alpha=alpha, linewidth=2)

    axes[0].set_title("Spindle EDT-QMIX Current Experiments: Evaluation Win Rate")
    axes[0].set_ylabel("Eval win rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_title("Training Recent Battle-Won Mean")
    axes[1].set_ylabel("Train win rate")
    axes[1].set_xlabel("t_env (million steps)")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()

    plot_path = out_dir / "win_rate_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Spindle EDT-QMIX current experiment win-rate comparison\n")
        f.write("Generated with tensorboard EventAccumulator inside conda env: pymarl2_env\n\n")
        for label, rows in series.items():
            f.write(f"[{label}]\n")
            for tag in TAGS:
                pts = [row for row in rows if row["tag"] == tag]
                if not pts:
                    f.write(f"  {tag}: no data\n")
                    continue
                last = pts[-1]
                best = max(pts, key=lambda row: row["value"])
                f.write(
                    f"  {tag}: n={len(pts)} last={last['value']:.4f}@{last['t_env_m']:.3f}M "
                    f"max={best['value']:.4f}@{best['t_env_m']:.3f}M\n"
                )
            f.write("\n")

    print(plot_path)
    print(csv_path)
    print(summary_path)
    for label, rows in series.items():
        pts = [row for row in rows if row["tag"] == "eval/test_win_rate"]
        if pts:
            print(
                label,
                "last_eval_win",
                f"{pts[-1]['value']:.4f}",
                "at",
                f"{pts[-1]['t_env_m']:.3f}M",
                "n",
                len(pts),
            )


if __name__ == "__main__":
    main()
