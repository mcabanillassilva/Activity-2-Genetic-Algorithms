import csv
from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(history_csv: Path, out_png: Path, title: str) -> None:
    generations = []
    best = []

    with open(history_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["generation"]))
            best.append(int(row["best_fitness"]))

    plt.figure(figsize=(8, 5))
    plt.plot(generations, best)
    plt.xlabel("Generation")
    plt.ylabel("Best makespan")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all_from_summary(results_dir: Path, dataset_name: str) -> None:
    summary_path = results_dir / f"summary_{dataset_name}.csv"

    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    best_row = min(rows, key=lambda r: int(r["best_fitness"]))

    for row in rows:
        tag = f'{row["selection"]}_{row["crossover"]}_{row["mutation"]}'
        history_csv = results_dir / f"history_{tag}.csv"
        out_png = results_dir / f"plot_{tag}.png"

        plot_history(
            history_csv,
            out_png,
            title=f"{dataset_name} — {row['selection']} + {row['crossover']} + {row['mutation']}",
        )

    # plot best solution
    best_tag = f'{best_row["selection"]}_{best_row["crossover"]}_{best_row["mutation"]}'
    best_history_csv = results_dir / f"history_{best_tag}.csv"
    best_png = results_dir / "best_plot.png"

    plot_history(
        best_history_csv,
        best_png,
        title=f"{dataset_name} — BEST ({best_tag})",
    )

    print(f"Plots generated for dataset {dataset_name}")
    print(f"Best configuration: {best_tag}, makespan={best_row['best_fitness']}")


if __name__ == "__main__":
    dataset_name = "abz9"
    results_dir = Path("results") / dataset_name

    plot_all_from_summary(results_dir, dataset_name)
