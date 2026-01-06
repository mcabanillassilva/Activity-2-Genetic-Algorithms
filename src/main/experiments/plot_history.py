import csv
import matplotlib.pyplot as plt
from pathlib import Path


def plot_history(csv_file: Path, title: str):
    generations = []
    best = []

    with open(csv_file, "r") as f:
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
    plt.show()


if __name__ == "__main__":
    plot_history(
        Path("results/history_tournament_pox_swap.csv"),
        "Evolution of Best Makespan (Tournament + POX + Swap)",
    )
    
    
    
