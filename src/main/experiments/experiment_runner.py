import csv
from pathlib import Path

from src.main.utils.jssp_instance import load_orlib_jobshop
from src.main.ga.ga_core import run_ga, GAConfig
from src.main.ga.selection import tournament_selection, roulette_selection
from src.main.ga.crossover import pox_crossover, two_point_crossover
from src.main.ga.mutation import swap_mutation, insertion_mutation


from src.main.ga.ga_core import GAConfig


def config_for(filename: str) -> GAConfig:
    if filename == "ft06.txt":
        return GAConfig(
            population_size=60,
            generations=300,
            crossover_rate=0.9,
            mutation_rate=0.2,
            elitism_size=1,
            seed=42,
            patience=80,
        )
    if filename == "abz5.txt":
        return GAConfig(
            population_size=120,
            generations=600,
            crossover_rate=0.9,
            mutation_rate=0.25,
            elitism_size=2,
            seed=42,
            patience=150,
        )
    if filename == "abz9.txt":
        return GAConfig(
            population_size=150,
            generations=800,
            crossover_rate=0.9,
            mutation_rate=0.30,
            elitism_size=2,
            seed=42,
            patience=200,
        )

    raise ValueError("Unknown dataset")


def run_experiments():

    filename = "ft06.txt"
    instance = load_orlib_jobshop(f"datasets/{filename}")

    output_dir = Path("results") / filename.replace(".txt", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"summary_{filename.replace('.txt', '')}.csv"

    experiments = [
        (
            "tournament",
            "pox",
            "swap",
            tournament_selection,
            pox_crossover,
            swap_mutation,
        ),
        (
            "tournament",
            "pox",
            "insertion",
            tournament_selection,
            pox_crossover,
            insertion_mutation,
        ),
        (
            "tournament",
            "two_point",
            "swap",
            tournament_selection,
            two_point_crossover,
            swap_mutation,
        ),
        ("roulette", "pox", "swap", roulette_selection, pox_crossover, swap_mutation),
        (
            "roulette",
            "pox",
            "insertion",
            roulette_selection,
            pox_crossover,
            insertion_mutation,
        ),
        (
            "roulette",
            "two_point",
            "insertion",
            roulette_selection,
            two_point_crossover,
            insertion_mutation,
        ),
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["selection", "crossover", "mutation", "best_fitness", "stopped_generation"]
        )

        for sel_name, cx_name, mut_name, sel, cx, mut in experiments:
            print(f"Running {sel_name} + {cx_name} + {mut_name}")

            config = config_for(filename)

            result = run_ga(
                instance=instance,
                selection=sel,
                crossover=cx,
                mutation=mut,
                config=config,
            )

            writer.writerow(
                [
                    sel_name,
                    cx_name,
                    mut_name,
                    result.best_fitness,
                    result.stopped_generation,
                ]
            )

            hist_file = output_dir / f"history_{sel_name}_{cx_name}_{mut_name}.csv"
            with open(hist_file, "w", newline="") as hf:
                hw = csv.writer(hf)
                hw.writerow(["generation", "best_fitness"])
                for i, val in enumerate(result.history_best):
                    hw.writerow([i, val])


if __name__ == "__main__":
    run_experiments()
