from pathlib import Path
import csv

from src.main.utils.jssp_instance import load_orlib_jobshop
from src.main.sa.simulated_annealing import simulated_annealing, SAConfig
from src.main.ga.mutation import swap_mutation, insertion_mutation


def run_sa_experiments():
    instance = load_orlib_jobshop("datasets/abz9.txt")

    output_dir = Path("results") / "abz9" / "sa"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_sa_abz9.csv"

    experiments = [
        ("swap", swap_mutation, 1000.0, 0.995),
        ("swap", swap_mutation, 1500.0, 0.995),
        ("insertion", insertion_mutation, 1000.0, 0.995),
        ("insertion", insertion_mutation, 1500.0, 0.995),
        ("swap", swap_mutation, 1000.0, 0.990),
        ("insertion", insertion_mutation, 1000.0, 0.990),
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mutation",
                "initial_temperature",
                "cooling_rate",
                "best_fitness",
                "iterations",
            ]
        )

        for mut_name, mut_fn, temp, cooling in experiments:
            print(f"Running SA: mutation={mut_name}, T0={temp}, cooling={cooling}")

            config = SAConfig(
                initial_temperature=temp,
                cooling_rate=cooling,
                min_temperature=0.1,
                max_iterations=50000,
                seed=42,
            )

            result = simulated_annealing(
                instance=instance,
                mutation=mut_fn,
                config=config,
            )

            writer.writerow(
                [
                    mut_name,
                    temp,
                    cooling,
                    result.best_fitness,
                    len(result.history_best),
                ]
            )

    print("SA experiments completed.")
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    run_sa_experiments()
