import random

from src.main.utils.jssp_instance import load_orlib_jobshop
from src.main.ga.ga_core import run_ga, GAConfig
from src.main.ga.selection import tournament_selection, roulette_selection
from src.main.ga.crossover import pox_crossover, two_point_crossover
from src.main.ga.mutation import swap_mutation, insertion_mutation


def main():
    instance = load_orlib_jobshop("datasets/ft06.txt")

    config = GAConfig(
        population_size=60,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism_size=1,
        seed=42,
        patience=80,
    )

    result = run_ga(
        instance=instance,
        selection=tournament_selection,
        crossover=pox_crossover,
        mutation=swap_mutation,
        config=config,
    )

    print("Best fitness:", result.best_fitness)
    print("Stopped at generation:", result.stopped_generation)
    print("First 10 best history:", result.history_best[:10])
    print("Last 10 best history:", result.history_best[-10:])


if __name__ == "__main__":
    main()
