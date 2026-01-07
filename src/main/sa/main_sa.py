from src.main.utils.jssp_instance import load_orlib_jobshop
from src.main.sa.simulated_annealing import simulated_annealing, SAConfig
from src.main.ga.mutation import swap_mutation


def main():
    instance = load_orlib_jobshop("datasets/abz9.txt")

    config = SAConfig(
        initial_temperature=1000.0,
        cooling_rate=0.995,
        min_temperature=0.1,
        max_iterations=50000,
        seed=42,
    )

    result = simulated_annealing(
        instance=instance, mutation=swap_mutation, config=config
    )

    print("SA best fitness:", result.best_fitness)


if __name__ == "__main__":
    main()
