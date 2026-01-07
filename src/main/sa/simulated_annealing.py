from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Callable, List

from src.main.utils.jssp_instance import JSSPInstance
from src.main.ga.chromosome import Chromosome, generate_random_chromosome
from src.main.utils.schedule import fitness


MutationFn = Callable[[Chromosome, random.Random], Chromosome]


@dataclass
class SAConfig:
    initial_temperature: float = 1000.0
    cooling_rate: float = 0.995
    min_temperature: float = 0.1
    max_iterations: int = 50_000
    seed: int = 42


@dataclass
class SAResult:
    best_chromosome: Chromosome
    best_fitness: int
    history_best: List[int]


def simulated_annealing(
    instance: JSSPInstance, mutation: MutationFn, config: SAConfig
) -> SAResult:

    rng = random.Random(config.seed)

    current = generate_random_chromosome(instance, rng)
    current_fit = fitness(instance, current)

    best = current
    best_fit = current_fit

    temperature = config.initial_temperature
    history: List[int] = []

    for _ in range(config.max_iterations):
        # Stop if temperature is too low
        if temperature < config.min_temperature:
            break

        # Select a random neighbor
        neighbor = mutation(current, rng)
        neighbor_fit = fitness(instance, neighbor)

        # Calculate fitness difference
        delta = neighbor_fit - current_fit

        # Accept if better or probabilistically if worse
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            # Move to neighbor
            current = neighbor
            # Update current fitness
            current_fit = neighbor_fit

            # Update best found
            if current_fit < best_fit:
                best = current
                best_fit = current_fit

        history.append(best_fit)
        # Cool down
        temperature *= config.cooling_rate

    return SAResult(best_chromosome=best, best_fitness=best_fit, history_best=history)
