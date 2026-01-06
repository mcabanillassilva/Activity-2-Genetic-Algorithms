import random
from typing import List

from src.main.utils.jssp_instance import JSSPInstance
from src.main.ga.chromosome import Chromosome
from src.main.utils.schedule import fitness


def tournament_selection(
    population: List[Chromosome], instance: JSSPInstance, rng: random.Random, k: int = 3
) -> Chromosome:
    """
    Tournament selection: choose k individuals at random
    and return the one with the best (lowest) fitness.
    """
    candidates = rng.sample(population, k)
    return min(candidates, key=lambda c: fitness(instance, c))


def roulette_selection(
    population: List[Chromosome], instance: JSSPInstance, rng: random.Random
) -> Chromosome:
    """
    Roulette wheel selection.
    Probability of selection is proportional to 1 / fitness.
    """
    # Compute inverted fitness scores
    scores = []
    for chrom in population:
        f = fitness(instance, chrom)
        scores.append(1.0 / f)

    total_score = sum(scores)

    # Spin the roulette wheel
    pick = rng.uniform(0, total_score)

    current = 0.0
    for chrom, score in zip(population, scores):
        current += score
        if current >= pick:
            return chrom

    return population[-1]
