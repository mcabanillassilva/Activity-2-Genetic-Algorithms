import random
from src.main.ga.chromosome import Chromosome


def swap_mutation(
    chromosome: Chromosome,
    rng: random.Random
) -> Chromosome:
    """
    Swap mutation: swap two random positions of the chromosome.
    """
    mutated = chromosome.copy()

    i, j = rng.sample(range(len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated 
