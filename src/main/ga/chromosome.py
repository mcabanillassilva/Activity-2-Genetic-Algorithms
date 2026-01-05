from typing import List
import random

from src.main.utils.jssp_instance import JSSPInstance

Chromosome = List[int]


def generate_random_chromosome(
    instance: JSSPInstance, rng: random.Random
) -> Chromosome:
    """
    Job-based chromosome representation.
    Each job appears exactly n_machines times.
    """
    chromosome: Chromosome = []

    for job_id in range(instance.n_jobs):
        chromosome.extend([job_id] * instance.n_machines)

    rng.shuffle(chromosome)
    return chromosome
