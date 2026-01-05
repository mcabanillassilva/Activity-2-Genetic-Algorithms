from typing import List, Optional, Set, cast
import random

from src.main.ga.chromosome import Chromosome
from src.main.utils.jssp_instance import JSSPInstance


# Partial Order Crossover (POX) for Job Shop Scheduling Problem
def pox_crossover(
    parent1: Chromosome, parent2: Chromosome, instance: JSSPInstance, rng: random.Random
) -> Chromosome:

    n = len(parent1)
    child: List[Optional[int]] = [None] * n

    jobs = list(range(instance.n_jobs))
    rng.shuffle(jobs)
    k = rng.randint(1, instance.n_jobs - 1)
    selected_jobs: Set[int] = set(jobs[:k])

    # Copy genes from parent1 for selected jobs
    for i in range(n):
        if parent1[i] in selected_jobs:
            child[i] = parent1[i]

    # Fill remaining positions from parent2
    p2_iter = (job for job in parent2 if job not in selected_jobs)

    # Fill in the None positions
    for i in range(n):
        if child[i] is None:
            child[i] = next(p2_iter)

    # Sanity check
    assert all(g is not None for g in child)

    return child  # type: ignore
