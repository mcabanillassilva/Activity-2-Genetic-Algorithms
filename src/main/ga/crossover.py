from typing import List, Optional, Set
import random

from src.main.ga.chromosome import Chromosome
from src.main.utils.jssp_instance import JSSPInstance


# Partial Order Crossover (POX) for Job Shop Scheduling Problem
def pox_crossover(
    parent1: Chromosome, parent2: Chromosome, instance: JSSPInstance, rng: random.Random
) -> Chromosome:
    """
    Partial Order Crossover (POX) adapted for job-based encoding. We need to ensure that we maintain
    the correct number of occurrences of each job in the child chromosome to conserve validity.

    eg. Parent1: [0, 1, 2, 0, 1, 2]
         Parent2: [1, 0, 1, 2, 0, 2]
         if selected jobs are {0,2} then
         Possible Child: [0, 1, 2, 0, 1, 2]
    """

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


def two_point_crossover(
    parent1: Chromosome, parent2: Chromosome, instance: JSSPInstance, rng: random.Random
) -> Chromosome:
    """
    Two-point crossover adapted for job-based encoding. We need to ensure that we maintain
    the correct number of occurrences of each job in the child chromosome to conserve validity.

    eg. Parent1: [0, 1, 2, 0, 1, 2]
         Parent2: [1, 0, 1, 2, 0, 2]
         if crossover points are (2,4) then
         _,_,2,0,_,_  (from Parent1)
         1,0,2,0,1,2 (from Parent2)
         Possible Child: [1,0,2,0,1,2]
    """

    n = len(parent1)
    child: List[Optional[int]] = [None] * n

    i, j = sorted(rng.sample(range(n), 2))

    # Copy segment from parent1
    child[i:j] = parent1[i:j]

    # Fill remaining positions from parent2
    required = {job: instance.n_machines for job in range(instance.n_jobs)}

    # Decrease counts for already copied genes
    for gene in child[i:j]:
        if gene is not None:
            required[gene] -= 1
    p2_idx = 0
    # Fill in the None positions
    for idx in range(n):
        if child[idx] is None:
            # º Find next gene in parent2 that is still required
            while required[parent2[p2_idx]] == 0:
                p2_idx += 1
            gene = parent2[p2_idx]
            child[idx] = gene
            required[gene] -= 1
            p2_idx += 1

    # Sanity check
    assert all(g is not None for g in child)

    return child  # type: ignore
