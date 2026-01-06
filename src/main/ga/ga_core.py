from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import random

from src.main.utils.jssp_instance import JSSPInstance
from src.main.ga.chromosome import Chromosome, generate_random_chromosome
from src.main.utils.schedule import fitness


SelectionFn = Callable[[List[Chromosome], JSSPInstance, random.Random], Chromosome]
CrossoverFn = Callable[
    [Chromosome, Chromosome, JSSPInstance, random.Random], Chromosome
]
MutationFn = Callable[[Chromosome, random.Random], Chromosome]


@dataclass
class GAConfig:
    population_size: int = 50  # number of individuals in population
    generations: int = 200  # total number of generations
    crossover_rate: float = 0.9  # probability of crossover per pair
    mutation_rate: float = 0.2  # probability of mutation per individual
    elitism_size: int = 1  # number of elite individuals to carry over
    seed: int = 42  # random seed
    # early stop (stationary)
    patience: int = 50  # generations with no improvement
    min_delta: int = 0  # minimum improvement to reset patience


@dataclass
class GAResult:
    best_chromosome: Chromosome
    best_fitness: int  # best fitness value
    history_best: List[int]  # best of each generation
    history_avg: List[float]  # avg fitness each generation
    stopped_generation: int  # generation at which the GA stopped


def run_ga(
    instance: JSSPInstance,
    selection: SelectionFn,
    crossover: CrossoverFn,
    mutation: MutationFn,
    config: GAConfig,
) -> GAResult:
    rng = random.Random(config.seed)

    # --- init population ---
    population: List[Chromosome] = [
        generate_random_chromosome(instance, rng) for _ in range(config.population_size)
    ]

    def eval_pop(pop: List[Chromosome]) -> List[int]:
        return [fitness(instance, c) for c in pop]

    best_history: List[int] = []
    avg_history: List[float] = []

    best_overall: Optional[Chromosome] = None
    best_overall_fit: int = 10**18

    no_improve = 0

    for gen in range(config.generations):
        fits = eval_pop(population)

        # stats
        gen_best_fit = min(fits)
        gen_avg_fit = sum(fits) / len(fits)
        best_history.append(gen_best_fit)
        avg_history.append(gen_avg_fit)

        # track global best
        if gen_best_fit < best_overall_fit - config.min_delta:
            best_overall_fit = gen_best_fit
            best_overall = population[fits.index(gen_best_fit)]
            no_improve = 0
        else:
            no_improve += 1

        # stationary stop
        if config.patience > 0 and no_improve >= config.patience:
            return GAResult(
                best_chromosome=(
                    best_overall if best_overall is not None else population[0]
                ),
                best_fitness=(
                    best_overall_fit if best_overall is not None else gen_best_fit
                ),
                history_best=best_history,
                history_avg=avg_history,
                stopped_generation=gen,
            )

        # elitism: keep top E individuals
        elite_count = max(0, min(config.elitism_size, config.population_size))
        elites: List[Chromosome] = []
        if elite_count > 0:
            ranked = sorted(zip(population, fits), key=lambda x: x[1])
            elites = [c for c, _ in ranked[:elite_count]]

        # build next generation
        new_population: List[Chromosome] = []
        new_population.extend(elites)

        while len(new_population) < config.population_size:
            # select parents
            p1 = selection(population, instance, rng)
            p2 = selection(population, instance, rng)

            # crossover
            if rng.random() < config.crossover_rate:
                child = crossover(p1, p2, instance, rng)
            else:
                child = p1.copy()

            # mutation
            if rng.random() < config.mutation_rate:
                child = mutation(child, rng)

            new_population.append(child)

        population = new_population

    # finished all generations
    if best_overall is None:
        fits = eval_pop(population)
        gen_best_fit = min(fits)
        best_overall = population[fits.index(gen_best_fit)]
        best_overall_fit = gen_best_fit

    return GAResult(
        best_chromosome=best_overall,
        best_fitness=best_overall_fit,
        history_best=best_history,
        history_avg=avg_history,
        stopped_generation=config.generations - 1,
    )
