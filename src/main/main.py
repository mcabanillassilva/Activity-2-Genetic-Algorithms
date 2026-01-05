import random

from src.main.utils.jssp_instance import load_orlib_jobshop
from src.main.ga.chromosome import generate_random_chromosome
from src.main.ga.crossover import pox_crossover, two_point_crossover
from src.main.utils.schedule import fitness

jssp_instance = load_orlib_jobshop("datasets/ft06.txt")

rng = random.Random(87)

parent1 = generate_random_chromosome(jssp_instance, rng)
parent2 = generate_random_chromosome(jssp_instance, rng)

print("=== Parents ===")
print("Parent 1 fitness:", fitness(jssp_instance, parent1))
print("Parent 2 fitness:", fitness(jssp_instance, parent2))

child_pox = pox_crossover(parent1, parent2, jssp_instance, rng)

print("\n=== POX Crossover ===")
print("Child fitness:", fitness(jssp_instance, child_pox))
print("Length:", len(child_pox))
print("Counts:", {j: child_pox.count(j) for j in range(jssp_instance.n_jobs)})

child_tp = two_point_crossover(parent1, parent2, jssp_instance, rng)

print("\n=== Two-point Crossover ===")
print("Child fitness:", fitness(jssp_instance, child_tp))
print("Length:", len(child_tp))
print("Counts:", {j: child_tp.count(j) for j in range(jssp_instance.n_jobs)})
