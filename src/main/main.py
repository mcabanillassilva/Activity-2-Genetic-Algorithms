import random

from src.main.utils.jssp_instance import load_orlib_jobshop
from src.main.ga.chromosome import generate_random_chromosome
from src.main.ga.crossover import pox_crossover
from src.main.utils.schedule import fitness

jssp_instance = load_orlib_jobshop("datasets/ft06.txt")

rng = random.Random(79)

parent1 = generate_random_chromosome(jssp_instance, rng)
parent2 = generate_random_chromosome(jssp_instance, rng)

print("Parent 1 fitness:", fitness(jssp_instance, parent1))
print("Parent 2 fitness:", fitness(jssp_instance, parent2))

child = pox_crossover(parent1, parent2, jssp_instance, rng)

print("\nChild fitness:", fitness(jssp_instance, child))

print("\nSanity checks:")
print("Child length:", len(child))
print("Counts:", {j: child.count(j) for j in range(jssp_instance.n_jobs)})
