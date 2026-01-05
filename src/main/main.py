from pathlib import Path
from random import shuffle
import random
from src.main.ga.chromosome import generate_random_chromosome
from src.main.utils.schedule import fitness
from src.main.utils.jssp_instance import load_orlib_jobshop, JSSPInstance


jssp_instance: JSSPInstance = load_orlib_jobshop("datasets/ft06.txt")

rng = random.Random(37)

chrom = generate_random_chromosome(jssp_instance, rng)

print("Chromosome length:", len(chrom))
print("Counts:", {j: chrom.count(j) for j in range(jssp_instance.n_jobs)})
print("Fitness:", fitness(jssp_instance, chrom))
