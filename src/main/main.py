from pathlib import Path
from random import shuffle
import random
from src.main.ga.chromosome import generate_random_chromosome
from src.main.ga.selection import tournament_selection, roulette_selection
from src.main.utils.schedule import fitness
from src.main.utils.jssp_instance import load_orlib_jobshop, JSSPInstance


jssp_instance: JSSPInstance = load_orlib_jobshop("datasets/ft06.txt")

rng = random.Random(48)


population = [generate_random_chromosome(jssp_instance, rng) for _ in range(10)]

winner = tournament_selection(population, jssp_instance, rng)
selected = roulette_selection(population, jssp_instance, rng)

print("Tournament winner fitness:", fitness(jssp_instance, winner))
print("Roulette selected fitness:", fitness(jssp_instance, selected))
