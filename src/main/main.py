from pathlib import Path
from random import shuffle

from src.main.utils.jssp_instance import load_orlib_jobshop, JSSPInstance
from src.main.utils.schedule import decode_and_evaluate, fitness


jssp_instance: JSSPInstance = load_orlib_jobshop("datasets/ft06.txt")

chromosome = []
for job_id in range(jssp_instance.n_jobs):
    chromosome.extend([job_id] * jssp_instance.n_machines)

shuffle(chromosome)

print("Chromosome length:", len(chromosome))
print("Chromosome (first 20 genes):", chromosome[:20])

result = decode_and_evaluate(jssp_instance, chromosome)
print("Makespan:", result.makespan)
print("Fitness:", fitness(jssp_instance, chromosome))