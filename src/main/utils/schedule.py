from dataclasses import dataclass
from typing import List

from src.main.utils.jssp_instance import JSSPInstance

Chromosome = List[int]


@dataclass
class ScheduleResult:
    """
    Result of decoding a chromosome into a schedule.
    """

    makespan: int


def decode_and_evaluate(
    instance: JSSPInstance, chromosome: Chromosome
) -> ScheduleResult:
    """
    Decodes a job-based chromosome into a schedule.
    """

    # Initialize tracking structures
    next_operation = [0] * instance.n_jobs
    job_ready_time = [0] * instance.n_jobs
    machine_ready_time = [0] * instance.n_machines

    # Validate chromosome length
    if len(chromosome) != instance.n_jobs * instance.n_machines:
        raise ValueError("Invalid chromosome length")

    # Decode the chromosome
    for job_id in chromosome:
        # Get the next operation for this job
        op_idx = next_operation[job_id]
        # Get machine and duration for this operation
        machine, duration = instance.jobs[job_id][op_idx]

        # Determine the earliest start time
        start_time = max(job_ready_time[job_id], machine_ready_time[machine])

        # Schedule the operation
        finish_time = start_time + duration

        # Update ready times
        job_ready_time[job_id] = finish_time
        machine_ready_time[machine] = finish_time
        next_operation[job_id] += 1

    # The makespan is the maximum completion time across all jobs
    makespan = max(job_ready_time)
    return ScheduleResult(makespan=makespan)


def fitness(instance: JSSPInstance, chromosome: Chromosome) -> int:
    """
    Fitness function to minimize makespan.
    """
    return decode_and_evaluate(instance, chromosome).makespan
