from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

Task = Tuple[int, int]  # (machine_id, processing_time)


@dataclass(frozen=True)
class JSSPInstance:
    n_jobs: int
    n_machines: int
    jobs: List[List[Task]]

    @property
    def total_operations(self) -> int:
        return self.n_jobs * self.n_machines


def load_orlib_jobshop(path: str) -> JSSPInstance:

    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_jobs, n_machines = map(int, lines[0].split())

    if len(lines) != n_jobs + 1:
        raise ValueError(f"Expected {n_jobs} job lines, got {len(lines) - 1}")

    jobs: List[List[Task]] = []

    for i in range(1, n_jobs + 1):
        tokens = list(map(int, lines[i].split()))
        if len(tokens) != 2 * n_machines:
            raise ValueError(f"Job line {i} does not contain {2*n_machines} values")

        job = [(tokens[j], tokens[j + 1]) for j in range(0, len(tokens), 2)]
        jobs.append(job)

    return JSSPInstance(n_jobs=n_jobs, n_machines=n_machines, jobs=jobs)
