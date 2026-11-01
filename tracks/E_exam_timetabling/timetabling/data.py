"""Problem instances for uncapacitated exam timetabling (Carter et al., 1996).

An instance is a set of exams and a set of students, each student enrolled in
some exams. Two exams *conflict* if at least one student takes both; the
number of shared students is the conflict weight w_ij.

Two sources are supported:
  * Carter/Toronto benchmark files (.crs + .stu), downloaded manually from
    https://www.cs.nott.ac.uk/~pszrq/data.htm into tracks/E_exam_timetabling/data/
  * a seeded synthetic generator that mimics their structure (students grouped
    into programmes that share core courses), used for tests and for small
    instances that an exact ILP can still solve.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

# Number of time slots used in the standard Carter benchmark versions (v1).
CARTER_SLOTS = {
    "car-f-92": 32, "car-s-91": 35, "ear-f-83": 24, "hec-s-92": 18,
    "kfu-s-93": 20, "lse-f-91": 18, "pur-s-93": 42, "rye-s-93": 23,
    "sta-f-83": 13, "tre-s-92": 23, "uta-s-92": 35, "ute-s-92": 10,
    "yor-f-83": 21,
}


@dataclass
class Instance:
    name: str
    n_exams: int
    students: list[list[int]]          # each student's exams, 0-indexed
    n_slots: int
    conflicts: dict[tuple[int, int], int] = field(default_factory=dict)  # (i<j) -> w_ij
    neighbours: list[set[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_exams <= 0:
            raise ValueError("instance needs at least one exam")
        if self.n_slots <= 0:
            raise ValueError("instance needs at least one time slot")
        w: dict[tuple[int, int], int] = defaultdict(int)
        for exams in self.students:
            for e in exams:
                if not 0 <= e < self.n_exams:
                    raise ValueError(f"exam id {e} out of range 0..{self.n_exams - 1}")
            for i, j in combinations(sorted(set(exams)), 2):
                w[(i, j)] += 1
        self.conflicts = dict(w)
        self.neighbours = [set() for _ in range(self.n_exams)]
        for i, j in self.conflicts:
            self.neighbours[i].add(j)
            self.neighbours[j].add(i)

    @property
    def n_students(self) -> int:
        return len(self.students)

    @property
    def density(self) -> float:
        """Conflict density: share of exam pairs that conflict (Carter's metric)."""
        n = self.n_exams
        return 2 * len(self.conflicts) / (n * (n - 1)) if n > 1 else 0.0

    def summary(self) -> str:
        return (f"{self.name}: {self.n_exams} exams, {self.n_students} students, "
                f"{self.n_slots} slots, {len(self.conflicts)} conflicting pairs, "
                f"density {self.density:.3f}")


def load_carter(stu_path: str | Path, n_slots: int | None = None) -> Instance:
    """Load a Carter .stu file. Exam ids in the file are 1-based integers.

    The .crs file only lists enrolments, which can be rebuilt from .stu, so it is
    not needed. Lines may be blank (students with no exams) and are skipped.
    """
    stu_path = Path(stu_path)
    name = stu_path.stem
    if n_slots is None:
        if name not in CARTER_SLOTS:
            raise ValueError(f"unknown instance {name!r}; pass n_slots explicitly")
        n_slots = CARTER_SLOTS[name]
    students: list[list[int]] = []
    max_id = 0
    for line in stu_path.read_text().splitlines():
        ids = [int(tok) for tok in line.split()]
        if ids:
            students.append([i - 1 for i in ids])
            max_id = max(max_id, *ids)
    return Instance(name, max_id, students, n_slots)


def synthetic(n_exams: int, n_students: int, n_slots: int, *, n_programmes: int = 6,
              exams_per_student: tuple[int, int] = (3, 6), seed: int = 0) -> Instance:
    """Seeded synthetic instance with programme structure.

    Each programme owns a block of exams; a student takes mostly exams from their
    own programme plus occasionally an elective from elsewhere. This produces the
    clustered conflict graphs seen in real university data rather than a uniform
    random graph.
    """
    if n_programmes > n_exams:
        raise ValueError("more programmes than exams")
    rng = random.Random(seed)
    exams = list(range(n_exams))
    rng.shuffle(exams)
    blocks = [exams[p::n_programmes] for p in range(n_programmes)]
    students = []
    for _ in range(n_students):
        home = blocks[rng.randrange(n_programmes)]
        k = rng.randint(*exams_per_student)
        chosen = set(rng.sample(home, min(k, len(home))))
        if rng.random() < 0.3:                     # cross-programme elective
            chosen.add(rng.randrange(n_exams))
        students.append(sorted(chosen))
    return Instance(f"synth-e{n_exams}-s{n_students}-t{n_slots}-r{seed}",
                    n_exams, students, n_slots)
