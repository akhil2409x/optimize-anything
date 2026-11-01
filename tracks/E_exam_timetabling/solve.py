"""Demo: timetable one Toronto instance with the baseline and the optimized method.

    python solve.py data/yor-f-83.stu
"""
import sys
import time

from timetabling import data, evaluate, greedy, local_search

path = sys.argv[1] if len(sys.argv) > 1 else "data/yor-f-83.stu"
inst = data.load_carter(path)
print(inst.summary())


def report(name, fn):
    t0 = time.perf_counter()
    slots = fn()
    dt = time.perf_counter() - t0
    if slots is None or not evaluate.is_feasible(inst, slots):
        print(f"  {name:32s} no clash-free timetable in {inst.n_slots} slots  ({dt:.3f} s)")
        return None
    print(f"  {name:32s} proximity cost {evaluate.proximity_cost(inst, slots):8.2f}  ({dt:.3f} s)")
    return slots


report("first fit (baseline)", lambda: greedy.first_fit(inst))
report("DSatur", lambda: greedy.dsatur(inst))
start = report("DSatur-proximity", lambda: greedy.dsatur_proximity(inst))
if start is not None:
    report("+ hill climbing", lambda: local_search.hill_climb(inst, start, seed=0))
