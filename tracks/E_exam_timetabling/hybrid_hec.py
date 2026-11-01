"""Hybrid for instances where every greedy construction fails (hec-s-92):
the min-slots ILP finds a clash-free timetable, then hill climbing lowers its
proximity cost. Five hill-climbing seeds, same as Experiment 1.

    python hybrid_hec.py [instance-name]
"""
import statistics as st
import sys
import time

from timetabling import data, evaluate, ilp, local_search

name = sys.argv[1] if len(sys.argv) > 1 else "hec-s-92"
inst = data.load_carter(f"data/{name}.stu")
t0 = time.perf_counter()
r = ilp.min_slots(inst, time_limit=20)
t_ilp = time.perf_counter() - t0
assert evaluate.is_feasible(inst, r.slots)
print(f"{inst.summary()}")
print(f"min-slots ILP: {r.status}, {r.objective:.0f} slots, feasible, cost {evaluate.proximity_cost(inst, r.slots):.2f}, "
      f"{t_ilp:.1f} s")
costs, secs = [], []
for seed in range(5):
    t0 = time.perf_counter()
    s = local_search.hill_climb(inst, r.slots, seed=seed)
    secs.append(time.perf_counter() - t0)
    assert evaluate.is_feasible(inst, s)
    costs.append(evaluate.proximity_cost(inst, s))
print(f"ILP + hill climb (5 seeds): cost {st.mean(costs):.2f} ± {st.pstdev(costs):.2f}, "
      f"min {min(costs):.2f}, {st.mean(secs):.2f} s per run")
