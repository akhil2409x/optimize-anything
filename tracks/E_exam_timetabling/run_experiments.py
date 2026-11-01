"""Track E experiments: baseline vs optimized exam timetabling.

    python run_experiments.py            # full run (~15 min on a laptop; the ILPs use most of it)
    python run_experiments.py --quick    # smoke run (~30 s)

Experiment 1 - heuristics at scale. Synthetic instances of 50-400 exams, five
seeds per size, plus any Carter .stu files placed in data/. Methods:
    first_fit (baseline), largest_degree, dsatur, dsatur_proximity,
    dsatur_proximity + hill_climb
Metrics: feasibility rate, Carter proximity cost, slots used, runtime.
For instances up to 100 exams the min-slots ILP gives the optimal slot count.

Experiment 2 - exact vs heuristic on small instances. The proximity ILP
(branch-and-bound in HiGHS, fixed time limit) against the best heuristic,
reporting the optimality gap the solver can prove.

All objective values are recomputed by evaluate.py; solver-reported values are
never used. Outputs go to results/.
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics as st
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from timetabling import data, evaluate, greedy, ilp, local_search

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def run_heuristics(inst, seed):
    """Return one row per method for this instance."""
    rows = []

    def record(method, fn):
        t0 = time.perf_counter()
        slots = fn()
        dt = time.perf_counter() - t0
        ok = evaluate.is_feasible(inst, slots)
        rows.append(dict(instance=inst.name, exams=inst.n_exams, students=inst.n_students,
                         slots_allowed=inst.n_slots, density=round(inst.density, 4),
                         seed=seed, method=method, feasible=ok,
                         cost=round(evaluate.proximity_cost(inst, slots), 4) if ok else "",
                         slots_used=evaluate.slots_used(slots) if ok else "",
                         seconds=round(dt, 4)))
        return slots if ok else None

    record("first_fit (baseline)", lambda: greedy.first_fit(inst))
    record("largest_degree", lambda: greedy.largest_degree(inst))
    record("dsatur", lambda: greedy.dsatur(inst))
    start = record("dsatur_proximity", lambda: greedy.dsatur_proximity(inst))
    if start is not None:
        record("dsatur_proximity+hill_climb",
               lambda: local_search.hill_climb(inst, start, seed=seed))
    else:
        rows.append(dict(rows[-1], method="dsatur_proximity+hill_climb"))
    return rows


def experiment_1(quick: bool):
    sizes = [(50, 400, 14), (100, 1000, 20), (200, 2500, 20), (400, 6000, 24)]
    seeds = range(2) if quick else range(5)
    if quick:
        sizes = sizes[:2]
    instances = [(data.synthetic(n, s, t, n_programmes=max(6, n // 15), seed=r), r)
                 for n, s, t in sizes for r in seeds]
    for stu in sorted((HERE / "data").glob("*.stu")):
        instances.append((data.load_carter(stu), 0))
    rows, ilp_rows = [], []
    for inst, seed in instances:
        print("  ", inst.summary())
        rows += run_heuristics(inst, seed)
        if inst.n_exams <= 100:
            r = ilp.min_slots(inst, time_limit=10 if quick else 20)
            ilp_rows.append(dict(instance=inst.name, exams=inst.n_exams, status=r.status,
                                 ilp_slots=r.objective if r.slots else "",
                                 feasible=evaluate.is_feasible(inst, r.slots),
                                 seconds=round(r.seconds, 2)))
    return rows, ilp_rows


def experiment_2(quick: bool):
    # Small instances are dense, so the slot budget is set per instance to the
    # optimal slot count (from the min-slots ILP) plus 3 spare slots. That leaves
    # room for the proximity objective to matter while staying tight.
    sizes = [(12, 100), (15, 150), (18, 150)]
    seeds = range(1) if quick else range(3)
    limit = 10 if quick else 30
    rows = []
    for n, s in sizes:
        for seed in seeds:
            inst = data.synthetic(n, s, n, n_programmes=4, seed=seed)
            chi = ilp.min_slots(inst, time_limit=30)
            if chi.status != "Optimal":
                continue
            t = int(chi.objective) + 3
            inst.n_slots = t
            inst.name = f"synth-e{n}-s{s}-t{t}-r{seed}"
            start = greedy.dsatur_proximity(inst) or greedy.dsatur(inst)
            if start is None:
                continue
            hc = local_search.hill_climb(inst, start, seed=seed)
            hc_cost = evaluate.proximity_cost(inst, hc)
            for warm in (False, True):
                r = ilp.min_proximity(inst, time_limit=limit, warm_start=hc if warm else None)
                ok = evaluate.is_feasible(inst, r.slots)
                gap = ""
                if ok and r.bound is not None and r.objective:
                    gap = round(100 * (r.objective - r.bound) / r.objective, 1)
                rows.append(dict(instance=inst.name, exams=n, slots_allowed=t,
                                 heuristic_cost=round(hc_cost, 4),
                                 ilp_warm_start=warm, ilp_status=r.status,
                                 ilp_cost=round(r.objective, 4) if ok else "",
                                 ilp_lower_bound=round(r.bound, 4) if r.bound is not None else "",
                                 gap_percent=gap, ilp_seconds=round(r.seconds, 1),
                                 ilp_vars=r.n_vars, ilp_constraints=r.n_constraints))
            print(f"   {inst.name}: heuristic {hc_cost:.2f} | ILP {rows[-2]['ilp_cost']} "
                  f"(cold) {rows[-1]['ilp_cost']} (warm), bound {rows[-1]['ilp_lower_bound']}")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    """mean/std per (size, method) over seeds, synthetic instances only."""
    groups = {}
    for r in rows:
        if not r["instance"].startswith("synth"):
            continue
        groups.setdefault((r["exams"], r["method"]), []).append(r)
    out = []
    for (n, m), rs in sorted(groups.items()):
        costs = [r["cost"] for r in rs if r["feasible"]]
        secs = [r["seconds"] for r in rs]
        out.append(dict(exams=n, method=m, runs=len(rs),
                        feasible=f"{sum(r['feasible'] for r in rs)}/{len(rs)}",
                        cost_mean=round(st.mean(costs), 2) if costs else "",
                        cost_std=round(st.stdev(costs), 2) if len(costs) > 1 else "",
                        slots_mean=round(st.mean(r["slots_used"] for r in rs if r["feasible"]), 1) if costs else "",
                        seconds_mean=round(st.mean(secs), 4)))
    return out


def plot(agg, path):
    methods = list(dict.fromkeys(a["method"] for a in agg))
    sizes = sorted({a["exams"] for a in agg})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        means, errs, xs = [], [], []
        for i, n in enumerate(sizes):
            a = next((a for a in agg if a["exams"] == n and a["method"] == m), None)
            if a and a["cost_mean"] != "":
                xs.append(i + k * width)
                means.append(a["cost_mean"])
                errs.append(a["cost_std"] or 0)
        ax.bar(xs, means, width, yerr=errs, capsize=2, label=m)
    ax.set_xticks([i + width * (len(methods) - 1) / 2 for i in range(len(sizes))])
    ax.set_xticklabels([f"{n} exams" for n in sizes])
    ax.set_ylabel("Carter proximity cost per student (lower is better)")
    ax.set_title("Exam timetabling: mean cost over feasible runs (error bar = 1 std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def write_summary(agg, ilp_rows, exp2_rows, path):
    parts = ["# Track E results (generated by run_experiments.py)\n",
             "Cost = Carter proximity cost per student (lower is better), recomputed "
             "independently by `evaluate.py`. Mean and std over seeds; only feasible "
             "runs enter the cost mean.\n",
             "## Experiment 1 — heuristics on synthetic instances\n",
             md_table(agg, ["exams", "method", "feasible", "cost_mean", "cost_std",
                            "slots_mean", "seconds_mean"]),
             "\n![cost by method](exp1_cost_by_method.png)\n",
             "## Experiment 1b — minimum slots needed (ILP, time-limited)\n",
             md_table(ilp_rows, ["instance", "status", "ilp_slots", "feasible", "seconds"]),
             "\n## Experiment 2 — proximity ILP vs best heuristic (small instances)\n",
             "Heuristic = DSatur-proximity + hill climbing. Lower bound is the best bound "
             "HiGHS proved; it stays at 0 because the LP relaxation of this model is weak.\n",
             md_table(exp2_rows, ["instance", "heuristic_cost", "ilp_warm_start",
                                  "ilp_status", "ilp_cost", "ilp_lower_bound", "ilp_seconds"])]
    path.write_text("\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(f"Python {platform.python_version()} on {platform.platform()}, {platform.processor() or 'cpu'}")
    print("Experiment 1: heuristics at scale")
    rows, ilp_rows = experiment_1(args.quick)
    write_csv(RESULTS / "exp1_heuristics_raw.csv", rows)
    write_csv(RESULTS / "exp1_min_slots_ilp.csv", ilp_rows)
    agg = aggregate(rows)
    write_csv(RESULTS / "exp1_heuristics_summary.csv", agg)
    plot(agg, RESULTS / "exp1_cost_by_method.png")
    print("Experiment 2: exact ILP vs heuristic (small instances)")
    exp2_rows = experiment_2(args.quick)
    write_csv(RESULTS / "exp2_ilp_vs_heuristic.csv", exp2_rows)
    write_summary(agg, ilp_rows, exp2_rows, RESULTS / "SUMMARY.md")
    with open(RESULTS / "environment.txt", "w") as f:
        f.write(f"python {platform.python_version()}\nplatform {platform.platform()}\n"
                f"machine {platform.machine()}\nrun {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"mode {'quick' if args.quick else 'full'}\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
