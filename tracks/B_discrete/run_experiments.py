"""Track B experiments.

    python run_experiments.py            # full run (~5 min)
    python run_experiments.py --quick    # smoke run (~40 s)

Exp 1  Knapsack: 5 methods x 3 instance classes x sizes, 3 seeds.
       Metrics: gap to the proven optimum (%), runtime, B&B nodes.
Exp 2  Pseudo-polynomial DP: fixed n, growing capacity. DP time grows with C;
       B&B and ILP do not care about the size of the numbers.
Exp 3  Knapsack as QUBO: penalty weight sweep. How often is the QUBO ground
       state a feasible / optimal knapsack solution?
Exp 4  Max-Cut QUBO/Ising (10 nodes): brute-force optimum vs one-flip local
       search over random restarts. Writes results/maxcut_instance.json for Track C.
Exp 5  Feature selection: exhaustive vs greedy forward vs random subsets with
       the same evaluation budget as greedy.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics as st
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discrete import feature_selection as FS
from discrete import knapsack as K
from discrete import qubo as Q

HERE = Path(__file__).parent
RESULTS = HERE / "results"
KINDS = ["uncorrelated", "weakly_correlated", "strongly_correlated"]
DP_CELL_LIMIT = 5e7          # skip DP above this table size (memory/time)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ Exp 1
def exp1(quick):
    sizes = [50, 100, 200] if quick else [50, 100, 200, 400, 800]
    seeds = range(2) if quick else range(3)
    rows = []
    for kind in KINDS:
        for n in sizes:
            for seed in seeds:
                k = K.generate(n, kind, R=1000, seed=seed)
                sols = {}
                for name, fn in K.METHODS.items():
                    if name == "dp" and n * (k.capacity + 1) > DP_CELL_LIMIT:
                        continue
                    if name == "branch_and_bound":
                        s = fn(k, time_limit=10)
                    elif name == "ilp_highs":
                        s = fn(k, time_limit=30)
                    else:
                        s = fn(k)
                    sols[name] = s
                proven = [s.value for nm, s in sols.items() if s.status == "Optimal"]
                best = max(s.value for s in sols.values())
                ref = max(proven) if proven else best
                for name, s in sols.items():
                    rows.append(dict(kind=kind, n=n, seed=seed, capacity=k.capacity, method=name,
                                     value=s.value, reference=ref, reference_proven=bool(proven),
                                     gap_percent=round(100 * (ref - s.value) / ref, 4),
                                     status=s.status, seconds=round(s.seconds, 7), nodes=s.nodes))
                print(f"   {kind:20s} n={n:4d} seed={seed} ref={ref}"
                      + ("" if proven else " (NOT proven)"), flush=True)
    return rows


def exp1_summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["n"], r["method"]), []).append(r)
    out = []
    for (kind, n, m), rs in groups.items():
        out.append(dict(kind=kind, n=n, method=m, runs=len(rs),
                        proven_optimal=f"{sum(r['status'] == 'Optimal' for r in rs)}/{len(rs)}",
                        gap_mean_pct=round(st.mean(r["gap_percent"] for r in rs), 3),
                        gap_max_pct=round(max(r["gap_percent"] for r in rs), 3),
                        seconds_mean=float(f"{st.mean(r['seconds'] for r in rs):.3g}"),
                        nodes_mean=int(st.mean(r["nodes"] for r in rs)) if m == "branch_and_bound" else ""))
    return out


def exp1_plot(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for m in K.METHODS:
            pts = sorted((r["n"], r["seconds_mean"]) for r in summary
                         if r["kind"] == kind and r["method"] == m)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=m)
        ax.set_xscale("log"); ax.set_yscale("log")
        ns = sorted({r["n"] for r in summary})
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_title(kind.replace("_", " "))
        ax.set_xlabel("items n")
    axes[0].set_ylabel("mean runtime (s, log)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Knapsack runtime by instance class (B&B capped at 10 s, ILP at 30 s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ------------------------------------------------------------------ Exp 2
def exp2(quick):
    Rs = [100, 1000, 10000] if quick else [100, 1000, 10000, 50000, 200000]
    rows = []
    for R in Rs:
        k = K.generate(60, "weakly_correlated", R=R, seed=0)
        for name in ["dp", "branch_and_bound", "ilp_highs"]:
            if name == "dp" and k.n * (k.capacity + 1) > 2e8:
                continue
            s = K.METHODS[name](k)
            rows.append(dict(R=R, capacity=k.capacity, method=name, value=s.value,
                             seconds=float(f"{s.seconds:.3g}"),
                             dp_table_cells=s.extra.get("table_cells", "")))
        print(f"   R={R} C={k.capacity}", flush=True)
    return rows


def exp2_plot(rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ["dp", "branch_and_bound", "ilp_highs"]:
        pts = [(r["capacity"], r["seconds"]) for r in rows if r["method"] == m]
        ax.plot(*zip(*pts), marker="o", label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("capacity C (log)"); ax.set_ylabel("runtime (s, log)")
    ax.set_title("n = 60 items: DP is O(n·C), the others are not")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ------------------------------------------------------------------ Exp 3
def exp3(quick):
    seeds = range(8) if quick else range(30)
    multipliers = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    rows = []
    for mult in multipliers:
        feas = opt = 0
        for seed in seeds:
            k = K.generate(6, "uncorrelated", R=12, seed=seed)
            best = K.brute_force(k).value
            # penalty expressed relative to the largest item value
            q, n, Ks = Q.knapsack_qubo(k.values, k.weights, k.capacity, penalty=mult * max(k.values))
            x, _, _ = Q.brute_force(q)
            v, _, ok = K.evaluate(k, [i for i in range(n) if x[i]])
            feas += ok
            opt += ok and v == best
        rows.append(dict(penalty_over_max_value=mult, instances=len(seeds),
                         ground_state_feasible=f"{feas}/{len(seeds)}",
                         ground_state_optimal=f"{opt}/{len(seeds)}",
                         qubits=f"{n} items + {Ks} slack"))
        print(f"   P={mult}*max(v): feasible {feas}/{len(seeds)}, optimal {opt}/{len(seeds)}", flush=True)
    return rows


# ------------------------------------------------------------------ Exp 4
def exp4(quick):
    n = 10
    edges = Q.random_graph(n, 0.5, seed=0)
    q = Q.maxcut_qubo(n, edges)
    t0 = time.perf_counter()
    x, e, energies = Q.brute_force(q)
    bf_secs = time.perf_counter() - t0
    best = int(round(-e))
    n_opt = int((energies == e).sum())
    ising = Q.qubo_to_ising(q)
    import numpy as np
    opt_idx = np.flatnonzero(np.isclose(energies, e))
    all_opt = [[(int(i) >> b) & 1 for b in range(n)] for i in opt_idx]
    Q.save_maxcut(RESULTS / "maxcut_instance.json", n, edges, x, best, ising, all_opt)
    restarts = 20 if quick else 200
    cuts = [Q.cut_value(edges, Q.maxcut_greedy(n, edges, seed=s)) for s in range(restarts)]
    rows = [dict(method="brute force (2^10 states)", runs=1, best_cut=best, mean_cut=best,
                 hit_optimum="1/1", seconds=round(bf_secs, 4)),
            dict(method="one-flip local search", runs=restarts, best_cut=max(cuts),
                 mean_cut=round(st.mean(cuts), 2),
                 hit_optimum=f"{sum(c == best for c in cuts)}/{restarts}", seconds="")]
    info = dict(n_nodes=n, n_edges=len(edges), total_weight=sum(w for *_, w in edges),
                optimal_cut=best, optimal_states=n_opt,
                approx_ratio_mean=round(st.mean(cuts) / best, 3))
    print(f"   Max-Cut: {info}", flush=True)
    return rows, info


# ------------------------------------------------------------------ Exp 5
def exp5(quick):
    d, k = (10, 3) if quick else (12, 4)
    rows = []
    p = FS.Problem(d=d)
    ex_s, ex_acc, ex_t = FS.exhaustive(p, k)
    n_ex = p.evaluations
    p = FS.Problem(d=d)
    gr_s, gr_acc, gr_t = FS.greedy_forward(p, k)
    budget = p.evaluations
    rows.append(dict(method="exhaustive", evaluations=n_ex, accuracy=round(ex_acc, 4),
                     features=str(ex_s), seconds=round(ex_t, 2)))
    rows.append(dict(method="greedy forward", evaluations=budget, accuracy=round(gr_acc, 4),
                     features=str(gr_s), seconds=round(gr_t, 2)))
    accs = []
    for seed in range(3 if quick else 10):
        p = FS.Problem(d=d)
        _, acc, _ = FS.random_subsets(p, k, budget, seed=seed)
        accs.append(acc)
    rows.append(dict(method=f"random subsets (baseline, {len(accs)} seeds)", evaluations=budget,
                     accuracy=f"{st.mean(accs):.4f} ± {st.stdev(accs):.4f}",
                     features="best per seed varies", seconds=""))
    print(f"   d={d}, k={k}: exhaustive {ex_acc:.4f} ({n_ex} evals), greedy {gr_acc:.4f} "
          f"({budget} evals), random {st.mean(accs):.4f}", flush=True)
    return rows, dict(d=d, k=k)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(f"Python {platform.python_version()} on {platform.platform()}")

    print("Exp 1: knapsack methods"); r1 = exp1(args.quick)
    s1 = exp1_summary(r1)
    write_csv(RESULTS / "exp1_knapsack_raw.csv", r1)
    write_csv(RESULTS / "exp1_knapsack_summary.csv", s1)
    exp1_plot(s1, RESULTS / "exp1_runtime.png")

    print("Exp 2: DP vs capacity"); r2 = exp2(args.quick)
    write_csv(RESULTS / "exp2_dp_capacity.csv", r2)
    exp2_plot(r2, RESULTS / "exp2_dp_capacity.png")

    print("Exp 3: knapsack QUBO penalty sweep"); r3 = exp3(args.quick)
    write_csv(RESULTS / "exp3_qubo_penalty.csv", r3)

    print("Exp 4: Max-Cut QUBO/Ising"); r4, info4 = exp4(args.quick)
    write_csv(RESULTS / "exp4_maxcut.csv", r4)

    print("Exp 5: feature selection"); r5, info5 = exp5(args.quick)
    write_csv(RESULTS / "exp5_feature_selection.csv", r5)

    (RESULTS / "SUMMARY.md").write_text("\n".join([
        "# Track B results (generated by run_experiments.py)\n",
        f"Mode: {'quick' if args.quick else 'full'}. Python {platform.python_version()}, "
        f"{platform.machine()}. Values are recomputed from chosen items, never taken from a solver.\n",
        "## Exp 1 — knapsack: gap to proven optimum and runtime\n",
        "Reference = best value among runs that *proved* optimality. Greedy methods are heuristics, "
        "so `proven_optimal` is 0 for them by definition.\n",
        md_table(s1, ["kind", "n", "method", "proven_optimal", "gap_mean_pct", "gap_max_pct",
                      "seconds_mean", "nodes_mean"]),
        "\n![runtime](exp1_runtime.png)\n",
        "## Exp 2 — DP is pseudo-polynomial\n",
        md_table(r2, ["R", "capacity", "method", "value", "seconds", "dp_table_cells"]),
        "\n![dp vs capacity](exp2_dp_capacity.png)\n",
        "## Exp 3 — knapsack as QUBO: penalty weight\n",
        "6 items, uncorrelated, brute-force ground state of the QUBO checked against the true optimum.\n",
        md_table(r3, ["penalty_over_max_value", "instances", "ground_state_feasible",
                      "ground_state_optimal", "qubits"]),
        "\n## Exp 4 — Max-Cut QUBO/Ising (input for Track C)\n",
        f"Graph: {info4['n_nodes']} nodes, {info4['n_edges']} edges, optimum cut "
        f"{info4['optimal_cut']} ({info4['optimal_states']} optimal bitstrings, incl. complements). "
        f"Mean local-search approximation ratio {info4['approx_ratio_mean']}. "
        "Instance saved to `maxcut_instance.json`.\n",
        md_table(r4, ["method", "runs", "best_cut", "mean_cut", "hit_optimum", "seconds"]),
        f"\n## Exp 5 — feature selection (d = {info5['d']}, k = {info5['k']}, breast-cancer data)\n",
        "Accuracy = 5-fold stratified CV of standardised logistic regression.\n",
        md_table(r5, ["method", "evaluations", "accuracy", "features", "seconds"]),
        "",
    ]))
    (RESULTS / "environment.txt").write_text(
        f"python {platform.python_version()}\nplatform {platform.platform()}\n"
        f"run {time.strftime('%Y-%m-%d %H:%M:%S')}\nmode {'quick' if args.quick else 'full'}\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
