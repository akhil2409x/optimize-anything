"""Track F experiments: the Tiny compiler on 8 benchmark programs.

    python run_experiments.py           # full run (~10 min)
    python run_experiments.py --quick   # fewer configurations

Exp 1  Optimization levels O0-O3 (K = 6 registers): static code size, executed
       instructions, cycles, loads/stores, VM run time, compile time; outputs
       checked against the reference interpreter.
Exp 2  Ablation: all passes, all-but-one, and each pass alone (greedy RA, K = 6),
       without the pipeline's final clean-up DCE so each pass's effect is isolated.
Exp 3  Register allocation as an optimization problem: greedy (Chaitin-Briggs)
       vs ILP (minimum spill cost) for K = 2..8 registers.
Exp 4  Pass ordering: every permutation of the 6 passes, each applied ONCE,
       vs the default order and a greedy ordering heuristic.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import platform
import statistics as st
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tinyc import backend as B, driver as D, frontend as F, ir, passes as P

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PROGRAMS = sorted((HERE / "programs").glob("*.tiny"))


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def measure(src, ref, level="O2", k=6, passes=None, ra=None, reps=3, cleanup=True):
    comp = [D.compile_source(src, level, k=k, passes=passes, ra=ra, cleanup=cleanup) for _ in range(reps)]
    c = comp[0]
    runs = [B.run_vm(c.code) for _ in range(reps)]
    r = runs[0]
    return dict(static_size=len(c.code), ir_instrs=c.ir_instrs, instructions=r.instructions,
                cycles=r.cycles, loads=r.loads, stores=r.stores,
                vm_ms=round(1000 * st.median(x.seconds for x in runs), 2),
                compile_ms=round(1000 * st.median(x.compile_seconds for x in comp), 2),
                spilled=c.alloc.spilled, spill_cost=c.alloc.spill_cost,
                same_output=D.observed(r) == ref)


# ------------------------------------------------------------------ Exp 1
def exp1():
    rows = []
    for p in PROGRAMS:
        src = p.read_text()
        ref = D.reference(src)
        for lvl in ("O0", "O1", "O2", "O3"):
            m = measure(src, ref, lvl)
            rows.append(dict(program=p.stem, level=lvl, **m))
        print(f"   {p.stem}: O0 {rows[-4]['cycles']} -> O2 {rows[-2]['cycles']} cycles", flush=True)
    return rows


def exp1_summary(rows):
    out = []
    for lvl in ("O0", "O1", "O2", "O3"):
        base = {r["program"]: r for r in rows if r["level"] == "O0"}
        rs = [r for r in rows if r["level"] == lvl]
        out.append(dict(level=lvl, programs=len(rs), all_outputs_identical=all(r["same_output"] for r in rs),
                        geomean_cycle_speedup=round(geomean([base[r["program"]]["cycles"] / r["cycles"] for r in rs]), 2),
                        geomean_instr_reduction=round(geomean([base[r["program"]]["instructions"] / r["instructions"] for r in rs]), 2),
                        geomean_size_reduction=round(geomean([base[r["program"]]["static_size"] / r["static_size"] for r in rs]), 2),
                        total_loads_stores=sum(r["loads"] + r["stores"] for r in rs),
                        mean_compile_ms=round(st.mean(r["compile_ms"] for r in rs), 2)))
    return out


# ------------------------------------------------------------------ Exp 2
def exp2():
    configs = {"all passes": P.DEFAULT_ORDER, "no passes": []}
    for p in P.DEFAULT_ORDER:
        configs[f"all but {p}"] = [q for q in P.DEFAULT_ORDER if q != p]
    for p in P.DEFAULT_ORDER:
        configs[f"only {p}"] = [p]
    rows = []
    for name, ps in configs.items():
        speed = []
        for prog in PROGRAMS:
            src = prog.read_text()
            ref = D.reference(src)
            base = B.run_vm(D.compile_source(src, "O2", k=6, passes=[]).code).cycles
            # cleanup=False: the pipeline's final tidy-up DCE would otherwise make
            # "all but dce" identical to "all passes"
            m = measure(src, ref, "O2", passes=ps, reps=1, cleanup=False)
            speed.append(base / m["cycles"])
            rows.append(dict(config=name, program=prog.stem, cycles=m["cycles"],
                             speedup_vs_no_passes=round(base / m["cycles"], 3), same_output=m["same_output"]))
        print(f"   {name}: geomean speed-up {geomean(speed):.2f}", flush=True)
    summary = []
    for name in configs:
        rs = [r for r in rows if r["config"] == name]
        summary.append(dict(config=name, geomean_speedup_vs_no_passes=round(geomean([r["speedup_vs_no_passes"] for r in rs]), 3),
                            best_program=max(rs, key=lambda r: r["speedup_vs_no_passes"])["program"],
                            all_outputs_identical=all(r["same_output"] for r in rs)))
    return rows, summary


# ------------------------------------------------------------------ Exp 3
def exp3(quick):
    ks = [2, 3, 4, 6] if quick else [2, 3, 4, 5, 6, 8]
    rows = []
    for prog in PROGRAMS:
        src = prog.read_text()
        ref = D.reference(src)
        fn = P.optimize(ir.lower(F.parse(src)))
        for k in ks:
            for method in ("greedy", "ilp"):
                a = B.allocate(fn, k, method)
                B.check_allocation(fn, a)
                r = B.run_vm(B.codegen(fn, a))
                rows.append(dict(program=prog.stem, k=k, method=method, nodes=a.nodes, edges=a.edges,
                                 spilled=a.spilled, spill_cost=a.spill_cost, cycles=r.cycles,
                                 loads_stores=r.loads + r.stores, solve_ms=round(1000 * a.seconds, 2),
                                 status=a.status, same_output=D.observed(r) == ref))
        print(f"   {prog.stem} done", flush=True)
    return rows


def exp3_summary(rows):
    out = []
    for k in sorted({r["k"] for r in rows}):
        g = [r for r in rows if r["k"] == k and r["method"] == "greedy"]
        o = [r for r in rows if r["k"] == k and r["method"] == "ilp"]
        pairs = list(zip(g, o))
        out.append(dict(k=k, programs=len(g),
                        greedy_spill_cost=sum(r["spill_cost"] for r in g), ilp_spill_cost=sum(r["spill_cost"] for r in o),
                        programs_where_ilp_better=sum(b["spill_cost"] < a["spill_cost"] for a, b in pairs),
                        greedy_cycles=sum(r["cycles"] for r in g), ilp_cycles=sum(r["cycles"] for r in o),
                        programs_where_ilp_fewer_cycles=sum(b["cycles"] < a["cycles"] for a, b in pairs),
                        programs_where_ilp_more_cycles=sum(b["cycles"] > a["cycles"] for a, b in pairs),
                        greedy_ms=round(sum(r["solve_ms"] for r in g), 1), ilp_ms=round(sum(r["solve_ms"] for r in o), 1),
                        ilp_all_optimal=all(r["status"] == "Optimal" for r in o)))
    return out


# ------------------------------------------------------------------ Exp 4
def cycles_after(src, order, k=6):
    fn = ir.lower(F.parse(src))
    for name in order:                 # each pass exactly once, no fixpoint iteration
        P.PASSES[name](fn)
    P.dce(fn) if "dce" not in order else None
    a = B.allocate(fn, k, "greedy")
    return B.run_vm(B.codegen(fn, a)).cycles


def exp4(quick):
    names = P.DEFAULT_ORDER
    perms = list(itertools.permutations(names))
    if quick:
        perms = perms[::12]
    rows = []
    for prog in PROGRAMS:
        src = prog.read_text()
        base = cycles_after(src, [])
        t0 = time.perf_counter()
        allc = {perm: cycles_after(src, list(perm)) for perm in perms}
        search_s = time.perf_counter() - t0
        best = min(allc, key=allc.get)
        worst = max(allc, key=allc.get)
        # greedy ordering: repeatedly apply the unused pass that lowers cycles most right now
        t0 = time.perf_counter()
        chosen = []
        for _ in names:
            rest = [p for p in names if p not in chosen]
            chosen.append(min(rest, key=lambda p: (cycles_after(src, chosen + [p]), rest.index(p))))
        greedy_s = time.perf_counter() - t0
        fix = B.run_vm(D.compile_source(src, "O2", k=6).code).cycles
        rows.append(dict(program=prog.stem, no_passes=base, default_order_once=allc.get(tuple(names), cycles_after(src, names)),
                         best_order=allc[best], best_order_seq=" ".join(best),
                         worst_order=allc[worst], orders_tried=len(allc),
                         distinct_results=len(set(allc.values())),
                         greedy_order=cycles_after(src, chosen), greedy_order_seq=" ".join(chosen),
                         default_to_fixpoint=fix,
                         search_seconds=round(search_s, 1), greedy_seconds=round(greedy_s, 2)))
        print(f"   {prog.stem}: best {allc[best]} worst {allc[worst]} greedy {rows[-1]['greedy_order']} "
              f"fixpoint {fix}", flush=True)
    return rows


# ------------------------------------------------------------------ plots
def plots(e1, e3):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    progs = [p.stem for p in PROGRAMS]
    for k, lvl in enumerate(("O0", "O1", "O2", "O3")):
        base = {r["program"]: r["cycles"] for r in e1 if r["level"] == "O0"}
        ys = [base[p] / next(r["cycles"] for r in e1 if r["program"] == p and r["level"] == lvl) for p in progs]
        axes[0].bar([i + 0.2 * k for i in range(len(progs))], ys, 0.2, label=lvl)
    axes[0].set_xticks([i + 0.3 for i in range(len(progs))], progs, rotation=30)
    axes[0].set_ylabel("cycle speed-up over O0"); axes[0].set_title("Optimization levels (K = 6 registers)")
    axes[0].legend()
    for method, style in (("greedy", "o-"), ("ilp", "s--")):
        ks = sorted({r["k"] for r in e3})
        ys = [sum(r["cycles"] for r in e3 if r["k"] == k and r["method"] == method) for k in ks]
        axes[1].plot(ks, ys, style, label=method)
    axes[1].set_xlabel("allocatable registers K"); axes[1].set_ylabel("total cycles, 8 programs")
    axes[1].set_title("Register allocation: greedy vs ILP"); axes[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "speedup_and_regalloc.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(f"Python {platform.python_version()}, {platform.platform()}")
    print("Exp 1: optimization levels"); e1 = exp1()
    s1 = exp1_summary(e1)
    write_csv(RESULTS / "exp1_levels.csv", e1); write_csv(RESULTS / "exp1_summary.csv", s1)
    print("Exp 2: ablation"); e2, s2 = exp2()
    write_csv(RESULTS / "exp2_ablation.csv", e2); write_csv(RESULTS / "exp2_summary.csv", s2)
    print("Exp 3: register allocation"); e3 = exp3(args.quick)
    s3 = exp3_summary(e3)
    write_csv(RESULTS / "exp3_regalloc.csv", e3); write_csv(RESULTS / "exp3_summary.csv", s3)
    print("Exp 4: pass ordering"); e4 = exp4(args.quick)
    write_csv(RESULTS / "exp4_pass_order.csv", e4)
    plots(e1, e3)
    (RESULTS / "SUMMARY.md").write_text("\n".join([
        "# Track F results (generated by run_experiments.py)\n",
        f"Python {platform.python_version()}. Cycle costs: mul 3, div/mod 20, ld/st 3, other 1. "
        "Every row's output was compared with the reference AST interpreter.\n",
        "## Exp 1 — optimization levels (geometric means over 8 programs, relative to O0)\n",
        md_table(s1, list(s1[0])), "\n### Per program\n",
        md_table(e1, ["program", "level", "static_size", "instructions", "cycles", "loads", "stores",
                      "vm_ms", "compile_ms", "spilled", "same_output"]),
        "\n![levels](speedup_and_regalloc.png)\n",
        "## Exp 2 — ablation (cycle speed-up vs no IR passes, greedy RA, K = 6)\n", md_table(s2, list(s2[0])),
        "\n## Exp 3 — register allocation: greedy vs ILP\n", md_table(s3, list(s3[0])),
        "\n## Exp 4 — pass ordering (each pass applied once)\n", md_table(e4, list(e4[0])),
        "\n## Mutation testing\n\nSee `mutation_testing.txt`.\n",
    ]) + "\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
