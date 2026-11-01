"""Track C experiments. Everything runs on PennyLane SIMULATORS (see README).

    python run_experiments.py            # full run (~45 min)
    python run_experiments.py --quick    # smoke run (~4 min)

Exp 1  QAOA depth p = 1..4 on the Track B Max-Cut (10 qubits) vs classical
       baselines (brute force, local search, random cut).
Exp 2  Optimizers under an equal budget of circuit runs: parameter-shift GD,
       Adam, SPSA, QNG, on QAOA (p=2) and VQE (H2, hardware-efficient ansatz);
       exact expectations and 1000 shots; step sizes tuned on a held-out seed.
Exp 3  Finite shots: spread of the cost and of a parameter-shift gradient vs shots.
Exp 4  Hardware noise: depolarizing noise model, QAOA p = 1..3 (8-qubit sub-graph).
Exp 5  Barren plateaus: gradient variance vs qubits, global vs local cost.
Exp 6  VQE dissociation curve of H2 (UCC ansatz) vs Hartree-Fock and exact (FCI).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics as st
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qopt import barren, maxcut as M, train as T, vqe as V

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TRACK_B_EXP4 = HERE.parent / "B_discrete" / "results" / "exp4_maxcut.csv"


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            fields = list(dict.fromkeys(k for r in rows for k in r))   # rows may differ in keys
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def fit_adam(cost, params, steps, lr=0.05):
    """Fast noiseless training (backprop on the simulator) used where the result,
    not the training cost, is what we study (Exp 1, 4)."""
    opt = qml.AdamOptimizer(lr)
    for _ in range(steps):
        params = opt.step(cost, params)
    return params


def shots_to_see_optimum(p_opt, confidence=0.99):
    if p_opt <= 0:
        return math.inf
    if p_opt >= 1:
        return 1
    return math.ceil(math.log(1 - confidence) / math.log(1 - p_opt))


# ---------------------------------------------------------------- Exp 1
def exp1(mc, quick):
    seeds = range(2) if quick else range(5)
    steps = 80 if quick else 250
    probs = M.probs_qnode(mc)
    cost, _ = M.cost_qnode(mc, diff_method="backprop")
    rows, best_params = [], {}
    for p in range(1, 5):
        stats = []
        for seed in seeds:
            params = fit_adam(cost, M.init_params(p, seed), steps)
            s = mc.stats_from_probs(probs(params))
            stats.append((s, params))
        ratios = [s["approx_ratio"] for s, _ in stats]
        popts = [s["p_optimal"] for s, _ in stats]
        best_params[p] = max(stats, key=lambda t: t[0]["approx_ratio"])[1]
        rows.append(dict(method=f"QAOA p={p}", runs=len(stats),
                         approx_ratio=f"{st.mean(ratios):.3f} ± {st.pstdev(ratios):.3f}",
                         p_optimal=f"{st.mean(popts):.3f}",
                         shots_for_99pct_to_see_optimum=shots_to_see_optimum(st.mean(popts)),
                         params=2 * p))
        print(f"   QAOA p={p}: ratio {st.mean(ratios):.3f}, P(opt) {st.mean(popts):.3f}", flush=True)
    ls_mean = None
    if TRACK_B_EXP4.exists():
        for r in csv.DictReader(open(TRACK_B_EXP4)):
            if r["method"].startswith("one-flip"):
                ls_mean = float(r["mean_cut"]) / mc.optimal_cut
                ls_hit = r["hit_optimum"]
    rows += [dict(method="brute force (classical, exact)", runs=1, approx_ratio="1.000",
                  p_optimal="1.000", shots_for_99pct_to_see_optimum="n/a", params=""),
             dict(method="one-flip local search (Track B)", runs="200",
                  approx_ratio=f"{ls_mean:.3f}" if ls_mean else "see Track B",
                  p_optimal=f"{ls_hit}" if ls_mean else "", shots_for_99pct_to_see_optimum="n/a",
                  params=""),
             dict(method="uniformly random cut", runs="exact",
                  approx_ratio=f"{mc.total_weight / 2 / mc.optimal_cut:.3f}",
                  p_optimal=f"{len(mc.optimal_set) / 2 ** mc.n:.4f}",
                  shots_for_99pct_to_see_optimum=shots_to_see_optimum(len(mc.optimal_set) / 2 ** mc.n),
                  params="")]
    return rows, best_params


# ---------------------------------------------------------------- Exp 2
def exp2(mc, quick):
    H, nq = V.h2_hamiltonian()
    problems = {
        "QAOA p=2 (10 qubits)": dict(
            make=lambda shots, seed: M.cost_qnode(mc, shots=shots, seed=seed),
            exact=M.cost_qnode(mc, diff_method="backprop")[0],
            init=lambda seed: M.init_params(2, seed), budget=1500 if quick else 4000,
            target=-mc.optimal_cut),
        "VQE H2, HEA 2 layers (4 qubits)": dict(
            make=lambda shots, seed: V.energy_qnode(H, nq, "hea", shots=shots, seed=seed),
            exact=V.energy_qnode(H, nq, "hea", diff_method="backprop")[0],
            init=lambda seed: V.init_params("hea", nq, seed=seed), budget=800 if quick else 2500,
            target=V.exact_energy(H, nq)),
    }
    grids = {"gd": [0.003, 0.01, 0.03], "adam": [0.01, 0.03, 0.1],
             "spsa": [0.1, 0.3, 1.0], "qng": [0.01, 0.03, 0.1]}
    if quick:
        grids = {m: g[1:2] for m, g in grids.items()}
    seeds = range(1) if quick else range(3)
    rows, curves, tuned = [], {}, {}
    for pname, P in problems.items():
        for method in T.METHODS:
            every = 20 if method == "spsa" else 1
            # tune on held-out seed 99, exact expectations
            best = None
            for lr in grids[method]:
                cost, dev = P["make"](None, 99)
                _, h = T.train(cost, dev, P["init"](99), method, lr, P["budget"], P["exact"], every)
                if best is None or h[-1][1] < best[1]:
                    best = (lr, h[-1][1])
            lr = best[0]
            tuned[(pname, method)] = lr
            for shots in [None, 1000]:
                finals = []
                for seed in seeds:
                    cost, dev = P["make"](shots, seed)
                    _, h = T.train(cost, dev, P["init"](seed), method, lr, P["budget"], P["exact"], every)
                    finals.append(h[-1][1])
                    curves.setdefault((pname, method, shots), []).append(h)
                rows.append(dict(problem=pname, method=method, stepsize=lr,
                                 shots="exact" if shots is None else shots,
                                 budget_circuits=P["budget"], seeds=len(finals),
                                 final_cost_mean=round(st.mean(finals), 4),
                                 final_cost_std=round(st.pstdev(finals), 4),
                                 target=round(P["target"], 4),
                                 gap_mean=round(st.mean(finals) - P["target"], 4)))
                print(f"   {pname} | {method:4s} lr={lr} shots={shots}: "
                      f"{st.mean(finals):.4f} ± {st.pstdev(finals):.4f} (target {P['target']:.4f})",
                      flush=True)
    return rows, curves


def exp2_plot(curves, path):
    problems = sorted({k[0] for k in curves}, reverse=True)
    fig, axes = plt.subplots(1, len(problems), figsize=(12, 4.2))
    for ax, pname in zip(np.atleast_1d(axes), problems):
        for method in T.METHODS:
            hs = curves.get((pname, method, None))
            if not hs:
                continue
            grid = np.linspace(0, max(h[-1][0] for h in hs), 60)
            ys = [np.interp(grid, [u for u, _ in h], [c for _, c in h]) for h in hs]
            mean, std = np.mean(ys, axis=0), np.std(ys, axis=0)
            ax.plot(grid, mean, label=method)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        ax.set_title(pname, fontsize=10)
        ax.set_xlabel("circuits run (budget)")
        ax.set_ylabel("exact cost (lower is better)")
        ax.legend(fontsize=8)
    fig.suptitle("Optimizers at equal quantum cost (exact expectations; band = ±1 std over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 3
def exp3(mc, params, quick):
    shots_list = [10, 100, 1000, 10000] if quick else [10, 30, 100, 300, 1000, 3000, 10000]
    repeats = 30 if quick else 100
    exact_cost, _ = M.cost_qnode(mc, diff_method="backprop")
    exact = float(exact_cost(params))
    # The gradient is studied at the untrained starting point: at the trained
    # parameters it is ~0, so "wrong sign" would be meaningless there.
    start = M.init_params(1, seed=0)
    exact_grad = qml.grad(exact_cost)(start)[0, 0]
    rows = []
    for shots in shots_list:
        f, _ = M.cost_qnode(mc, shots=shots, seed=12345 + shots)
        vals = [float(f(params)) for _ in range(repeats)]
        row = dict(shots=shots, repeats=repeats, exact_cost=round(exact, 4),
                   cost_mean=round(st.mean(vals), 4), cost_std=round(st.pstdev(vals), 4),
                   std_times_sqrt_shots=round(st.pstdev(vals) * math.sqrt(shots), 3))
        if shots in (100, 1000, 10000):
            g = [float(qml.grad(f)(start)[0, 0]) for _ in range(max(10, repeats // 5))]
            row.update(grad_exact_at_start=round(float(exact_grad), 4), grad_std=round(st.pstdev(g), 4),
                       grad_sign_wrong=f"{sum(np.sign(x) != np.sign(exact_grad) for x in g)}/{len(g)}")
        rows.append(row)
        print(f"   shots={shots}: std {row['cost_std']}", flush=True)
    return rows


def exp3_plot(rows, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["shots"] for r in rows]
    ax.loglog(xs, [r["cost_std"] for r in rows], "o-", label="measured std of cost estimate")
    ref = rows[-1]["cost_std"] * math.sqrt(rows[-1]["shots"])
    ax.loglog(xs, [ref / math.sqrt(x) for x in xs], "--", label=r"$\propto 1/\sqrt{\rm shots}$")
    ax.set_xlabel("shots per circuit"); ax.set_ylabel("std of <H_C> estimate")
    ax.set_title("Finite-shot noise, QAOA p=1 (10 qubits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 4
def exp4(mc, quick):
    sub = mc.induced(8)
    levels = [0.0, 0.003, 0.01, 0.03] if quick else [0.0, 0.001, 0.003, 0.01, 0.02, 0.05]
    depths = [1, 2] if quick else [1, 2, 3]
    cost, _ = M.cost_qnode(sub, diff_method="backprop")
    rows = []
    for p in depths:
        params = max((fit_adam(cost, M.init_params(p, s), 80 if quick else 250) for s in range(3)),
                     key=lambda x: -float(cost(x)))
        for lvl in levels:
            probs = M.probs_qnode(sub, device="default.mixed", noise=lvl)
            s = sub.stats_from_probs(probs(params))
            rows.append(dict(qaoa_depth=p, noise=lvl, approx_ratio=round(s["approx_ratio"], 4),
                             p_optimal=round(s["p_optimal"], 4)))
            print(f"   p={p} noise={lvl}: ratio {s['approx_ratio']:.3f}", flush=True)
    info = dict(n=sub.n, edges=len(sub.edges), optimal_cut=sub.optimal_cut,
                random_ratio=round(sub.total_weight / 2 / sub.optimal_cut, 4))
    return rows, info


def exp4_plot(rows, info, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p in sorted({r["qaoa_depth"] for r in rows}):
        pts = [(r["noise"], r["approx_ratio"]) for r in rows if r["qaoa_depth"] == p]
        ax.plot(*zip(*pts), "o-", label=f"QAOA p={p}")
    ax.axhline(info["random_ratio"], ls=":", c="grey", label="random cut")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(left=0)
    ax.set_xlabel("depolarizing probability per gate (simulated noise model)")
    ax.set_ylabel("approximation ratio")
    ax.set_title(f"Noise vs depth, {info['n']}-node sub-graph")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 5
def exp5(quick):
    ns = [2, 4, 6, 8] if quick else [2, 4, 6, 8, 10, 12]
    samples = 60 if quick else 200
    rows = []
    for n in ns:
        for cost, layers, label in [("global", n, "global cost, L = n"),
                                    ("local", n, "local cost, L = n"),
                                    ("local", 2, "local cost, L = 2")]:
            var, mean = barren.gradient_variance(n, layers, cost, samples, seed=n)
            rows.append(dict(qubits=n, layers=layers, cost=cost, series=label,
                             samples=samples, grad_mean=f"{mean:.2e}", grad_variance=f"{var:.3e}"))
        print(f"   n={n} done", flush=True)
    return rows


def exp5_plot(rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label in dict.fromkeys(r["series"] for r in rows):
        pts = [(r["qubits"], float(r["grad_variance"])) for r in rows if r["series"] == label]
        ax.semilogy(*zip(*pts), "o-", label=label)
    ax.set_xlabel("qubits n"); ax.set_ylabel(r"Var[$\partial C/\partial\theta_0$] (log)")
    ax.set_title("Barren plateaus: random hardware-efficient circuits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 6
def exp6(quick):
    bonds = [0.5, 0.7414, 1.5] if quick else [0.3, 0.5, 0.7414, 1.0, 1.5, 2.0, 2.5]
    rows = []
    for b in bonds:
        H, nq = V.h2_hamiltonian(b)
        exact = V.exact_energy(H, nq)
        e, dev = V.energy_qnode(H, nq, "ucc")
        params = V.init_params("ucc", nq)
        hf = float(e(params))
        opt = qml.GradientDescentOptimizer(0.4)
        with qml.Tracker(dev) as tr:
            for _ in range(60):
                params = opt.step(e, params)
        vqe = float(e(params))
        rows.append(dict(bond_angstrom=b, hartree_fock=round(hf, 6), vqe_ucc=round(vqe, 6),
                         exact_fci=round(exact, 6), vqe_error_mHa=round(1000 * (vqe - exact), 4),
                         hf_error_mHa=round(1000 * (hf - exact), 2),
                         chemical_accuracy=abs(vqe - exact) < 1.6e-3,
                         theta=round(float(params[0]), 4),
                         circuits=int(tr.totals.get("simulations", 0))))
        print(f"   d={b}: VQE {vqe:.6f}, exact {exact:.6f}, HF {hf:.6f}", flush=True)
    return rows


def exp6_plot(rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = [r["bond_angstrom"] for r in rows]
    ax.plot(d, [r["hartree_fock"] for r in rows], "s--", label="Hartree–Fock (classical)")
    ax.plot(d, [r["exact_fci"] for r in rows], "-", lw=3, alpha=0.4, label="exact / FCI (classical)")
    ax.plot(d, [r["vqe_ucc"] for r in rows], "o", label="VQE, UCC ansatz (simulated)")
    ax.set_xlabel("H–H bond length (Å)"); ax.set_ylabel("energy (Hartree)")
    ax.set_title("H₂ dissociation curve, STO-3G")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- summary
def read_csv(name):
    path = RESULTS / name
    return list(csv.DictReader(open(path))) if path.exists() else None


def build_summary():
    """Rebuild SUMMARY.md from whatever CSVs exist, so one experiment can be rerun alone."""
    meta = json.loads((RESULTS / "meta.json").read_text()) if (RESULTS / "meta.json").exists() else {}
    out = ["# Track C results (generated by run_experiments.py)\n",
           f"PennyLane {qml.__version__}. **All results are classical simulations** (`default.qubit` "
           "state vector; `default.mixed` density matrix for noise; `lightning.qubit` for barren "
           "plateaus). No real quantum hardware was used.\n"]
    sections = [
        ("exp1_qaoa_depth.csv",
         f"## Exp 1 — QAOA on the Track B Max-Cut ({meta.get('n', '?')} nodes, optimum "
         f"{meta.get('optimal_cut', '?')})\n\nTrained noiselessly with Adam (backprop on the simulator); "
         "mean ± std over seeds. `shots_for_99pct…` = shots needed to sample an optimal cut at least "
         "once with 99% probability.\n",
         ["method", "runs", "params", "approx_ratio", "p_optimal", "shots_for_99pct_to_see_optimum"], None),
        ("exp2_optimizers.csv",
         "## Exp 2 — optimizers at an equal budget of circuit runs\n\nBudget counts distinct circuits run "
         "(`qml.Tracker` 'simulations'). Step size per method tuned on held-out seed 99 with exact "
         "expectations, then fixed. Cost = exact expectation at the final parameters; lower is better.\n",
         ["problem", "method", "stepsize", "shots", "budget_circuits", "seeds", "final_cost_mean",
          "final_cost_std", "target", "gap_mean"], "exp2_optimizers.png"),
        ("exp3_shots.csv",
         "## Exp 3 — finite measurement shots (QAOA p = 1)\n\nCost spread measured at the trained "
         "parameters; gradient (d/dγ) spread measured at the untrained start, where the true gradient is "
         "non-zero.\n",
         ["shots", "repeats", "exact_cost", "cost_mean", "cost_std", "std_times_sqrt_shots",
          "grad_exact_at_start", "grad_std", "grad_sign_wrong"], "exp3_shots.png"),
        ("exp4_noise.csv",
         f"## Exp 4 — hardware noise model ({meta.get('noise_n', '?')}-node induced sub-graph, "
         f"{meta.get('noise_edges', '?')} edges, optimum {meta.get('noise_optimal_cut', '?')})\n\n"
         "Simulated depolarizing channel after every gate: a simplified model, not a calibrated device. "
         f"Random-cut ratio = {meta.get('noise_random_ratio', '?')}.\n",
         ["qaoa_depth", "noise", "approx_ratio", "p_optimal"], "exp4_noise.png"),
        ("exp5_barren.csv", "## Exp 5 — barren plateaus\n",
         ["qubits", "layers", "cost", "samples", "grad_mean", "grad_variance"], "exp5_barren.png"),
        ("exp6_vqe_h2.csv",
         "## Exp 6 — VQE for H₂ (UCC ansatz, parameter-shift gradient descent, 60 steps)\n\n"
         "Chemical accuracy = within 1.6 mHa of exact.\n",
         ["bond_angstrom", "hartree_fock", "vqe_ucc", "exact_fci", "vqe_error_mHa", "hf_error_mHa",
          "chemical_accuracy", "circuits"], "exp6_vqe_h2.png"),
    ]
    for name, head, cols, img in sections:
        rows = read_csv(name)
        if rows:
            out += [head, md_table(rows, cols), f"\n![{name}]({img})\n" if img else ""]
    (RESULTS / "SUMMARY.md").write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", type=int, help="run only these experiment numbers")
    args = ap.parse_args()
    run = lambda k: not args.only or k in args.only
    RESULTS.mkdir(exist_ok=True)
    print(f"Python {platform.python_version()}, PennyLane {qml.__version__}, {platform.platform()}")
    mc = M.MaxCut.from_track_b()
    meta_path = RESULTS / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(n=mc.n, optimal_cut=mc.optimal_cut)

    if run(1) or run(3):
        print("Exp 1: QAOA depth vs classical baselines")
        r1, best = exp1(mc, args.quick)
        write_csv(RESULTS / "exp1_qaoa_depth.csv", r1)
    if run(2):
        print("Exp 2: optimizers at equal circuit budget")
        r2, curves = exp2(mc, args.quick)
        write_csv(RESULTS / "exp2_optimizers.csv", r2)
        exp2_plot(curves, RESULTS / "exp2_optimizers.png")
    if run(3):
        print("Exp 3: finite shots")
        r3 = exp3(mc, best[1], args.quick)
        write_csv(RESULTS / "exp3_shots.csv", r3)
        exp3_plot(r3, RESULTS / "exp3_shots.png")
    if run(4):
        print("Exp 4: noise")
        r4, info4 = exp4(mc, args.quick)
        write_csv(RESULTS / "exp4_noise.csv", r4)
        exp4_plot(r4, info4, RESULTS / "exp4_noise.png")
        meta.update(noise_n=info4["n"], noise_edges=info4["edges"],
                    noise_optimal_cut=info4["optimal_cut"], noise_random_ratio=info4["random_ratio"])
    if run(5):
        print("Exp 5: barren plateaus")
        r5 = exp5(args.quick)
        write_csv(RESULTS / "exp5_barren.csv", r5)
        exp5_plot(r5, RESULTS / "exp5_barren.png")
    if run(6):
        print("Exp 6: VQE dissociation curve")
        r6 = exp6(args.quick)
        write_csv(RESULTS / "exp6_vqe_h2.csv", r6)
        exp6_plot(r6, RESULTS / "exp6_vqe_h2.png")

    meta_path.write_text(json.dumps(meta, indent=2))
    build_summary()
    with open(RESULTS / "environment.txt", "a") as f:
        f.write(f"python {platform.python_version()} | pennylane {qml.__version__} | "
                f"{platform.platform()} | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"mode {'quick' if args.quick else 'full'} | experiments {args.only or 'all'} | "
                "hardware: none (simulation only)\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
