"""Track I experiments. All quantum results are CLASSICAL SIMULATIONS (PennyLane).

    python run_experiments.py              # full run (~25 min)
    python run_experiments.py --quick      # smoke run (~4 min)
    python run_experiments.py --only 2 6   # rerun some; SUMMARY.md is rebuilt from CSVs

Exp 1  Every algorithm taught: DJ, BV, QFT, QPE, Grover, Shor (N = 15), with
       query counts against the best classical deterministic algorithm.
Exp 2  Grover vs linear search: oracle queries, gate counts, wall-clock.
Exp 3  Hardware noise model: Grover and BV under depolarizing noise.
Exp 4  Quantum annealing (exact simulation) vs simulated annealing vs brute force
       on the Track B Max-Cut.
Exp 5  Embedding QUBOs on annealing hardware (Pegasus / Chimera): qubit overhead.
Exp 6  QML vs classical: variational classifier and quantum kernel vs logistic
       regression, RBF SVM and an MLP; accuracy, time, parameters, circuit evaluations.
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
from sklearn.metrics import accuracy_score

from qalgo import algorithms as A, annealing as AN, qml_models as QM

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def read_csv(name):
    p = RESULTS / name
    return list(csv.DictReader(open(p))) if p.exists() else None


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------- Exp 1
def exp1(quick):
    rows = []
    rng = np.random.default_rng(0)
    for n in ([2, 4] if quick else [2, 3, 4, 5, 6]):
        s = "".join(rng.choice(["0", "1"], n))
        s = s if "1" in s else "1" + s[1:]           # balanced oracle needs s != 0
        pc = A.deutsch_jozsa("constant1", n)
        pb = A.deutsch_jozsa(f"balanced:{s}", n)
        rows.append(dict(algorithm="Deutsch–Jozsa", n=n, quantum_queries=1,
                         classical_queries_deterministic=2 ** (n - 1) + 1,
                         result=f"P(0..0): constant {pc:.3f}, balanced {pb:.3f}", correct=pc > .999 and pb < 1e-6))
    for n in ([3, 6] if quick else [2, 4, 6, 8]):
        s = "".join(rng.choice(["0", "1"], n))
        s = s if "1" in s else "1" + s[1:]           # an all-zero secret would prove nothing
        p = A.bernstein_vazirani(s)
        rows.append(dict(algorithm="Bernstein–Vazirani", n=n, quantum_queries=1,
                         classical_queries_deterministic=n,
                         result=f"secret {s} recovered with P = {p[int(s, 2)]:.3f}", correct=p[int(s, 2)] > .999))
    for n in [3, 5]:
        lib = qml.matrix(qml.QFT(wires=range(n)))
        mine = qml.matrix(lambda: A.qft(range(n)), wire_order=range(n))()
        rows.append(dict(algorithm="QFT (hand-written)", n=n, quantum_queries="",
                         classical_queries_deterministic="",
                         result=f"max |mine - qml.QFT| = {np.abs(mine - lib).max():.1e}",
                         correct=bool(np.allclose(mine, lib))))
    for t, phase in [(3, 0.625), (5, 0.3), (8, 1 / 3)]:
        U = np.diag([1, np.exp(2j * np.pi * phase)])
        pr = A.qpe(U, lambda w: qml.PauliX(w[0]), t, 1)
        est = np.argmax(pr) / 2 ** t
        rows.append(dict(algorithm="QPE (hand-written)", n=t + 1, quantum_queries=2 ** t - 1,
                         classical_queries_deterministic="",
                         result=f"phase {phase:.4f} -> {est:.4f} (P = {pr.max():.3f}, resolution 1/{2 ** t})",
                         correct=abs(est - phase) <= 1 / 2 ** t))
    for n in [3, 5] if quick else [3, 4, 5, 6]:
        target = "1" * n
        p = A.grover(target)
        rows.append(dict(algorithm="Grover (hand-written)", n=n, quantum_queries=A.grover_iterations(n),
                         classical_queries_deterministic=2 ** n - 1,
                         result=f"P(target) = {p:.3f}", correct=p > 0.9))
    for a in [2, 7, 11, 13, 14]:
        r = A.shor_factor(15, a)
        found = sorted({tuple(sorted(o["factors"])) for o in r["outcomes"] if o["factors"]})
        rows.append(dict(algorithm="Shor N=15 (order finding by QPE)", n=8 + 4, quantum_queries="",
                         classical_queries_deterministic="",
                         result=f"a={a}: order {r['true_order']}, P(factors per run) = {r['p_success']:.2f}, "
                                f"factors {found or 'none (a^(r/2) = -1 mod 15)'}",
                         correct=(a == 14 and r["p_success"] == 0) or (found == [(3, 5)])))
    for r in rows:
        print(f"   {r['algorithm']:34s} n={r['n']}: {r['result']}", flush=True)
    return rows


# ---------------------------------------------------------------- Exp 2
def exp2(quick):
    ns = [2, 3, 4, 5, 6, 7, 8] if quick else [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    rng = np.random.default_rng(0)
    rows = []
    for n in ns:
        N = 2 ** n
        target = "".join(rng.choice(["0", "1"], n))
        p, qsec = timed(A.grover, target)
        # classical linear search on an unsorted list: count oracle calls
        trials = 2000
        tgt = rng.integers(0, N, trials)
        queries = tgt + 1                                   # scan 0,1,2,... until found
        t0 = time.perf_counter()
        for t in tgt[:200]:
            for i in range(N):
                if i == t:
                    break
        csec = (time.perf_counter() - t0) / 200
        row = dict(n_qubits=n, N=N, grover_queries=A.grover_iterations(n), grover_p_success=round(p, 4),
                   linear_mean_queries=round(float(queries.mean()), 1), linear_worst_queries=N,
                   query_ratio=round(float(queries.mean()) / A.grover_iterations(n), 2),
                   grover_sim_seconds=float(f"{qsec:.3g}"), linear_seconds=float(f"{csec:.3g}"))
        if n <= (6 if quick else 8):
            g = A.grover_gate_counts(n)
            row.update(grover_gates_decomposed=g["total_gates"], grover_2q_gates=g["two_qubit_gates"])
        rows.append(row)
        print(f"   n={n}: Grover {row['grover_queries']} queries (P={p:.3f}), linear mean "
              f"{row['linear_mean_queries']}, sim {qsec:.3g}s", flush=True)
    return rows


def exp2_plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    N = [int(r["N"]) for r in rows]
    ax = axes[0]
    ax.loglog(N, [float(r["linear_mean_queries"]) for r in rows], "o-", label="linear search, mean (measured)")
    ax.loglog(N, [int(r["grover_queries"]) for r in rows], "s-", label="Grover iterations")
    ax.loglog(N, [math.pi / 4 * math.sqrt(x) for x in N], ":", c="grey", label=r"$\frac{\pi}{4}\sqrt{N}$")
    ax.set_xlabel("search space size N"); ax.set_ylabel("oracle queries"); ax.set_title("Queries: quadratic advantage")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.loglog(N, [float(r["grover_sim_seconds"]) for r in rows], "s-", label="Grover, simulated on CPU")
    ax.loglog(N, [float(r["linear_seconds"]) for r in rows], "o-", label="linear search, Python")
    ax.set_xlabel("search space size N"); ax.set_ylabel("wall-clock seconds")
    ax.set_title("Wall-clock: simulation is slower")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 3
def exp3(quick):
    levels = [0.0, 0.005, 0.02] if quick else [0.0, 0.001, 0.003, 0.01, 0.02, 0.05]
    rows = []
    for n in ([3, 4] if quick else [3, 4, 5, 6]):
        target = "1" * n
        for lvl in levels:
            rows.append(dict(algorithm="Grover", n=n, noise=lvl, p_success=round(A.grover(target, noise=lvl), 4),
                             random_guess=round(1 / 2 ** n, 4)))
        print(f"   Grover n={n} done", flush=True)
    for n in ([4] if quick else [4, 6, 8]):
        s = ("10" * n)[:n]
        for lvl in levels:
            rows.append(dict(algorithm="Bernstein–Vazirani", n=n, noise=lvl,
                             p_success=round(float(A.bernstein_vazirani(s, noise=lvl)[int(s, 2)]), 4),
                             random_guess=round(1 / 2 ** n, 4)))
        print(f"   BV n={n} done", flush=True)
    # does noise change the best number of Grover iterations?
    n = 5
    for lvl in ([0.0, 0.02] if quick else [0.0, 0.01, 0.03]):
        for k in range(1, 7):
            rows.append(dict(algorithm=f"Grover n={n}, k iterations", n=n, noise=lvl, iterations=k,
                             p_success=round(A.grover("1" * n, iterations=k, noise=lvl), 4),
                             random_guess=round(1 / 2 ** n, 4)))
    return rows


def exp3_plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for alg in ["Grover", "Bernstein–Vazirani"]:
        for n in sorted({int(r["n"]) for r in rows if r["algorithm"] == alg}):
            pts = [(float(r["noise"]), float(r["p_success"])) for r in rows
                   if r["algorithm"] == alg and int(r["n"]) == n]
            ax.plot(*zip(*pts), "o-" if alg == "Grover" else "s--", label=f"{alg} n={n}")
    ax.set_xscale("symlog", linthresh=1e-3); ax.set_xlim(left=0)
    ax.set_xlabel("depolarizing probability per gate"); ax.set_ylabel("P(correct answer)")
    ax.set_title("Noise vs circuit size"); ax.legend(fontsize=7)
    ax = axes[1]
    k_rows = [r for r in rows if "iterations" in r["algorithm"]]
    for lvl in sorted({float(r["noise"]) for r in k_rows}):
        pts = [(int(r["iterations"]), float(r["p_success"])) for r in k_rows if float(r["noise"]) == lvl]
        ax.plot(*zip(*pts), "o-", label=f"noise {lvl}")
    ax.set_xlabel("Grover iterations k (n = 5)"); ax.set_ylabel("P(target)")
    ax.set_title("Noise shifts the best k"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 4
def exp4(quick):
    n, e = AN.load_maxcut()
    rows = []
    (cut, bf_sec) = timed(AN.cut_values, n, e)
    rows.append(dict(method="brute force (classical)", setting="all 2^10 states", p_optimal=1.0,
                     approx_ratio=1.0, seconds=float(f"{bf_sec:.3g}")))
    for T in ([1, 10, 100, 1000] if quick else [1, 3, 10, 30, 100, 300, 1000, 3000]):
        r, sec = timed(AN.quantum_anneal, n, e, T)
        rows.append(dict(method="quantum annealing (exact simulation)", setting=f"anneal time T = {T}",
                         p_optimal=round(r["p_optimal"], 4), approx_ratio=round(r["approx_ratio"], 4),
                         seconds=float(f"{sec:.3g}")))
        print(f"   QA T={T}: P(opt) {r['p_optimal']:.3f}", flush=True)
    for sweeps in ([10, 100] if quick else [3, 10, 30, 100, 300, 1000]):
        r = AN.simulated_annealing(n, e, num_reads=100, num_sweeps=sweeps)
        rows.append(dict(method="simulated annealing (classical)", setting=f"{sweeps} sweeps, 100 reads",
                         p_optimal=round(r["p_optimal"], 4), approx_ratio=round(r["approx_ratio"], 4),
                         seconds=float(f"{r['seconds'] / 100:.3g}")))
        print(f"   SA sweeps={sweeps}: P(opt) {r['p_optimal']:.3f}", flush=True)
    gap, s_at = AN.spectral_gap(n, e, 30 if quick else 80)
    meta = dict(qa_min_gap=round(gap, 4), qa_gap_at_s=round(s_at, 3), qa_adiabatic_time_scale=round(1 / gap ** 2, 1))
    print(f"   min gap {gap:.4f} at s={s_at:.2f}", flush=True)
    return rows, meta


def exp4_plot(rows, meta, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    qa = [r for r in rows if r["method"].startswith("quantum")]
    T = [float(r["setting"].split("=")[1]) for r in qa]
    ax.semilogx(T, [float(r["p_optimal"]) for r in qa], "o-", label="quantum annealing (simulated)")
    ax.axvline(meta["qa_adiabatic_time_scale"], ls=":", c="grey", label=r"1/gap$^2$")
    ax.axhline(8 / 1024, ls="--", c="lightgrey", label="random guess")
    ax.set_xlabel("anneal time T (units of 1/energy scale)"); ax.set_ylabel("P(optimal cut)")
    ax.set_title("Adiabatic theorem: slow anneals find the optimum")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 5
def exp5(quick):
    rows = []
    timeout = 10 if quick else 30
    hws = {"pegasus_p6": AN.hardware_graph("pegasus_p6"), "chimera_c16": AN.hardware_graph("chimera_c16")}
    n, e = AN.load_maxcut()
    cases = [("Track B Max-Cut (10 nodes, 19 edges)", [(i, j) for i, j, _ in e], n)]
    for k in ([5, 15, 25] if quick else [5, 10, 15, 20, 25, 30, 35, 40]):
        cases.append((f"complete graph K{k}", AN.complete_graph_edges(k), k))
    for items in ([4, 10] if quick else [4, 8, 12, 16, 20]):
        edges, m = AN.knapsack_qubo_edges(items, capacity=25 * items)
        cases.append((f"knapsack QUBO, {items} items (+{m - items} slack)", edges, m))
    for hw_name, hw in hws.items():
        for label, edges, logical in cases:
            r, sec = timed(AN.embed, edges, hw, 0, timeout, 2 if not quick else 1)
            rows.append(dict(hardware=f"{hw_name} ({hw.number_of_nodes()} qubits)", problem=label,
                             logical_vars=logical, logical_couplings=len(edges),
                             physical_qubits=r["physical_qubits"] if r else "no embedding found",
                             overhead=round(r["physical_qubits"] / logical, 2) if r else "",
                             max_chain=r["max_chain"] if r else "", seconds=round(sec, 1)))
            print(f"   {hw_name} {label}: {rows[-1]['physical_qubits']}", flush=True)
    return rows


def exp5_plot(rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for hw in dict.fromkeys(r["hardware"] for r in rows):
        pts = [(int(r["logical_vars"]), int(r["physical_qubits"])) for r in rows
               if r["hardware"] == hw and r["problem"].startswith("complete")
               and str(r["physical_qubits"]).isdigit()]
        if pts:
            ax.plot(*zip(*pts), "o-", label=f"{hw}: complete graphs")
    xs = range(5, 41)
    ax.plot(xs, xs, ":", c="grey", label="1 physical qubit per variable")
    ax.set_xlabel("logical variables (fully connected QUBO)"); ax.set_ylabel("physical qubits used")
    ax.set_title("Minor-embedding overhead on annealing hardware graphs")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- Exp 6
def exp6(quick):
    seeds = range(2) if quick else range(5)
    n_samples = 100 if quick else 200
    raw = []
    for ds in ["moons", "cancer"]:
        for seed in seeds:
            Xtr, Xte, ytr, yte = QM.load(ds, n_samples, seed)
            nf = Xtr.shape[1]
            for name, m in QM.classical_models(seed).items():
                _, sec = timed(m.fit, Xtr, ytr)
                raw.append(dict(dataset=ds, seed=seed, model=name, kind="classical",
                                accuracy=accuracy_score(yte, m.predict(Xte)),
                                train_seconds=sec, params=QM.classical_param_count(name, m),
                                circuit_evals_train=0, circuit_evals_test=0))
            v = QM.VQC(nf, layers=3, steps=30 if quick else 60, seed=seed).fit(Xtr, ytr)
            acc = accuracy_score(yte, v.predict(Xte))
            raw.append(dict(dataset=ds, seed=seed, model="variational classifier (VQC)", kind="quantum (simulated)",
                            accuracy=acc, train_seconds=v.train_seconds, params=v.n_params,
                            circuit_evals_train=v.circuit_evals, circuit_evals_test=v.circuit_evals_predict))
            k = QM.QuantumKernelSVM(nf).fit(Xtr, ytr)
            acc = accuracy_score(yte, k.predict(Xte))
            raw.append(dict(dataset=ds, seed=seed, model="quantum kernel + SVM", kind="quantum (simulated)",
                            accuracy=acc, train_seconds=k.train_seconds, params=k.n_params,
                            circuit_evals_train=k.circuit_evals, circuit_evals_test=k.circuit_evals_predict))
            print(f"   {ds} seed {seed}: " + ", ".join(f"{r['model'].split(' (')[0]} {r['accuracy']:.2f}"
                                                      for r in raw[-5:]), flush=True)
    summary = []
    for ds in ["moons", "cancer"]:
        for model in dict.fromkeys(r["model"] for r in raw):
            rs = [r for r in raw if r["dataset"] == ds and r["model"] == model]
            acc = [r["accuracy"] for r in rs]
            summary.append(dict(dataset=ds, model=model, kind=rs[0]["kind"], seeds=len(rs),
                                accuracy=f"{st.mean(acc):.3f} ± {st.pstdev(acc):.3f}",
                                train_seconds=float(f"{st.mean(r['train_seconds'] for r in rs):.3g}"),
                                params=round(st.mean(r["params"] for r in rs)),
                                circuit_evals_train=round(st.mean(r["circuit_evals_train"] for r in rs)),
                                circuit_evals_test=round(st.mean(r["circuit_evals_test"] for r in rs))))
    return raw, summary


# ---------------------------------------------------------------- summary
def build_summary():
    meta = json.loads((RESULTS / "meta.json").read_text()) if (RESULTS / "meta.json").exists() else {}
    out = ["# Track I results (generated by run_experiments.py)\n",
           f"PennyLane {qml.__version__}. **All quantum results are classical simulations**; no real "
           "quantum hardware was used. 'Noise' means a simulated depolarizing noise model.\n"]
    sections = [
        ("exp1_algorithms.csv", "## Exp 1 — every algorithm taught\n\n`classical_queries_deterministic` = worst case "
         "for the best deterministic classical algorithm.\n",
         ["algorithm", "n", "quantum_queries", "classical_queries_deterministic", "result", "correct"], None),
        ("exp2_grover_vs_linear.csv", "## Exp 2 — Grover vs linear search\n\nClassical queries measured over 2,000 "
         "random targets. Gate counts after decomposition into 1- and 2-qubit gates (no ancilla qubits).\n",
         ["n_qubits", "N", "grover_queries", "grover_p_success", "linear_mean_queries", "linear_worst_queries",
          "query_ratio", "grover_gates_decomposed", "grover_2q_gates", "grover_sim_seconds", "linear_seconds"],
         "exp2_grover_vs_linear.png"),
        ("exp3_noise.csv", "## Exp 3 — hardware noise model\n\nDepolarizing channel after every *logical* gate "
         "(the multi-controlled Z counts as one gate; on hardware it decomposes into many, so real error "
         "would be larger).\n",
         ["algorithm", "n", "noise", "iterations", "p_success", "random_guess"], "exp3_noise.png"),
        ("exp4_annealing.csv", f"## Exp 4 — annealing on the Track B Max-Cut (10 nodes, optimum 14)\n\n"
         f"Minimum spectral gap {meta.get('qa_min_gap', '?')} at s = {meta.get('qa_gap_at_s', '?')}, so the "
         f"adiabatic time scale 1/gap² ≈ {meta.get('qa_adiabatic_time_scale', '?')}. Seconds for simulated "
         "annealing are per read.\n",
         ["method", "setting", "p_optimal", "approx_ratio", "seconds"], "exp4_annealing.png"),
        ("exp5_embedding.csv", "## Exp 5 — embedding QUBOs on annealing hardware graphs (minorminer)\n",
         ["hardware", "problem", "logical_vars", "logical_couplings", "physical_qubits", "overhead",
          "max_chain"], "exp5_embedding.png"),
        ("exp6_qml_summary.csv", "## Exp 6 — QML vs classical (same splits, test accuracy, mean ± std over seeds)\n\n"
         "Quantum training time is the time to *simulate* the circuits. VQC circuit evaluations are the "
         "parameter-shift count real hardware would need; kernel evaluations are one circuit per pair of points.\n",
         ["dataset", "model", "kind", "accuracy", "train_seconds", "params", "circuit_evals_train",
          "circuit_evals_test"], None),
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
    ap.add_argument("--only", nargs="*", type=int)
    args = ap.parse_args()
    run = lambda k: not args.only or k in args.only
    RESULTS.mkdir(exist_ok=True)
    meta_path = RESULTS / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    print(f"Python {platform.python_version()}, PennyLane {qml.__version__}")
    if run(1):
        print("Exp 1: algorithms"); write_csv(RESULTS / "exp1_algorithms.csv", exp1(args.quick))
    if run(2):
        print("Exp 2: Grover vs linear search"); r = exp2(args.quick)
        write_csv(RESULTS / "exp2_grover_vs_linear.csv", r); exp2_plot(r, RESULTS / "exp2_grover_vs_linear.png")
    if run(3):
        print("Exp 3: noise"); r = exp3(args.quick)
        write_csv(RESULTS / "exp3_noise.csv", r); exp3_plot(r, RESULTS / "exp3_noise.png")
    if run(4):
        print("Exp 4: annealing"); r, m = exp4(args.quick); meta.update(m)
        write_csv(RESULTS / "exp4_annealing.csv", r); exp4_plot(r, m, RESULTS / "exp4_annealing.png")
    if run(5):
        print("Exp 5: embedding"); r = exp5(args.quick)
        write_csv(RESULTS / "exp5_embedding.csv", r); exp5_plot(r, RESULTS / "exp5_embedding.png")
    if run(6):
        print("Exp 6: QML vs classical"); raw, summ = exp6(args.quick)
        write_csv(RESULTS / "exp6_qml_raw.csv", raw); write_csv(RESULTS / "exp6_qml_summary.csv", summ)
    meta_path.write_text(json.dumps(meta, indent=2))
    build_summary()
    with open(RESULTS / "environment.txt", "a") as f:
        f.write(f"python {platform.python_version()} | pennylane {qml.__version__} | {platform.platform()} | "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | mode {'quick' if args.quick else 'full'} | "
                f"experiments {args.only or 'all'} | hardware: none (simulation only)\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
