"""Track D experiments: submodular, parallel and distributed optimization.

    python run_experiments.py           # full run (~6 min on 2 cores)
    python run_experiments.py --quick   # ~1.5 min
    python run_experiments.py --only 3 4

Exp 1  Test the assumptions: monotonicity and diminishing returns on 4 objectives.
Exp 2  Greedy vs exact (ILP / brute force) vs random sampling; ratio vs 1 - 1/e;
       lazy greedy evaluation counts.
Exp 3  Strong scaling (fixed problem, Amdahl fit) and weak scaling (problem grows
       with p, Gustafson), including oversubscription p > cores.
Exp 4  Load balancing: real runs (block / cyclic / dynamic) and a scheduler
       simulator on the measured task times (block, cyclic, dynamic, LPT,
       work stealing) vs the optimal makespan (ILP), for 2-16 workers.
Exp 5  Overheads: task granularity, data shipping, cache locality, synchronization.
Exp 6  Data, model and pipeline parallelism (results checked against serial).
Exp 7  FedAvg vs centralized training: IID vs non-IID clients, local epochs,
       communication rounds and bytes.
Exp 8  Multi-task learning: equal weights vs MGDA vs GradNorm, step sizes tuned
       on a validation split, 3 seeds.
"""
from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics as st
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from distopt import learning as L, parallel as P, submodular as S

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SUMMARY: list[str] = []


def out(line=""):
    print(line)
    SUMMARY.append(line)


def write_csv(name, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    out("| " + " | ".join(cols) + " |")
    out("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c].format(v) if c in fmt else (f"{v:.4g}" if isinstance(v, float) else str(v)))
        out("| " + " | ".join(cells) + " |")
    out()


# ------------------------------------------------------------------ Exp 1
def exp1(quick):
    out("## Exp 1 · Testing the assumptions (random A ⊆ B, e ∉ B)")
    out()
    trials = 500 if quick else 3000
    rows = []
    for obj in (S.Coverage(), S.FacilityLocation(), S.Synergy(), S.R2Selection()):
        r = S.test_properties(obj, trials)
        ex = r["example"]
        rows.append(dict(objective=obj.name, n=obj.n, trials=trials,
                         monotone_violations=r["monotone_violations"],
                         submodular_violations=r["submodular_violations"],
                         worst_violation=r["worst_violation"],
                         example=(f"A={ex[0]}, B={ex[1]}, e={ex[2]}: gain(A)={ex[3]:.3g} < gain(B)={ex[4]:.3g}"
                                  if ex else "")))
    write_csv("exp1_properties.csv", rows)
    table(rows, ["objective", "trials", "monotone_violations", "submodular_violations", "worst_violation"])
    for r in rows:
        if r["example"]:
            out(f"- Counter-example for {r['objective']}: {r['example']}")
    out()


# ------------------------------------------------------------------ Exp 2
def exp2(quick):
    out("## Exp 2 · Greedy vs exact vs random sampling")
    out()
    bound = 1 - 1 / math.e
    rows = []
    cases = [(S.Coverage(), "ILP", range(2, 9 if not quick else 6)),
             (S.FacilityLocation(), "brute force", range(2, 5 if not quick else 4)),
             (S.Synergy(), "brute force", range(2, 7)),
             (S.R2Selection(), "brute force", range(2, 6))]
    for obj, how, ks in cases:
        for k in ks:
            Sg, vg, eg = S.greedy(obj, k)
            Sl, vl, el = S.lazy_greedy(obj, k)
            t0 = time.perf_counter()
            if how == "ILP":
                _, vx, status = obj.exact(k)
            else:
                _, vx = S.brute_force(obj, k)
                status = f"{math.comb(obj.n, k)} subsets"
            tx = time.perf_counter() - t0
            vr = S.random_best(obj, k, tries=eg)              # same number of evaluations as greedy
            rows.append(dict(objective=obj.name, k=k, exact_method=how, exact=vx, exact_seconds=tx,
                             greedy=vg, greedy_ratio=vg / vx, random_same_budget_ratio=vr / vx,
                             greedy_evals=eg, lazy_evals=el, lazy_same_answer=(Sg == Sl),
                             below_guarantee=vg / vx < bound, status=status))
    write_csv("exp2_greedy.csv", rows)
    table(rows, ["objective", "k", "exact_method", "greedy_ratio", "random_same_budget_ratio",
                 "greedy_evals", "lazy_evals", "lazy_same_answer", "exact_seconds"],
          {"greedy_ratio": "{:.3f}", "random_same_budget_ratio": "{:.3f}", "exact_seconds": "{:.2f}"})
    out(f"Guarantee for monotone submodular f: greedy_ratio ≥ 1 − 1/e = {bound:.3f}.")
    out()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in dict.fromkeys(r["objective"] for r in rows):
        rr = [r for r in rows if r["objective"] == name]
        ax.plot([r["k"] for r in rr], [r["greedy_ratio"] for r in rr], "o-", label=name)
    ax.axhline(bound, ls="--", c="k", lw=1, label="1 − 1/e")
    ax.set_xlabel("k (items chosen)")
    ax.set_ylabel("greedy / optimum")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "greedy_ratio.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ Exp 3
def _median_time(n, p, serial_s, reps):
    return st.median(P.run_parallel(n, p, serial_s)[1] for _ in range(reps))


def exp3(quick):
    out(f"## Exp 3 · Speed-up: strong and weak scaling ({P.CORES} cores available)")
    out()
    reps = 1 if quick else 3
    n = 60_000 if quick else 200_000
    ps = [1, 2, 3, 4] if P.CORES <= 2 else sorted({1, 2, P.CORES // 2, P.CORES, 2 * P.CORES})
    rows = []
    for serial_s in (0.0, 0.25 if quick else 0.6):
        T1 = _median_time(n, 1, serial_s, reps)
        for p in ps:
            Tp = T1 if p == 1 else _median_time(n, p, serial_s, reps)
            rows.append(dict(scaling="strong", serial_part_s=serial_s, n=n, p=p, seconds=Tp,
                             speedup=T1 / Tp, efficiency=T1 / Tp / p))
    for serial_s in sorted({r["serial_part_s"] for r in rows}):
        rr = [r for r in rows if r["serial_part_s"] == serial_s]
        within = [r for r in rr if r["p"] <= P.CORES]
        f = P.fit_parallel_fraction([r["p"] for r in within], [r["speedup"] for r in within])
        T1 = rr[0]["seconds"]
        f_true = (T1 - serial_s) / T1
        for r in rr:
            r["amdahl_fitted_f"] = f
            r["f_from_timing"] = f_true
            r["amdahl_prediction"] = P.amdahl(min(r["p"], P.CORES), f_true)
    # weak scaling: work per process fixed
    n0 = n // 2
    serial_s = rows[-1]["serial_part_s"]
    T1 = _median_time(n0, 1, serial_s, reps)
    for p in ps:
        Tp = T1 if p == 1 else _median_time(n0 * p, p, serial_s, reps)
        Ts = _median_time(n0 * p, 1, serial_s, 1) if p > 1 else T1     # same big problem run serially
        f = (Ts - serial_s) / Ts
        rows.append(dict(scaling="weak", serial_part_s=serial_s, n=n0 * p, p=p, seconds=Tp,
                         speedup=Ts / Tp, efficiency=Ts / Tp / p, amdahl_fitted_f="", f_from_timing=f,
                         amdahl_prediction=P.gustafson(min(p, P.CORES), f)))
    write_csv("exp3_scaling.csv", rows)
    table(rows, ["scaling", "serial_part_s", "n", "p", "seconds", "speedup", "efficiency", "amdahl_prediction"],
          {"seconds": "{:.3f}", "speedup": "{:.2f}", "efficiency": "{:.2f}", "amdahl_prediction": "{:.2f}"})
    out("`amdahl_prediction` is Amdahl's law for strong rows and Gustafson's law for weak rows, using the "
        "parallel fraction measured from the serial timing and p capped at the core count. "
        "Measured speed-up for weak scaling = T(serial, same big problem) / T(p).")
    out()
    for s in sorted({r["serial_part_s"] for r in rows if r["scaling"] == "strong"}):
        r = next(r for r in rows if r["scaling"] == "strong" and r["serial_part_s"] == s)
        out(f"- serial part {s}s: parallel fraction from timing f = {r['f_from_timing']:.3f}, "
            f"Amdahl fit on p ≤ {P.CORES}: f = {r['amdahl_fitted_f']:.3f}; Amdahl limit 1/(1−f) = "
            + (f"{1 / (1 - r['f_from_timing']):.1f}×" if r['f_from_timing'] < 0.999 else "unbounded"))
    out()
    fig, ax = plt.subplots(figsize=(6, 4))
    pp = np.linspace(1, max(ps), 50)
    for key, lab in [(("strong", 0.0), "strong, no serial part"),
                     (("strong", serial_s), f"strong, {serial_s}s serial"), (("weak", serial_s), "weak scaling")]:
        rr = [r for r in rows if (r["scaling"], r["serial_part_s"]) == key]
        ax.plot([r["p"] for r in rr], [r["speedup"] for r in rr], "o-", label=f"measured: {lab}")
    ax.plot(pp, np.minimum(pp, P.CORES), "k:", lw=1, label=f"ideal on {P.CORES} cores")
    ax.set_xlabel("processes p")
    ax.set_ylabel("speed-up")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "speedup.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ Exp 4
def exp4(quick):
    out("## Exp 4 · Load balancing and makespan")
    out()
    tasks = P.heterogeneous_tasks(48)
    durations = [P.timed(P._work_range, t)[1] for t in tasks]
    out(f"48 tasks, measured serially: total {sum(durations):.2f}s, shortest {min(durations) * 1e3:.1f} ms, "
        f"longest {max(durations) * 1e3:.1f} ms.")
    out()
    real = []
    for pol in ("block", "cyclic", "dynamic", "dynamic-big"):
        runs = [P.load_balance(tasks, P.CORES, pol) for _ in range(1 if quick else 3)]
        r = min(runs, key=lambda x: x["makespan"])
        real.append(dict(policy=pol, workers=P.CORES, makespan_s=r["makespan"], imbalance=r["imbalance"],
                         sim_makespan_s=P.simulate(durations, P.CORES,
                                                   {"dynamic-big": "block"}.get(pol, pol))[0]))
    write_csv("exp4_load_balance_real.csv", real)
    out(f"Real runs on {P.CORES} processes (best of {1 if quick else 3}); imbalance = max busy / mean busy; "
        "`sim_makespan_s` is the simulator's prediction from the serial task times:")
    out()
    table(real, ["policy", "workers", "makespan_s", "imbalance", "sim_makespan_s"],
          {"makespan_s": "{:.3f}", "imbalance": "{:.2f}", "sim_makespan_s": "{:.3f}"})
    sims = []
    for p in (2, 4, 8, 16):
        opt, status = P.optimal_makespan(durations, p, time_limit=20 if quick else 30)
        lb = max(sum(durations) / p, max(durations))
        for pol in ("block", "cyclic", "dynamic", "lpt", "steal"):
            m, info = P.simulate(durations, p, pol)
            sims.append(dict(workers=p, policy=pol, makespan_s=m, vs_optimal=m / opt, steals=info["steals"],
                             optimal_s=opt, ilp_status=status, lower_bound_s=lb))
    write_csv("exp4_load_balance_sim.csv", sims)
    out("Simulator on the same measured task times (zero scheduling cost); `vs_optimal` = makespan / ILP optimum:")
    out()
    wide = []
    for p in (2, 4, 8, 16):
        r = {"workers": p}
        for s in sims:
            if s["workers"] == p:
                r[s["policy"]] = f"{s['vs_optimal']:.3f}"
                r["ILP optimum (s)"] = f"{s['optimal_s']:.3f} ({s['ilp_status']})"
                if s["policy"] == "steal":
                    r["steals"] = s["steals"]
        wide.append(r)
    table(wide, ["workers", "block", "cyclic", "dynamic", "lpt", "steal", "steals", "ILP optimum (s)"])


# ------------------------------------------------------------------ Exp 5
def exp5(quick):
    out("## Exp 5 · Overheads: granularity, communication, locality, synchronization")
    out()
    rows = P.overhead(P.CORES)
    write_csv("exp5_overhead.csv", rows)
    table(rows, ["test", "setting", "seconds"], {"seconds": "{:.3f}"})
    loc = P.locality(n=2000 if quick else 3000)
    write_csv("exp5_locality.csv", loc)
    table(loc, ["access", "seconds", "ns_per_element"], {"seconds": "{:.3f}"})
    syn = P.synchronization(n=10_000 if quick else 40_000, procs=P.CORES)
    write_csv("exp5_synchronization.csv", syn)
    table(syn, ["method", "total", "seconds"], {"seconds": "{:.3f}"})


# ------------------------------------------------------------------ Exp 6
def exp6(quick):
    out("## Exp 6 · Data, model and pipeline parallelism")
    out()
    dp = P.data_parallel(n=100_000 if quick else 400_000, procs=P.CORES)
    mp_ = P.model_parallel(n=3000 if quick else 6000, procs=P.CORES)
    pl = P.pipeline(micro_batches=16, stage_work=1500 if quick else 2500)
    rows = [dict(kind="data parallel (gradient over data shards)", serial_s=dp["serial_s"],
                 shipped_s=dp["shipped_s"], resident_s=dp["resident_s"], max_abs_diff=dp["max_abs_diff"]),
            dict(kind="model parallel (layer rows split)", serial_s=mp_["serial_s"],
                 shipped_s=mp_["shipped_s"], resident_s=mp_["resident_s"], max_abs_diff=mp_["max_abs_diff"])]
    write_csv("exp6_data_model.csv", rows)
    out(f"{dp['steps']} steps each on {P.CORES} processes. 'shipped' sends the data/weights every step; "
        "'resident' keeps them in the workers and sends only the small vector:")
    out()
    table(rows, ["kind", "serial_s", "shipped_s", "resident_s", "max_abs_diff"],
          {"serial_s": "{:.3f}", "shipped_s": "{:.3f}", "resident_s": "{:.3f}", "max_abs_diff": "{:.1e}"})
    out(f"- Data parallel moves {dp['bytes_per_step_shipped'] / 2 ** 20:.0f} MB per step when shipped vs "
        f"{dp['bytes_per_step_resident']} bytes when resident.")
    write_csv("exp6_pipeline.csv", [pl])
    out(f"- Pipeline, 2 stages × {pl['micro_batches']} micro-batches: sequential {pl['sequential_s']:.3f}s, "
        f"pipelined {pl['pipelined_s']:.3f}s (ideal (M+S−1)·t = {pl['ideal_pipelined_s']:.3f}s, "
        f"fill/drain bubble {pl['bubble_fraction']:.0%}); outputs in order: {pl['in_order']}.")
    out()


# ------------------------------------------------------------------ Exp 7
def exp7(quick):
    out("## Exp 7 · FedAvg vs centralized (softmax regression on 8×8 digits, 10 clients)")
    out()
    Xtr, Xte, Ytr, Yte = L.digits_data()
    epochs = 20
    model_bytes = Xtr.shape[1] * 10 * 8
    rows = []
    _, h = L.centralized(Xtr, Ytr, Xte, Yte, epochs)
    rows.append(dict(setting="centralized SGD", data="all", local_epochs="", rounds="", epochs_of_work=epochs,
                     test_acc=h[-1]["test_acc"], train_loss=h[-1]["train_loss"], comm_MB=0.0))
    curves = {"centralized": [x["test_acc"] for x in h]}
    for iid in (True, False):
        clients = L.split_clients(Ytr, 10, iid)
        for E in (1, 5) if quick else (1, 2, 5, 10):
            rounds = epochs // E                      # same total local work as centralized
            for budget in (("same work",) if E == 1 else ("same work", "same rounds")):
                R = rounds if budget == "same work" else epochs
                _, h = L.fedavg(Xtr, Ytr, Xte, Yte, clients, R, E)
                rows.append(dict(setting=f"FedAvg ({budget})", data="IID" if iid else "non-IID",
                                 local_epochs=E, rounds=R, epochs_of_work=R * E, test_acc=h[-1]["test_acc"],
                                 train_loss=h[-1]["train_loss"], comm_MB=R * 10 * 2 * model_bytes / 2 ** 20))
                if R == epochs:
                    curves[f"{'IID' if iid else 'non-IID'} E={E}"] = [x["test_acc"] for x in h]
    write_csv("exp7_fedavg.csv", rows)
    table(rows, ["setting", "data", "local_epochs", "rounds", "epochs_of_work", "test_acc", "train_loss", "comm_MB"],
          {"test_acc": "{:.3f}", "train_loss": "{:.3f}", "comm_MB": "{:.2f}"})
    out("Communication = rounds × 10 clients × 2 transfers × model size "
        f"({model_bytes / 1024:.1f} KB). Centralized training moves the raw data once instead "
        f"({Xtr.nbytes / 2 ** 20:.2f} MB), which FedAvg avoids for privacy.")
    out()
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in curves.items():
        ax.plot(range(1, len(v) + 1), v, label=k, lw=2 if k == "centralized" else 1)
    ax.set_xlabel("epoch (centralized) / communication round (FedAvg)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.6, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "fedavg.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ Exp 8
def exp8(quick):
    out("## Exp 8 · Multi-task learning: equal weights vs MGDA vs GradNorm")
    out()
    tr, va, te = L.two_task_data()
    steps = 1500 if quick else 3000
    seeds = [0] if quick else [0, 1, 2]
    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    rows, curves = [], {}
    for m in ("equal", "mgda", "mgda_raw", "gradnorm"):
        best = None
        for lr in lrs:                                   # tuned on the validation split, seed 0
            _, h, _ = L.train_multitask(m, tr, va, steps, lr=lr)
            sc = h[-1]["test1"] + h[-1]["test2"]
            if np.isfinite(sc) and (best is None or sc < best[1]):
                best = (lr, sc)
        res = []
        for s in seeds:
            _, h, al = L.train_multitask(m, tr, te, steps, lr=best[0], seed=s)
            res.append((h, al))
        h0 = res[0][0]
        curves[m] = h0
        t1 = [r[0][-1]["test1"] for r in res]
        t2 = [r[0][-1]["test2"] for r in res]
        rows.append(dict(method=m, lr=best[0], task1_test_nmse=st.mean(t1), task1_sd=st.pstdev(t1),
                         task2_test_nmse=st.mean(t2), task2_sd=st.pstdev(t2),
                         final_weights=f"({h0[-1]['w1']:.2f}, {h0[-1]['w2']:.2f})" if m == "gradnorm" else "",
                         mean_alpha=f"{np.mean(res[0][1]):.3f}" if res[0][1] else "",
                         cos_g1_g2_final=h0[-1]["cos_g1_g2"]))
    write_csv("exp8_multitask.csv", rows)
    out(f"Two regression tasks sharing a 16-unit tanh layer; task 2's targets are 30× larger. "
        f"nMSE = test MSE / target variance (1.0 = predicting the mean). {len(seeds)} seed(s), {steps} full-batch steps.")
    out()
    table(rows, ["method", "lr", "task1_test_nmse", "task1_sd", "task2_test_nmse", "task2_sd",
                 "mean_alpha", "final_weights", "cos_g1_g2_final"],
          {"task1_test_nmse": "{:.3f}", "task1_sd": "{:.3f}", "task2_test_nmse": "{:.4f}", "task2_sd": "{:.4f}",
           "cos_g1_g2_final": "{:.2f}"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for m, h in curves.items():
        axes[0].plot([x["step"] for x in h], [x["test1"] for x in h], label=m)
        axes[1].plot([x["step"] for x in h], [x["test2"] for x in h], label=m)
    for ax, t in zip(axes, ("task 1 (small targets)", "task 2 (30× targets)")):
        ax.set_yscale("log")
        ax.set_title(t)
        ax.set_xlabel("step")
        ax.set_ylabel("test nMSE")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "multitask.png", dpi=120)
    plt.close(fig)


EXPS = {1: exp1, 2: exp2, 3: exp3, 4: exp4, 5: exp5, 6: exp6, 7: exp7, 8: exp8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out("# Track D results")
    out()
    out(f"Machine: {platform.platform()}, Python {platform.python_version()}, {P.CORES} CPU cores, "
        f"{'quick' if a.quick else 'full'} run, {time.strftime('%Y-%m-%d %H:%M')}.")
    out()
    t0 = time.time()
    for k in a.only or EXPS:
        t = time.time()
        EXPS[k](a.quick)
        print(f"[exp {k}: {time.time() - t:.0f}s]")
    out(f"Total run time {time.time() - t0:.0f}s.")
    if not a.only:
        (RESULTS / "SUMMARY.md").write_text("\n".join(SUMMARY) + "\n")


if __name__ == "__main__":
    main()
