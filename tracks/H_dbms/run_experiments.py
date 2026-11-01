"""Track H experiments: index selection on TPC-H SF 0.1 / PostgreSQL 16.

    python run_experiments.py           # full run (~20 min)
    python run_experiments.py --quick   # 2 budgets, 1 timed pass

Steps
  1. Load TPC-H SF 0.1 (dbgen), primary keys only, VACUUM ANALYZE.
  2. Generate candidate indexes from the workload; estimate sizes and costs with hypopg.
  3. For each storage budget, run the five methods (default, knapsack DP,
     greedy ratio, greedy what-if, ILP) -> chosen index sets + estimated cost.
  4. Build every distinct chosen set FOR REAL (plus "all candidates" as a
     no-budget reference) and measure latency per query, workload time,
     throughput, storage, build time; check results are identical to default.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tpchopt import benchmark as BM, selection as S, tpch, whatif as W

HERE = Path(__file__).parent
RESULTS = HERE / "results"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-load", action="store_true")
    args = ap.parse_args()
    budgets = [16, 25] if args.quick else [4, 16, 25, 40]
    repeats = 1 if args.quick else 3
    RESULTS.mkdir(exist_ok=True)
    conn = tpch.connect()
    with conn.cursor() as cur:
        server = cur.execute("select version()").fetchone()[0]
        settings = {k: cur.execute(f"show {k}").fetchone()[0] for k in
                    ["shared_buffers", "work_mem", "effective_cache_size", "random_page_cost",
                     "max_parallel_workers_per_gather", "jit"]}
    if not args.no_load:
        t0 = time.time()
        tpch.load(conn)
        print(f"loaded in {time.time() - t0:.1f}s")
    BM.drop_secondary_indexes(conn)
    counts = tpch.row_counts(conn)
    sizes0 = tpch.database_size_mb(conn)
    workload = tpch.workload()

    # ---------------- selection
    wi = W.WhatIf(workload)
    cands = W.generate_candidates(workload)
    sel_rows, configs = [], {}
    base_est = wi.total([])
    cand_sizes = {}
    for B in budgets:
        out, cand_sizes, base, b = S.select_all(wi, cands, B)
        for m, r in out.items():
            key = tuple(sorted(c.name for c in r["indexes"]))
            configs.setdefault(key, {"indexes": r["indexes"], "label": [], "est_cost": r["est_cost"]})
            configs[key]["label"].append(f"{m} @ {B} MB")
            sel_rows.append(dict(budget_mb=B, method=m, n_indexes=len(r["indexes"]),
                                 est_size_mb=round(r["size_mb"], 1),
                                 est_cost=round(r["est_cost"]),
                                 est_cost_reduction_pct=round(100 * (1 - r["est_cost"] / base_est), 2),
                                 additive_model_value=round(r["model_value"]),
                                 solve_seconds=round(r["seconds"], 3), note=r["note"],
                                 indexes="; ".join(c.name for c in r["indexes"]),
                                 config_id=None))
        print(f"selection @ {B} MB done", flush=True)
    write_csv(RESULTS / "candidates.csv",
              [dict(index=c.name, est_size_mb=round(cand_sizes[c], 2),
                    additive_benefit=round(sum(S.single_benefits(wi, [c])[1][c].values())))
               for c in cands])
    all_key = tuple(sorted(c.name for c in cands))
    configs.setdefault(all_key, {"indexes": cands, "label": ["all candidates (no budget)"],
                                 "est_cost": wi.total(cands)})
    ids = {}
    for k, key in enumerate(configs):
        ids[key] = f"C{k}"
    for r in sel_rows:
        key = tuple(sorted(n for n in r["indexes"].split("; ") if n))
        r["config_id"] = ids[key]

    # ---------------- real measurement
    bench_rows, query_rows, plan_rows = [], [], []
    default_hashes = None
    for key, cfg in configs.items():
        cid = ids[key]
        info = BM.build(conn, cfg["indexes"])
        res = BM.run_workload(conn, workload, repeats)
        if default_hashes is None:
            default_hashes = res["hashes"]                      # first config is the default (no indexes)
        same = sum(res["hashes"][q] == default_hashes[q] for q in workload)
        mismatched = [q for q in workload if res["hashes"][q] != default_hashes[q]]
        bench_rows.append(dict(config_id=cid, used_by=" | ".join(cfg["label"]), n_indexes=len(cfg["indexes"]),
                               real_size_mb=round(info["real_size_mb"], 1),
                               build_seconds=round(info["build_seconds"], 2),
                               est_cost=round(cfg["est_cost"]),
                               workload_seconds=round(res["stream_seconds"], 3),
                               throughput_qpm=round(22 / res["stream_seconds"] * 60, 1),
                               q17_s=round(res["latency"][17], 3), q20_s=round(res["latency"][20], 3),
                               others_s=round(sum(v for q, v in res["latency"].items() if q not in (17, 20)), 3),
                               results_identical=f"{same}/22", mismatched=str(mismatched or "")))
        for q in workload:
            query_rows.append(dict(config_id=cid, query=q, median_s=round(res["latency"][q], 4),
                                   runs=";".join(f"{t:.4f}" for t in res["timings"][q]),
                                   rows=res["hashes"][q][1], hash=res["hashes"][q][0]))
            p = res["plans"][q]
            plan_rows.append(dict(config_id=cid, query=q, est_cost=round(p["est_cost"]),
                                  explain_analyze_ms=round(p["actual_ms"], 1),
                                  nodes=" ".join(p["nodes"]), indexes_used=" ".join(p["indexes_used"])))
        print(f"{cid}: {len(cfg['indexes'])} idx, {info['real_size_mb']:.1f} MB, workload "
              f"{res['stream_seconds']:.2f}s, identical {same}/22  [{cfg['label'][0]}]", flush=True)
    BM.drop_secondary_indexes(conn)

    write_csv(RESULTS / "selection.csv", sel_rows)
    write_csv(RESULTS / "benchmark.csv", bench_rows)
    write_csv(RESULTS / "query_latency.csv", query_rows)
    write_csv(RESULTS / "plans.csv", plan_rows)

    # ---------------- plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    methods = list(dict.fromkeys(r["method"] for r in sel_rows))
    bmap = {r["config_id"]: r for r in bench_rows}
    for m in methods:
        rs = [r for r in sel_rows if r["method"] == m]
        axes[0].plot([r["budget_mb"] for r in rs], [bmap[r["config_id"]]["workload_seconds"] for r in rs],
                     "o-", label=m)
    axes[0].set_xlabel("storage budget (MB)"); axes[0].set_ylabel("measured workload time, 22 queries (s)")
    axes[0].set_yscale("log"); axes[0].set_title("Measured: whole TPC-H workload"); axes[0].legend(fontsize=7)
    for m in methods:
        rs = [r for r in sel_rows if r["method"] == m]
        axes[1].plot([r["budget_mb"] for r in rs], [bmap[r["config_id"]]["others_s"] for r in rs], "o-", label=m)
    axes[1].set_xlabel("storage budget (MB)"); axes[1].set_ylabel("seconds")
    axes[1].set_title("Measured: the 20 queries other than Q17 and Q20")
    fig.tight_layout(); fig.savefig(RESULTS / "workload_vs_budget.png", dpi=150)

    (RESULTS / "SUMMARY.md").write_text("\n".join([
        "# Track H results (generated by run_experiments.py)\n",
        f"{server}. Settings: {json.dumps(settings)}. Host: {platform.platform()}, "
        f"{__import__('os').cpu_count()} CPUs. Data: TPC-H SF 0.1 {json.dumps(counts)}; "
        f"tables {sizes0['tables_mb']:.1f} MB, primary keys {sizes0['indexes_mb']:.1f} MB.\n",
        f"{len(cands)} candidate indexes; timed passes per configuration: {repeats} (after 1 warm-up), "
        "latency = median.\n",
        "## Selection (planner-estimated cost via hypopg)\n",
        md_table(sel_rows, ["budget_mb", "method", "n_indexes", "est_size_mb", "est_cost_reduction_pct",
                            "additive_model_value", "solve_seconds", "config_id", "indexes"]),
        "\n## Measured (indexes built for real)\n",
        md_table(bench_rows, ["config_id", "n_indexes", "real_size_mb", "build_seconds", "workload_seconds",
                              "throughput_qpm", "q17_s", "q20_s", "others_s", "results_identical", "used_by"]),
        "\n![workload](workload_vs_budget.png)\n",
    ]) + "\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
