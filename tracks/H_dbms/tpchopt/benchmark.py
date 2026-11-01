"""Build an index configuration for real and measure it.

For each configuration:
  1. drop every secondary index (primary keys stay), create the chosen ones,
     record build time and on-disk size (pg_relation_size of each index);
  2. one warm-up pass over the 22 queries, then R timed passes;
     latency = median of the timed passes;
  3. result check: an order-preserving SHA-256 of every query's rows, compared
     with the default configuration's hashes (rows are fetched as text, so
     numeric values compare exactly);
  4. EXPLAIN ANALYZE once per query: plan node types, which indexes were used,
     the planner's estimated cost and the actual time.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import time

from . import tpch
from .whatif import Candidate

TIMEOUT_S = 300


def drop_secondary_indexes(conn):
    with conn.cursor() as cur:
        rows = cur.execute("""
            select i.relname from pg_index x
            join pg_class i on i.oid = x.indexrelid
            join pg_class t on t.oid = x.indrelid
            where t.relname = any(%s) and not x.indisprimary""", (tpch.TABLES,)).fetchall()
        for (name,) in rows:
            cur.execute(f"drop index {name}")


def build(conn, config: list[Candidate]):
    drop_secondary_indexes(conn)
    t0 = time.perf_counter()
    names = []
    with conn.cursor() as cur:
        for k, c in enumerate(config):
            nm = f"opt_idx_{k}"
            cur.execute(c.ddl(nm))
            names.append(nm)
        build_s = time.perf_counter() - t0
        size = sum(cur.execute("select pg_relation_size(%s)", (n,)).fetchone()[0] for n in names)
        cur.execute("analyze")
    return dict(build_seconds=build_s, real_size_mb=size / 2 ** 20)


def _plan_info(node, out):
    out["nodes"].add(node["Node Type"])
    if "Index Name" in node:
        out["indexes"].add(node["Index Name"])
    for child in node.get("Plans", []):
        _plan_info(child, out)


def explain_analyze(cur, sql):
    plan = cur.execute("explain (analyze, format json) " + sql).fetchone()[0]
    if isinstance(plan, str):
        plan = json.loads(plan)
    root = plan[0]["Plan"]
    info = {"nodes": set(), "indexes": set()}
    _plan_info(root, info)
    return dict(est_cost=root["Total Cost"], actual_ms=plan[0]["Execution Time"],
                nodes=sorted(info["nodes"]), indexes_used=sorted(info["indexes"]))


def run_workload(conn, workload: dict[int, str], repeats: int = 3):
    """Returns per-query median latency (s), all timings, result hashes, plan info, stream time."""
    with conn.cursor() as cur:
        cur.execute(f"set statement_timeout = '{TIMEOUT_S}s'")
        hashes, plans = {}, {}
        for q, sql in workload.items():                       # warm-up pass + result hash
            cur.execute(sql)
            rows = cur.fetchall()
            h = hashlib.sha256()
            for r in rows:
                h.update(repr(tuple(str(v) for v in r)).encode())
            hashes[q] = (h.hexdigest()[:16], len(rows))
        timings = {q: [] for q in workload}
        streams = []
        for _ in range(repeats):
            t_stream = time.perf_counter()
            for q, sql in workload.items():
                t0 = time.perf_counter()
                cur.execute(sql)
                cur.fetchall()
                timings[q].append(time.perf_counter() - t0)
            streams.append(time.perf_counter() - t_stream)
        for q, sql in workload.items():
            plans[q] = explain_analyze(cur, sql)
    return dict(latency={q: st.median(t) for q, t in timings.items()}, timings=timings,
                hashes=hashes, plans=plans, stream_seconds=st.median(streams),
                stream_all=streams)
