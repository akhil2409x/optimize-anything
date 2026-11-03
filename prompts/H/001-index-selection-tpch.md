# H · 001 · Index selection on TPC-H (PostgreSQL 16)

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-29
- **Kind of work:** planning, code generation, benchmarking, debugging, documentation

## Original prompt

After Track G, Claude said H (PostgreSQL) was next if the student wanted to
continue. Reply:

```
yes continue
```

## AI response (summary)

Claude installed `postgresql-16-hypopg`, built TPC-H `dbgen`/`qgen`
(electrum/tpch-dbgen @ 32f1c1b), generated SF 0.1 data and the 22 queries with
default parameters, and wrote:

- `tpchopt/tpch.py`: schema, COPY loader, primary keys, and conversion from
  qgen's Oracle dialect to PostgreSQL (`interval … day (3)`, `rownum`, Q15 view → CTE).
- `tpchopt/whatif.py`: candidate generation from the workload's predicates,
  plus hypopg what-if costing with a cache.
- `tpchopt/selection.py`: knapsack DP (additive), greedy ratio, what-if greedy
  (AutoAdmin-style) and an ILP with "at most one index per query per table".
- `tpchopt/benchmark.py`: builds the indexes for real, runs a warm-up plus
  3 timed passes, hashes results, and runs EXPLAIN ANALYZE.
- `run_experiments.py`, `setup.sh` and `tests/test_tpchopt.py`.

## Findings that changed the write-up

- **Q17 and Q20 dominate the baseline:** 15 s and 23 s out of about 40 s,
  because of their correlated subqueries.
- **The additive knapsack double counts:** at 40 MB it builds two overlapping
  lineitem indexes.
- **Greedy is short-sighted:** at 25 MB it takes `l_partkey` first and then
  can't afford the better composite index.
- **The planner's estimate is not monotone:** some index sets *raise* the
  estimated workload cost.
- **hypopg overestimates B-tree sizes about 3×,** because it doesn't model
  deduplication.

## Resulting code change

Commit `[AI:claude] Track H: TPC-H index selection (knapsack DP, greedy, ILP)`
= local commit `8617dbc` (on GitHub: `https://github.com/akhil2409x/optimize-anything/commit/8617dbc`)

## Verification note

Done in the session: every configuration returned results identical to the
default (SHA-256 of all rows per query).

To do by the student:
- Run `setup.sh` and `run_experiments.py`.
- Open `results/plans.csv` and explain why Q17 changes from a sequential scan
  inside a SubPlan to an index scan.
- Explain why a knapsack over-counts benefits when indexes overlap, and what
  the ILP's extra constraint fixes.
