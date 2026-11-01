# Track H — DBMS optimization: index selection on TPC-H (PostgreSQL 16)

## Task and formulation

**Task:** choose which secondary indexes to build for the 22-query TPC-H
workload under a storage budget.

**Formulation:**
- **Candidates:** 44 indexes, generated automatically from the columns in the
  queries' WHERE and join predicates (`tpchopt/whatif.py`).
- **Variables:** xᵢ ∈ {0,1}, build candidate index i, whose size sᵢ MB is
  estimated by hypopg.
- **Constraint:** Σ sᵢ xᵢ ≤ B, with storage budgets B ∈ {4, 16, 25, 40} MB.
- **Objective:** minimise the workload's estimated cost C(S) = Σ_q cost_q(S),
  where S is the set of chosen indexes. Costs come from the PostgreSQL planner
  using **hypothetical** indexes (hypopg), so nothing is built while searching.
- **Why it's hard:** C(S) is **not additive**. Two indexes that both help Q20
  don't help it twice, and in PostgreSQL's planner adding an index can even
  raise the estimate.

**Methods** (`tpchopt/selection.py`):

| Method | Chapter 6 technique | Model of C(S) |
|---|---|---|
| default | none (as shipped) | primary keys only |
| knapsack DP | dynamic programming (exact 0/1 knapsack) | additive: vᵢ = Σ_q single-index benefit |
| greedy ratio | greedy | additive, highest vᵢ / sᵢ first |
| greedy what-if | greedy (AutoAdmin-style) | re-costs the *whole* workload with hypopg at each step |
| ILP | integer programming, branch-and-bound (HiGHS) | y_{q,i} = "query q uses index i"; at most one index per query per table; y ≤ x; budget on x |

**Data:** TPC-H at scale factor 0.1 from the official dbgen
(region 5, nation 25, part 20,000, supplier 1,000, partsupp 80,000,
customer 15,000, orders 150,000, lineitem 600,572 rows). Tables take 129 MB
and primary keys 18.7 MB. Queries are `qgen -d`, the specification's default
parameters, converted to PostgreSQL syntax (`tpchopt/tpch.py`).

**Environment:** PostgreSQL 16.15 with default settings (shared_buffers 128 MB,
work_mem 4 MB, up to 2 parallel workers) on a 2-CPU cloud VM. Every
configuration gets 1 warm-up pass then 3 timed passes; latency is the median.

## Results

Full tables are in `results/SUMMARY.md`; per-query timings and plans are in
`results/query_latency.csv` and `results/plans.csv`.

**1. The problem: two queries take 97% of the time.**

| | Default (PK only) | Best budgeted configuration (16 MB) |
|---|---|---|
| Whole workload (22 queries) | 51.6 s (passes: 41.9–52.2 s) | 1.48–1.71 s (every method at ≥ 16 MB) |
| Throughput | 26 queries/min | 770–890 queries/min |
| Q17 | 21.7 s | 0.06 s |
| Q20 | 28.1 s | 0.012–0.035 s |
| Other 20 queries | 1.58 s | 1.39–1.62 s (no consistent change) |
| Extra storage | 0 | 4.7–6.5 MB on disk |

- **Why Q17 and Q20 are slow:** both run a correlated subquery once per outer
  row, and each run scans all of `lineitem` (EXPLAIN shows Seq Scan inside a
  SubPlan).
- **The fix:** an index starting with `l_partkey` turns each inner scan into a
  bitmap index scan, so the whole workload runs **about 33× faster**.
- **Results are unchanged:** every configuration returned **identical results
  for all 22 queries**, checked by an order-preserving SHA-256 of every row.

**2. Comparing methods by the planner's estimate (what they optimize)**

| Budget | Knapsack DP | Greedy ratio | Greedy what-if | ILP |
|---|---|---|---|---|
| 4 MB | −0.58% (worse) | +0.05% | +0.05% | −0.58% (worse) |
| 16 MB | 97.51% | 97.51% | 97.51% | 97.51% |
| 25 MB | **97.98%** | 97.54% | 97.61% | **97.98%** |
| 40 MB | 97.98% (redundant index) | 98.01% | 98.08% | **98.09%** |

(Reduction in estimated workload cost.)

- **The additive knapsack double counts.** At 40 MB it builds both
  `lineitem(l_partkey, l_suppkey)` *and* `lineitem(l_partkey)`. It values the
  pair at 38.0M cost units, while together they are worth about 19.0M, so it
  wastes 15 MB of budget on the second.
- **The greedy is short-sighted.** At 25 MB both greedies take the cheaper
  `l_partkey` first (better benefit per MB) and can then no longer afford the
  composite `(l_partkey, l_suppkey)`, which serves Q17 *and* Q20. The ILP and
  the DP find the composite.
- **The planner's estimate is not monotone.** At 4 MB the DP and ILP pick
  indexes that *raise* the estimated cost by 0.6%, because PostgreSQL's
  planner heuristics don't guarantee more indexes never hurt. Only the what-if
  greedy, which checks every step against the real planner, refuses them.
- **The ILP was best or tied at every budget,** and it solved in under
  0.05 s. The what-if greedy needed up to 3,410 EXPLAIN calls (about 3 s).

**3. Comparing methods by measured time: the estimate's advantages mostly vanish.**
- **Q20 is the one per-query difference above noise:** 12–14 ms with the
  composite index (DP/ILP at 25 and 40 MB) vs 28–42 ms with `l_partkey` alone.
  It shows consistently in every pass and in EXPLAIN ANALYZE.
- **The whole-workload time doesn't separate the methods.** At ≥ 16 MB it is
  1.48–1.71 s for every method, a spread no bigger than the run-to-run spread
  (up to 0.26 s between passes of the same configuration). The other 20
  queries take about 1.5 s in every configuration, the default included: they
  are scans over data already in memory, and these indexes don't change their
  plans in a way that shows up.
- **At 4 MB nothing fits that fixes Q17 or Q20.** The apparent 44.8 s vs
  51.6 s is within noise: the default's own passes ranged 41.9–52.2 s.
- **Indexing everything isn't worth it.** All 44 candidates take 201 MB
  (31× the 6.5 MB ILP set) and a 7.4 s build, for a workload time of 1.35 s
  vs about 1.6 s.

**4. Estimates vs reality**
- **Index sizes:** hypopg overestimated B-tree sizes about 3× (15.6 MB
  estimated vs 4.7 MB on disk for the 16 MB DP set), because it doesn't model
  PostgreSQL 13+ B-tree deduplication. In practice the budgets bound about 3×
  more loosely than their nominal value.
- **Query costs:** the planner estimates Q17 90× cheaper than Q20, yet it took
  about 75% as long. An optimizer that minimizes estimated cost is only as
  good as the cost model.

## How to run

```bash
# Ubuntu/WSL: see setup.sh header for apt/pip packages
./setup.sh                          # dbgen/qgen, SF 0.1 data, queries, hypopg extension
python run_experiments.py           # load, select, build + benchmark (~20 min)
python run_experiments.py --quick   # 2 budgets, 1 timed pass
pytest -q                           # 9 tests (need the loaded tpch database)
```

## Correctness checks (`tests/test_tpchopt.py`)

- **Query conversion:** interval syntax, rownum → LIMIT, and Q15. The CTE
  version gives exactly the rows of the original create view / select / drop
  view version.
- **Data:** row counts equal the TPC-H specification for SF 0.1.
- **Candidates:** never duplicate a primary key's leading column.
- **What-if costing:** hypothetical indexes are always reset, and the
  `l_partkey` index lowers the estimated cost of Q17 and Q20.
- **Knapsack DP:** matches brute force on 4 budgets.
- **ILP:** respects the budget, uses at most one index per query and table,
  and never builds useless indexes (regression test for failure log F9).
- **Index build/drop:** leaves exactly the primary keys.
- **Results identical:** all 22 queries hashed and compared for every
  configuration in every run.

## Limitations

- **Small, in-memory data:** at SF 0.1 everything fits in RAM, so indexes only
  matter where they change the algorithm (correlated subqueries), not I/O. At
  SF 1 or above, scan-heavy queries would gain more.
- **Read-only workload:** index maintenance cost on INSERT/UPDATE is not
  counted. A real advisor would charge for it.
- **Budgets use hypopg's size estimates,** about 3× too high (see 4).
- **One environment:** a shared 2-CPU VM. Run-to-run noise is reported rather
  than hidden.
