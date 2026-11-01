"""Correctness checks for Track H. Need a running PostgreSQL with the `tpch`
database loaded (run_experiments.py does that). Run from tracks/H_dbms: `pytest -q`."""
import itertools

import pytest

from tpchopt import benchmark as BM, selection as S, tpch, whatif as W

RAW_Q1 = """-- using default substitutions
select l_returnflag from lineitem
where
	l_shipdate <= date '1998-12-01' - interval '90' day (3)
group by l_returnflag;
where rownum <= -1;"""

RAW_Q3_TAIL = "select 1 from orders order by 1;\nwhere rownum <= 10;"


@pytest.fixture(scope="module")
def conn():
    try:
        c = tpch.connect()
        c.execute("select 1 from lineitem limit 1")
    except Exception as e:
        pytest.skip(f"PostgreSQL tpch database not available: {e}")
    return c


# ------------------------------------------------------------------ query conversion
def test_convert_interval_and_rownum():
    s = tpch.convert(RAW_Q1, 1)
    assert "day (3)" not in s and "interval '90' day" in s
    assert "rownum" not in s and "limit" not in s and not s.endswith(";")
    assert tpch.convert(RAW_Q3_TAIL, 3).endswith("limit 10")


def test_all_queries_converted():
    w = tpch.workload()
    assert len(w) == 22
    for q, s in w.items():
        assert "rownum" not in s and "(3)" not in s and ";" not in s, q
    assert "create view" not in w[15] and w[15].startswith("with revenue0")


def test_row_counts_match_tpch_spec(conn):
    # TPC-H specification, SF 0.1
    assert tpch.row_counts(conn) == {"region": 5, "nation": 25, "part": 20000, "supplier": 1000,
                                     "partsupp": 80000, "customer": 15000, "orders": 150000,
                                     "lineitem": 600572}


def test_q15_cte_equals_original_view_form(conn):
    raw = (tpch.QUERIES / "q15.raw.sql").read_text()
    stmts = [s.strip() for s in raw.split(";") if s.strip() and "rownum" not in s]
    stmts = [s for s in stmts if not s.startswith("--") or "\n" in s]
    with conn.cursor() as cur:
        cur.execute("drop view if exists revenue0")
        cur.execute(stmts[0].split("\n", 2)[-1] if stmts[0].startswith("--") else stmts[0])
        original = cur.execute(stmts[1]).fetchall()
        cur.execute("drop view revenue0")
        mine = cur.execute(tpch.workload()[15]).fetchall()
    assert original == mine and len(mine) >= 1


# ------------------------------------------------------------------ candidates and what-if
def test_candidates_exclude_primary_key_leading_columns():
    cands = W.generate_candidates(tpch.workload())
    assert all(c.columns[0] not in W.PK_LEADING for c in cands)
    names = {c.name for c in cands}
    assert "lineitem(l_partkey)" in names and "lineitem(l_partkey, l_suppkey)" in names


def test_whatif_cache_and_reset(conn):
    wi = W.WhatIf(tpch.workload())
    base = wi.costs(frozenset())
    c = W.Candidate("lineitem", ("l_partkey",))
    with_c = wi.costs(frozenset([c]))
    assert with_c[17] < base[17] and with_c[20] < base[20]      # the correlated subqueries benefit
    # hypothetical indexes must not leak into later costings
    assert wi.costs(frozenset()) == base
    with wi.conn.cursor() as cur:
        assert cur.execute("select count(*) from hypopg_list_indexes").fetchone()[0] == 0


# ------------------------------------------------------------------ selection algorithms
def test_knapsack_dp_matches_brute_force():
    cands = [W.Candidate("t", (f"c{i}",)) for i in range(8)]
    sizes = dict(zip(cands, [3.0, 4.2, 2.5, 6.0, 1.1, 5.5, 2.0, 3.3]))
    vals = dict(zip(cands, [10, 14, 7, 20, 2, 16, 6, 11]))
    for B in [5, 9.5, 14, 20]:
        dp = S.knapsack_dp(cands, sizes, vals, B)
        assert sum(sizes[c] for c in dp) <= B
        best = max(sum(vals[c] for c in s) for r in range(9) for s in itertools.combinations(cands, r)
                   if sum(sizes[c] for c in s) <= B - 0.1 * len(s))   # DP rounds sizes up by < 0.1
        assert sum(vals[c] for c in dp) >= best


def test_ilp_respects_budget_and_one_index_per_query_table():
    a, b2, c = (W.Candidate("lineitem", ("x",)), W.Candidate("lineitem", ("y",)),
                W.Candidate("orders", ("z",)))
    sizes = {a: 5, b2: 5, c: 1}
    ben = {a: {1: 100, 2: 0}, b2: {1: 90, 2: 0}, c: {1: 0, 2: 10}}
    chosen, status = S.ilp([a, b2, c], sizes, ben, budget_mb=11)
    # additive knapsack would take a + b2 (190); the ILP knows query 1 uses one lineitem index
    assert status == "Optimal" and set(chosen) >= {a, c} and b2 not in chosen
    chosen, _ = S.ilp([a, b2, c], sizes, ben, budget_mb=4)
    assert chosen == [c]


# ------------------------------------------------------------------ benchmark harness
def test_build_and_drop_leave_primary_keys(conn):
    info = BM.build(conn, [W.Candidate("orders", ("o_custkey",))])
    assert info["real_size_mb"] > 0
    BM.drop_secondary_indexes(conn)
    with conn.cursor() as cur:
        n = cur.execute("""select count(*) from pg_index x join pg_class t on t.oid = x.indrelid
                           where t.relname = any(%s)""", (tpch.TABLES,)).fetchone()[0]
    assert n == len(tpch.PRIMARY_KEYS)
