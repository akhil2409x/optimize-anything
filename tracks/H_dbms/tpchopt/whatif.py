"""Candidate indexes and what-if costing with hypopg.

Candidates are generated from the workload itself (no hand-picking):
  * single-column B-tree indexes on every column that appears in a WHERE /
    join predicate of some query, for the six large tables;
  * two-column indexes for pairs of columns of the same table that appear
    together in one query's predicates, with a join key (…key) first;
  * minus anything already covered by the primary key's leading column.

What-if costing: hypopg creates *hypothetical* indexes that exist only in the
planner's catalog for this session. `EXPLAIN` (without ANALYZE) then plans as
if they existed, so the optimizer's estimated cost of every query under any
index configuration can be read without building anything. hypopg also
estimates each index's size.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field

from . import tpch

BIG_TABLES = {"part", "supplier", "partsupp", "customer", "orders", "lineitem"}
PK_LEADING = {"p_partkey", "s_suppkey", "ps_partkey", "c_custkey", "o_orderkey", "l_orderkey"}
COLUMN_RE = re.compile(r"\b(ps_\w+|[pslocnr]_\w+)\b")


@dataclass(frozen=True)
class Candidate:
    table: str
    columns: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.table}({', '.join(self.columns)})"

    def ddl(self, real_name: str | None = None) -> str:
        nm = f" {real_name}" if real_name else ""
        return f"create index{nm} on {self.table} ({', '.join(self.columns)})"


def predicate_columns(sql: str) -> list[set[str]]:
    """Column sets per WHERE clause (outer query and each subquery)."""
    groups = []
    for part in re.split(r"\bwhere\b", sql, flags=re.I)[1:]:
        part = re.split(r"\b(group by|order by|limit)\b", part, flags=re.I)[0]
        cols = {c.lower() for c in COLUMN_RE.findall(part)}
        cols = {c for c in cols if not c.startswith(("supplier_no", "total_revenue"))}
        groups.append(cols)
    return groups


def generate_candidates(workload: dict[int, str]) -> list[Candidate]:
    cands = set()
    for sql in workload.values():
        for cols in predicate_columns(sql):
            by_table = {}
            for c in cols:
                try:
                    t = tpch.table_of(c)
                except ValueError:
                    continue
                if t in BIG_TABLES:
                    by_table.setdefault(t, set()).add(c)
            for t, cs in by_table.items():
                for c in cs:
                    if c not in PK_LEADING:
                        cands.add(Candidate(t, (c,)))
                for a, b in itertools.permutations(sorted(cs), 2):
                    if a.endswith("key") and a not in PK_LEADING:
                        cands.add(Candidate(t, (a, b)))
    return sorted(cands, key=lambda c: (c.table, c.columns))


@dataclass
class WhatIf:
    """Holds a connection with hypopg loaded and caches workload costs per configuration."""
    workload: dict[int, str]
    conn: object = None
    cache: dict = field(default_factory=dict)
    calls: int = 0

    def __post_init__(self):
        self.conn = self.conn or tpch.connect()
        with self.conn.cursor() as cur:
            cur.execute("create extension if not exists hypopg")
            cur.execute("select hypopg_reset()")

    def _cost(self, sql: str) -> float:
        with self.conn.cursor() as cur:
            plan = cur.execute("explain (format json) " + sql).fetchone()[0]
        self.calls += 1
        if isinstance(plan, str):
            plan = json.loads(plan)
        return float(plan[0]["Plan"]["Total Cost"])

    def costs(self, config: frozenset[Candidate]) -> dict[int, float]:
        """Estimated cost of every query with exactly `config` as secondary indexes."""
        key = frozenset(config)
        if key not in self.cache:
            with self.conn.cursor() as cur:
                cur.execute("select hypopg_reset()")
                for c in key:
                    cur.execute("select * from hypopg_create_index(%s)", (c.ddl(),))
            self.cache[key] = {q: self._cost(s) for q, s in self.workload.items()}
            with self.conn.cursor() as cur:
                cur.execute("select hypopg_reset()")
        return self.cache[key]

    def total(self, config) -> float:
        return sum(self.costs(frozenset(config)).values())

    def size_mb(self, c: Candidate) -> float:
        with self.conn.cursor() as cur:
            cur.execute("select hypopg_reset()")
            oid = cur.execute("select indexrelid from hypopg_create_index(%s)", (c.ddl(),)).fetchone()[0]
            size = cur.execute("select hypopg_relation_size(%s)", (oid,)).fetchone()[0]
            cur.execute("select hypopg_reset()")
        return size / 2 ** 20
