"""TPC-H at scale factor 0.1 on PostgreSQL: schema, data load, workload.

Data   : official TPC-H dbgen (github.com/electrum/tpch-dbgen, commit 32f1c1b),
         `dbgen -s 0.1`: 600,572 lineitem rows, 150,000 orders, ... (~100 MB of text).
Queries: the 22 TPC-H queries from `qgen -d` (the specification's default
         substitution parameters), converted from qgen's Oracle dialect:
           * `interval 'N' day (3)`  ->  `interval 'N' day`
           * trailing `where rownum <= N;` -> `limit N` (dropped when N = -1)
           * Q15's create view / select / drop view -> one statement with a CTE
             (same result, one round trip, no DDL inside the workload)
Baseline physical design: primary keys only (dss.ri), no secondary indexes.
"""
from __future__ import annotations

import re
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parents[1]
DBGEN = HERE / "tpch-dbgen"
QUERIES = HERE / "queries"
DSN = "dbname=tpch"

TABLES = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]
PRIMARY_KEYS = {
    "region": "r_regionkey", "nation": "n_nationkey", "part": "p_partkey",
    "supplier": "s_suppkey", "partsupp": "ps_partkey, ps_suppkey", "customer": "c_custkey",
    "orders": "o_orderkey", "lineitem": "l_orderkey, l_linenumber",
}
# column prefix -> table (TPC-H naming convention)
PREFIX = {"r_": "region", "n_": "nation", "p_": "part", "s_": "supplier", "ps_": "partsupp",
          "c_": "customer", "o_": "orders", "l_": "lineitem"}

Q15_CTE = """with revenue0 (supplier_no, total_revenue) as (
	select
		l_suppkey,
		sum(l_extendedprice * (1 - l_discount))
	from
		lineitem
	where
		l_shipdate >= date '{d}'
		and l_shipdate < date '{d}' + interval '3' month
	group by
		l_suppkey
)
select
	s_suppkey,
	s_name,
	s_address,
	s_phone,
	total_revenue
from
	supplier,
	revenue0
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue0
	)
order by
	s_suppkey"""


def connect(**kw):
    return psycopg.connect(DSN, autocommit=True, **kw)


def convert(raw: str, qnum: int) -> str:
    """qgen Oracle dialect -> PostgreSQL, as a single statement without ';'."""
    lines = [l for l in raw.splitlines() if not l.startswith("--")]
    sql = "\n".join(lines).strip()
    m = re.search(r"\n?where rownum <= (-?\d+);\s*$", sql)
    limit = int(m.group(1)) if m else -1
    if m:
        sql = sql[:m.start()].rstrip()
    if qnum == 15:
        d = re.search(r"l_shipdate >= date '([\d-]+)'", sql).group(1)
        sql = Q15_CTE.format(d=d)
    sql = re.sub(r"interval '(\d+)' day \(\d+\)", r"interval '\1' day", sql)
    sql = sql.rstrip().rstrip(";").rstrip()
    if limit > 0:
        sql += f"\nlimit {limit}"
    return sql


def workload() -> dict[int, str]:
    out = {}
    for q in range(1, 23):
        p = QUERIES / f"q{q}.sql"
        if not p.exists():
            p.write_text(convert((QUERIES / f"q{q}.raw.sql").read_text(), q) + ";\n")
        out[q] = p.read_text().strip().rstrip(";")
    return out


def load(conn, scale: float = 0.1):
    ddl = (DBGEN / "dss.ddl").read_text()
    with conn.cursor() as cur:
        for t in TABLES:
            cur.execute(f"drop table if exists {t} cascade")
        for stmt in ddl.split(";"):
            if "CREATE TABLE" in stmt.upper():
                cur.execute(stmt)
        for t in TABLES:
            path = DBGEN / f"{t}.tbl"
            with cur.copy(f"copy {t} from stdin with (format csv, delimiter '|')") as cp:
                with open(path) as f:
                    for line in f:
                        cp.write(line.rstrip("\n").rstrip("|") + "\n")   # dbgen ends lines with '|'
        for t, cols in PRIMARY_KEYS.items():
            cur.execute(f"alter table {t} add primary key ({cols})")
        cur.execute("vacuum analyze")


def table_of(column: str) -> str:
    for pre in sorted(PREFIX, key=len, reverse=True):
        if column.startswith(pre):
            return PREFIX[pre]
    raise ValueError(column)


def row_counts(conn):
    with conn.cursor() as cur:
        return {t: cur.execute(f"select count(*) from {t}").fetchone()[0] for t in TABLES}


def database_size_mb(conn) -> dict:
    with conn.cursor() as cur:
        heap = sum(cur.execute("select pg_table_size(%s)", (t,)).fetchone()[0] for t in TABLES)
        idx = sum(cur.execute("select pg_indexes_size(%s)", (t,)).fetchone()[0] for t in TABLES)
    return dict(tables_mb=heap / 2 ** 20, indexes_mb=idx / 2 ** 20)
