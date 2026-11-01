#!/usr/bin/env bash
# Track H: PostgreSQL 16 + hypopg, TPC-H dbgen/qgen, SF 0.1 data and queries.
#
#   Ubuntu/Debian/WSL:
#     sudo apt-get install -y postgresql-16 postgresql-16-hypopg gcc make git
#     pip install "psycopg[binary]" pulp==2.9.0 highspy matplotlib pytest
#   The database user running the scripts needs a role and a database:
#     sudo -u postgres createuser --superuser "$USER" && createdb tpch
set -euo pipefail
cd "$(dirname "$0")"

COMMIT=32f1c1b92d1664dba542e927d23d86ffa57aa253     # electrum/tpch-dbgen
if [ ! -d tpch-dbgen ]; then
  git clone https://github.com/electrum/tpch-dbgen.git tpch-dbgen
fi
cd tpch-dbgen
git checkout -q "$COMMIT"
make -s CC=gcc
./dbgen -s 0.1 -f -q                                   # ~100 MB of .tbl files
mkdir -p ../queries
for q in $(seq 1 22); do
  DSS_QUERY=queries ./qgen -d "$q" > "../queries/q$q.raw.sql"   # default substitution parameters
done
cd ..
psql -d tpch -c "create extension if not exists hypopg"
echo "OK. Now: python run_experiments.py   (loads the data, then selects and benchmarks indexes)"
