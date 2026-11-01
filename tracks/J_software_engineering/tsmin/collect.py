"""Measure the subject's test suite: per-test time and per-test coverage.

Per-test time    : pytest --junitxml, WITHOUT coverage (coverage tracing slows tests
                   unevenly), median of several runs. Each test's time includes its
                   setup and teardown.
Per-test coverage: coverage.py with dynamic contexts (pytest-cov --cov-context=test),
                   statement and branch (arc) coverage of the library package only.
                   Read straight from coverage's SQLite database.

Requirements are ('L', file, line) for statements and ('A', file, from, to) for
branch arcs. Only requirements covered by EVERY full-suite coverage run are kept;
anything covered only in some runs is flaky and reported separately.
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics as st
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from coverage.numbits import numbits_to_nums

HERE = Path(__file__).resolve().parent.parent
SUBJECT = HERE / "more-itertools"
PACKAGE = "more_itertools"


def pytest_cmd(extra, tests=None, cwd=SUBJECT):
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
            *extra, *(tests or ["tests"])]


def env(subject=SUBJECT):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    e["PYTHONPATH"] = str(subject)
    return e


def node_id(classname: str, name: str) -> str:
    """junit 'tests.test_more.ChunkedTests' + 'test_even' -> 'tests/test_more.py::ChunkedTests::test_even'"""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        if (SUBJECT / ("/".join(parts[:i]) + ".py")).exists():
            return "/".join(parts[:i]) + ".py::" + "::".join(parts[i:] + [name])
    raise ValueError(classname)


def run_junit(tests=None, subject=SUBJECT, timeout_per_test=None, wall_timeout=None):
    """Run pytest once; return ({node_id: seconds}, {node_id: passed?}, wall seconds)."""
    import time
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "j.xml"
        extra = [f"--junitxml={xml}", "-o", "junit_family=xunit2"]
        if timeout_per_test:
            extra += [f"--timeout={timeout_per_test}"]
        t0 = time.perf_counter()
        try:
            subprocess.run(pytest_cmd(extra, tests), cwd=subject, env=env(subject), capture_output=True,
                           timeout=wall_timeout)
        except subprocess.TimeoutExpired:
            return None, None, time.perf_counter() - t0
        wall = time.perf_counter() - t0
        if not xml.exists():
            return {}, {}, wall
        times, ok = {}, {}
        for tc in ET.parse(xml).getroot().iter("testcase"):
            nid = node_id(tc.get("classname"), tc.get("name"))
            times[nid] = float(tc.get("time", 0))
            ok[nid] = not any(ch.tag in ("failure", "error") for ch in tc)
        return times, ok, wall


def measure_times(runs=3):
    all_runs, walls = [], []
    for _ in range(runs):
        t, ok, w = run_junit()
        assert all(ok.values()), [k for k, v in ok.items() if not v]
        all_runs.append(t)
        walls.append(w)
    tests = sorted(all_runs[0])
    return {t: st.median(r[t] for r in all_runs) for t in tests}, walls


def run_coverage(db: Path, tests=None, subject=SUBJECT):
    cmd = pytest_cmd([f"--cov={PACKAGE}", "--cov-branch", "--cov-context=test", "--cov-report="], tests)
    subprocess.run(cmd, cwd=subject, env=dict(env(subject), COVERAGE_FILE=str(db)), check=True,
                   capture_output=True)


def coverage_of(tests, subject=SUBJECT):
    """Union of requirements covered when ONLY `tests` are run (fresh process), plus the
    per-test map. Used to verify a reduced suite for real, not from the stored matrix."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "cov"
        run_coverage(db, tests, subject)
        per = read_coverage(db, subject)
    per.pop(None, None)
    return set().union(*per.values()) if per else set(), per


def read_coverage(db: Path, subject=SUBJECT):
    """{test node id: set of requirements} from a coverage SQLite database."""
    con = sqlite3.connect(db)
    files = {fid: os.path.relpath(path, subject) for fid, path in con.execute("select id, path from file")}
    ctx = {}
    for cid, name in con.execute("select id, context from context"):
        if not name:
            ctx[cid] = None                       # code run at import/collection time, not by a test
            continue
        ctx[cid] = name.rsplit("|", 1)[0]          # 'tests/x.py::C::t|run' -> node id (merge setup/run/teardown)
    req = defaultdict(set)
    for fid, cid, bits in con.execute("select file_id, context_id, numbits from line_bits"):
        for ln in numbits_to_nums(bits):
            req[ctx[cid]].add(("L", files[fid], ln))
    for fid, cid, a, b in con.execute("select file_id, context_id, fromno, tono from arc"):
        req[ctx[cid]].add(("A", files[fid], a, b))
    con.close()
    return {k: with_lines(v) for k, v in req.items()}


def with_lines(reqs):
    """With branch coverage on, coverage.py stores only arcs; executed lines are the arc
    endpoints > 0 (negative numbers mark function entry/exit). Add them as 'L' requirements."""
    out = set(reqs)
    for r in reqs:
        if r[0] == "A":
            for ln in r[2:]:
                if ln > 0:
                    out.add(("L", r[1], ln))
    return out


def collect(out: Path, coverage_runs=2, time_runs=3):
    """Measure everything once and save it to `out` (JSON)."""
    times, walls = measure_times(time_runs)
    covs = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(coverage_runs):
            db = Path(td) / f"cov{i}"
            run_coverage(db)
            covs.append(read_coverage(db))
    import_time = set().union(*(c.get(None, set()) for c in covs))
    tests = sorted(times)
    stable, flaky = {}, {}
    for t in tests:
        sets = [c.get(t, set()) for c in covs]
        stable[t] = set.intersection(*sets)
        flaky[t] = set.union(*sets) - stable[t]
    data = dict(
        subject=str(SUBJECT.name),
        commit=subprocess.run(["git", "-C", str(SUBJECT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
        times=times, full_suite_wall=walls,
        coverage={t: sorted(map(list, stable[t])) for t in tests},
        flaky={t: sorted(map(list, flaky[t])) for t in tests if flaky[t]},
        import_time=sorted(map(list, import_time)),
    )
    out.write_text(json.dumps(data))
    return data


def load(path: Path):
    d = json.loads(path.read_text())
    d["coverage"] = {t: with_lines({tuple(r) for r in v}) for t, v in d["coverage"].items()}
    d["flaky"] = {t: {tuple(r) for r in v} for t, v in d["flaky"].items()}
    d["import_time"] = with_lines({tuple(r) for r in d["import_time"]})
    return d
