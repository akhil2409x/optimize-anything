"""Mutation testing: what fault-detection does a reduced suite give up?

A mutant is the library with ONE small change (a seeded bug) on one line. The suite
"kills" a mutant if at least one test fails. We build a kill matrix
    kills[mutant] = set of tests that fail on it
once, and then evaluate any subset or ordering of tests from it without rerunning.

Speed-up that keeps the matrix exact: a test that never executes the mutated line
cannot see the change, so only the tests whose coverage includes that line are run.
Lines executed at import time are excluded (every test would 'cover' them).

Operators (one AST node, single-line nodes only, so line numbers never shift):
  comparison  <  <=  >  >=  ==  !=  is / is not  in / not in   (swapped with its partner)
  arithmetic  + <-> -    * <-> //    // -> *    % -> //
  boolean     and <-> or ;  not x -> x
  constant    integer k -> k + 1 ;  True <-> False
"""
from __future__ import annotations

import ast
import copy
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import collect as C

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
       ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}


@dataclass(frozen=True)
class Mutant:
    id: int
    file: str          # relative to the subject root
    line: int
    col: int
    end_col: int
    original: str
    replacement: str
    operator: str


def _variants(node):
    """Yield (operator name, mutated copy of node)."""
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            if type(op) in CMP:
                m = copy.deepcopy(node)
                m.ops[i] = CMP[type(op)]()
                yield f"cmp {type(op).__name__}->{CMP[type(op)].__name__}", m
    elif isinstance(node, ast.BinOp) and type(node.op) in BIN:
        m = copy.deepcopy(node)
        m.op = BIN[type(node.op)]()
        yield f"arith {type(node.op).__name__}->{type(m.op).__name__}", m
    elif isinstance(node, ast.BoolOp):
        m = copy.deepcopy(node)
        m.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        yield f"bool {type(node.op).__name__}->{type(m.op).__name__}", m
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield "not x->x", copy.deepcopy(node.operand)
    elif isinstance(node, ast.Constant) and type(node.value) is bool:
        yield "bool const", ast.Constant(not node.value)
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        yield "int k->k+1", ast.Constant(node.value + 1)


def generate(subject: Path, files, lines_by_file):
    """All mutants on the given lines. Nested replacements are fine because each mutant
    replaces exactly one node's source text."""
    out = []
    for rel in files:
        src = (subject / rel).read_text()
        src_lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        docstrings = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and n.body:
                b = n.body[0]
                if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant):
                    docstrings.add(id(b.value))
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) is None or node.lineno != node.end_lineno:
                continue
            if node.lineno not in lines_by_file.get(rel, ()) or id(node) in docstrings:
                continue
            orig = src_lines[node.lineno - 1].encode()[node.col_offset:node.end_col_offset].decode()
            for opname, new in _variants(node):
                rep = ast.unparse(new)
                if isinstance(node, (ast.Compare, ast.BinOp, ast.BoolOp)) or opname == "not x->x":
                    rep = f"({rep})"
                if rep.strip("()") == orig.strip("()"):
                    continue
                out.append((rel, node.lineno, node.col_offset, node.end_col_offset, orig, rep, opname))
    out.sort()
    return [Mutant(i, *m) for i, m in enumerate(out)]


def apply(subject: Path, m: Mutant, original_text: str):
    lines = original_text.splitlines(keepends=True)
    b = lines[m.line - 1].encode()
    lines[m.line - 1] = (b[:m.col] + m.replacement.encode() + b[m.end_col:]).decode()
    text = "".join(lines)
    compile(text, m.file, "exec")                    # mutant must still be valid Python
    (subject / m.file).write_text(text)


def make_workdir(dst: Path, src: Path = C.SUBJECT, patch: Path | None = None):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".tox", "docs"))
    if patch:
        subprocess.run(["patch", "-s", "-p1", "-i", str(patch)], cwd=dst, check=True)
    return dst


def _budget(tests, times):
    """Wall-clock allowance for running `tests` on a mutant: generous (3x + 10 s) relative
    to their measured time, so only a genuinely hung or runaway test exceeds it."""
    return 10 + 3 * sum(times.get(t, 1.0) for t in tests)


def run_mutant(workdir: Path, m: Mutant, tests, times):
    """Return the set of `tests` that fail on mutant m (None if the mutant does not compile).

    A mutant can make a test loop forever, sometimes inside one huge C-level integer
    operation that pytest-timeout's signal cannot interrupt. So the whole run gets a wall
    clock budget; if it is exceeded, every test is rerun on its own with its own budget,
    and a test that exceeds it counts as failing (in CI it would time out = fault detected)."""
    path = workdir / m.file
    original = path.read_text()
    try:
        apply(workdir, m, original)
    except SyntaxError:
        return None
    try:
        for d in workdir.rglob("__pycache__"):
            shutil.rmtree(d, ignore_errors=True)
        _, ok, _ = C.run_junit(tests, subject=workdir, wall_timeout=_budget(tests, times))
        if ok is not None:
            return {t for t in tests if not ok.get(t, False)}     # missing from the report = crashed
        failed = set()
        for t in tests:
            _, ok1, _ = C.run_junit([t], subject=workdir, wall_timeout=_budget([t], times))
            if ok1 is None or not ok1.get(t, False):
                failed.add(t)
        return failed
    finally:
        path.write_text(original)


def sample(mutants, n, seed=0):
    if n >= len(mutants):
        return list(mutants)
    return sorted(random.Random(seed).sample(list(mutants), n), key=lambda m: m.id)


def _worker(args):
    workdir, items, times, log = args
    out = []
    for m, tests in items:
        k = run_mutant(Path(workdir), m, tests, times)
        out.append((m.id, k))
        with open(log, "a") as f:                       # progress log, lets an interrupted run resume
            f.write(json.dumps([m.id, sorted(k) if k is not None else None]) + "\n")
    return out


def kill_matrix(mutants, covering, workdirs, times, log: Path, procs=None):
    """covering[mutant id] -> tests to run. Runs the mutants on len(workdirs) processes
    (one private copy of the subject per process). Results already in `log` are reused.
    Returns {mutant id: failing tests}."""
    import multiprocessing as mp
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            mid, k = json.loads(line)
            done[mid] = set(k) if k is not None else None
    procs = procs or len(workdirs)
    jobs = [(m, sorted(covering[m.id])) for m in mutants if covering[m.id] and m.id not in done]
    # most expensive first, dealt round-robin: an LPT-style balance of the two processes
    jobs.sort(key=lambda j: -_budget(j[1], times))
    chunks = [[] for _ in range(procs)]
    for i, j in enumerate(jobs):
        chunks[i % procs].append(j)
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(_worker, [(str(workdirs[i]), chunks[i], times, str(log)) for i in range(procs)],
                       chunksize=1)
    done.update({mid: k for part in res for mid, k in part})
    return {m.id: done[m.id] for m in mutants if covering[m.id]}


def covering_tests(mutants, cov):
    by_line = {}
    for t, rs in cov.items():
        for r in rs:
            if r[0] == "L":
                by_line.setdefault((r[1], r[2]), set()).add(t)
    return {m.id: by_line.get((m.file, m.line), set()) for m in mutants}


def mutable_lines(cov, import_time):
    lines = {}
    imp = {(r[1], r[2]) for r in import_time if r[0] == "L"}
    for rs in cov.values():
        for r in rs:
            if r[0] == "L" and (r[1], r[2]) not in imp:
                lines.setdefault(r[1], set()).add(r[2])
    return lines


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
