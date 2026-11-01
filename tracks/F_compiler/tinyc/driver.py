"""Compiler driver: source -> AST -> IR -> (passes) -> register allocation -> VM code.

Optimization levels used in the experiments:
    O0  no IR passes, no register allocation (every variable in memory)
    O1  no IR passes, greedy register allocation
    O2  all IR passes, greedy register allocation
    O3  all IR passes, ILP (optimal) register allocation
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from . import backend as B, frontend as F, ir, passes as P

LEVELS = {
    "O0": dict(passes=[], ra="none"),
    "O1": dict(passes=[], ra="greedy"),
    "O2": dict(passes=P.DEFAULT_ORDER, ra="greedy"),
    "O3": dict(passes=P.DEFAULT_ORDER, ra="ilp"),
}


@dataclass
class Compiled:
    code: list
    fn: ir.Function
    alloc: B.Allocation
    compile_seconds: float
    ir_instrs: int
    pass_stats: dict


def compile_source(src: str, level: str = "O2", k: int = 6, passes=None, ra=None,
                   cleanup: bool = True) -> Compiled:
    cfg = LEVELS[level]
    passes = cfg["passes"] if passes is None else passes
    ra = cfg["ra"] if ra is None else ra
    t0 = time.perf_counter()
    fn = ir.lower(F.parse(src))
    stats = {}
    if passes:
        P.optimize(fn, passes, stats=stats, cleanup=cleanup)
    alloc = B.allocate(fn, k, ra)
    B.check_allocation(fn, alloc)
    code = B.codegen(fn, alloc)
    return Compiled(code, fn, alloc, time.perf_counter() - t0, fn.instr_count(), stats)


def reference(src: str):
    """(printed output, error kind or None) from the AST interpreter."""
    try:
        return (F.interpret(F.parse(src)), None)
    except F.TinyRuntimeError as e:
        return (e.partial_output, _norm(str(e)))


def _norm(msg: str) -> str:
    return "division by zero" if "division by zero" in msg else ("step limit" if "step" in msg else "other")


def observed(stats: B.RunStats):
    """(printed output, error kind or None) from a compiled program's VM run.
    Comparing both catches an optimization that drops output printed before an error."""
    return (stats.output, _norm(stats.error) if stats.error else None)
