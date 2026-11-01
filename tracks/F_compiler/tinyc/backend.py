"""Back end: register allocation (graph colouring), code generation, and a
register virtual machine with an instruction/cycle counter.

Target machine
    K allocatable registers r0..r{K-1}, plus two scratch registers s0, s1
    reserved for loading spilled values (as real compilers reserve some).
    Spilled values live in memory slots, accessed with ld/st.
    Cycle costs:  mul 3, div/mod 20, ld/st 3, everything else 1.

Register allocation as an optimization problem (Chapter 6):
    Interference graph G = (V, E): one node per IR variable, an edge when two
    variables are live at the same time. Assign each node a register (colour)
    so neighbours differ, or spill it to memory.
        minimise  sum_v cost_v * spill_v
        s.t.      sum_r x_{v,r} + spill_v = 1            for every v
                  x_{u,r} + x_{v,r} <= 1                  for every edge (u,v), register r
                  x, spill binary
    cost_v = sum over v's uses and defs of 10^(loop depth)   (the usual estimate).
    Methods:
      'none'   : no allocation, every variable in memory (what -O0 compilers do)
      'greedy' : Chaitin-Briggs simplify/select with optimistic colouring,
                 spill choice by lowest cost/degree
      'ilp'    : the model above, solved exactly by HiGHS branch-and-bound
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pulp

from .frontend import TinyRuntimeError, c_div, c_mod
from .ir import Function
from .passes import liveness, natural_loops

CYCLES = {"mul": 3, "div": 20, "mod": 20, "ld": 3, "st": 3}


# ------------------------------------------------------------------ interference graph
def interference(fn: Function):
    _, live_out = liveness(fn)
    nodes, edges = set(), set()
    for l, b in fn.blocks.items():
        live = set(live_out[l]) | set(b.term.uses())
        nodes |= live
        for i in reversed(b.instrs):
            nodes |= set(i.uses()) | set(i.defs())
            for d in i.defs():
                for v in live:
                    if v != d and not (i.op == "copy" and v == i.a):
                        edges.add(tuple(sorted((d, v))))
                live.discard(d)
            live |= set(i.uses())
    return nodes, edges


def spill_costs(fn: Function):
    depth = {l: 0 for l in fn.blocks}
    for _, body in natural_loops(fn):
        for l in body:
            depth[l] += 1
    cost = {}
    for l, b in fn.blocks.items():
        w = 10 ** depth[l]
        for i in b.instrs + [b.term]:
            for v in i.uses() + i.defs():
                cost[v] = cost.get(v, 0) + w
    return cost


@dataclass
class Allocation:
    loc: dict                      # var -> "rN" or ("mem", slot)
    method: str
    k: int
    spill_cost: float
    spilled: int
    seconds: float
    nodes: int
    edges: int
    status: str = ""
    extra: dict = field(default_factory=dict)


def _finish(fn, colours, k, method, t0, nodes, edges, cost, status=""):
    loc, slot = {}, 0
    for v in sorted(nodes):
        if v in colours:
            loc[v] = f"r{colours[v]}"
        else:
            loc[v] = ("mem", slot)
            slot += 1
    spilled = [v for v in nodes if v not in colours]
    return Allocation(loc, method, k, sum(cost.get(v, 0) for v in spilled), len(spilled),
                      time.perf_counter() - t0, len(nodes), len(edges), status)


def allocate(fn: Function, k: int, method: str = "greedy", time_limit: float = 30) -> Allocation:
    t0 = time.perf_counter()
    nodes, edges = interference(fn)
    cost = spill_costs(fn)
    adj = {v: set() for v in nodes}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    if method == "none":
        return _finish(fn, {}, k, method, t0, nodes, edges, cost)
    if method == "greedy":
        stack, removed = [], set()
        while len(removed) < len(nodes):
            deg = {v: len(adj[v] - removed) for v in nodes if v not in removed}
            low = [v for v, d in deg.items() if d < k]
            if low:
                v = min(low)                          # deterministic
            else:                                     # optimistic spill candidate
                v = min(deg, key=lambda u: (cost.get(u, 0) / max(deg[u], 1), u))
            stack.append(v)
            removed.add(v)
        colours = {}
        for v in reversed(stack):
            used = {colours[u] for u in adj[v] if u in colours}
            free = [c for c in range(k) if c not in used]
            if free:
                colours[v] = free[0]
        return _finish(fn, colours, k, method, t0, nodes, edges, cost)
    if method == "ilp":
        vs = sorted(nodes)
        prob = pulp.LpProblem("regalloc", pulp.LpMinimize)
        x = {(v, r): pulp.LpVariable(f"x_{n}_{r}", cat="Binary") for n, v in enumerate(vs) for r in range(k)}
        s = {v: pulp.LpVariable(f"s_{n}", cat="Binary") for n, v in enumerate(vs)}
        prob += pulp.lpSum(cost.get(v, 0) * s[v] for v in vs)
        for v in vs:
            prob += pulp.lpSum(x[v, r] for r in range(k)) + s[v] == 1
        for a, b in edges:
            for r in range(k):
                prob += x[a, r] + x[b, r] <= 1
        prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0.0))
        status = "Optimal" if prob.sol_status == pulp.LpSolutionOptimal else "TimeLimit"
        colours = {v: r for v in vs for r in range(k) if (x[v, r].value() or 0) > 0.5}
        return _finish(fn, colours, k, method, t0, nodes, edges, cost, status)
    raise ValueError(method)


def check_allocation(fn: Function, alloc: Allocation):
    """Independent validity check: interfering variables never share a register."""
    _, edges = interference(fn)
    for a, b in edges:
        la, lb = alloc.loc.get(a), alloc.loc.get(b)
        if isinstance(la, str) and la == lb:
            raise AssertionError(f"{a} and {b} interfere but share {la}")


# ------------------------------------------------------------------ code generation
def codegen(fn: Function, alloc: Allocation):
    code, labels, fixups = [], {}, []
    order = list(fn.blocks)

    def operand(x, scratch):
        """Return a register (or immediate when allowed) holding x."""
        if isinstance(x, int):
            code.append(("li", scratch, x))
            return scratch
        loc = alloc.loc[x]
        if isinstance(loc, str):
            return loc
        code.append(("ld", scratch, loc[1]))
        return scratch

    def dest(d):
        loc = alloc.loc[d]
        return (loc, None) if isinstance(loc, str) else ("s0", loc[1])

    for n, l in enumerate(order):
        labels[l] = len(code)
        b = fn.blocks[l]
        for i in b.instrs:
            if i.op == "print":
                code.append(("print", operand(i.a, "s0")))
                continue
            rd, slot = dest(i.dst)
            if i.op == "const":
                code.append(("li", rd, i.a))
            elif i.op == "copy":
                if isinstance(i.a, int):
                    code.append(("li", rd, i.a))
                else:
                    src = operand(i.a, "s1")
                    if src != rd:
                        code.append(("mov", rd, src))
            elif i.op == "neg":
                code.append(("neg", rd, operand(i.a, "s1")))
            else:
                ra = operand(i.a, "s0")
                rb = i.b if isinstance(i.b, int) else operand(i.b, "s1")
                code.append((i.op, rd, ra, rb))
            if slot is not None:
                code.append(("st", "s0", slot))
        t = b.term
        nxt = order[n + 1] if n + 1 < len(order) else None
        if t.op == "halt":
            code.append(("halt",))
        elif t.op == "jmp":
            if t.targets[0] != nxt:
                fixups.append(len(code)); code.append(("jmp", t.targets[0]))
        else:
            r = operand(t.a, "s0")
            tt, ff = t.targets
            if ff == nxt:
                fixups.append(len(code)); code.append(("bnz", r, tt))
            elif tt == nxt:
                fixups.append(len(code)); code.append(("bz", r, ff))
            else:
                fixups.append(len(code)); code.append(("bnz", r, tt))
                fixups.append(len(code)); code.append(("jmp", ff))
    for k in fixups:
        ins = code[k]
        code[k] = ins[:-1] + (labels[ins[-1]],)
    return code


# ------------------------------------------------------------------ virtual machine
VMOPS = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": c_div, "mod": c_mod, "shl": lambda a, b: a << b,
    "lt": lambda a, b: int(a < b), "le": lambda a, b: int(a <= b), "gt": lambda a, b: int(a > b),
    "ge": lambda a, b: int(a >= b), "eq": lambda a, b: int(a == b), "ne": lambda a, b: int(a != b),
}


@dataclass
class RunStats:
    output: list
    instructions: int
    cycles: int
    loads: int
    stores: int
    seconds: float
    error: str | None = None


def run_vm(code, max_steps: int = 50_000_000) -> RunStats:
    regs, mem, out = {}, {}, []
    pc = n = cycles = loads = stores = 0
    t0 = time.perf_counter()
    err = None

    def rv(x):
        if isinstance(x, int):
            return x
        if x not in regs:
            raise TinyRuntimeError(f"read of uninitialised register {x}")
        return regs[x]
    try:
        while True:
            ins = code[pc]
            op = ins[0]
            n += 1
            cycles += CYCLES.get(op, 1)
            if n > max_steps:
                raise TinyRuntimeError("step limit exceeded")
            pc += 1
            if op == "li":
                regs[ins[1]] = ins[2]
            elif op == "mov":
                regs[ins[1]] = rv(ins[2])
            elif op == "neg":
                regs[ins[1]] = -rv(ins[2])
            elif op == "ld":
                if ins[2] not in mem:
                    raise TinyRuntimeError("read of uninitialised memory slot")
                regs[ins[1]] = mem[ins[2]]; loads += 1
            elif op == "st":
                mem[ins[2]] = rv(ins[1]); stores += 1
            elif op == "print":
                out.append(rv(ins[1]))
            elif op == "jmp":
                pc = ins[1]
            elif op == "bnz":
                if rv(ins[1]) != 0:
                    pc = ins[2]
            elif op == "bz":
                if rv(ins[1]) == 0:
                    pc = ins[2]
            elif op == "halt":
                break
            else:
                regs[ins[1]] = VMOPS[op](rv(ins[2]), rv(ins[3]))
    except TinyRuntimeError as e:
        err = str(e)
    return RunStats(out, n, cycles, loads, stores, time.perf_counter() - t0, err)
