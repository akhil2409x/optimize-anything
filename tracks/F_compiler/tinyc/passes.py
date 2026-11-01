"""Optimization passes on the IR. Each pass mutates the Function and returns
True if it changed anything. `optimize` runs a pipeline until nothing changes.

  constprop  global constant propagation + folding (dataflow over the CFG),
             folds branches on constants
  copyprop   local copy propagation, plus copy coalescing:
             `%t = op ...; v = copy %t` (single-use temp) -> `v = op ...`
  cse        local common sub-expression elimination (value numbering)
  strength   strength reduction / algebraic simplification:
             x*2^k -> x shl k, x*1 -> x, x*0 -> 0, x+0, x-0, x-x, x/1, x%1, x*-1 -> neg
             (NOT x/2^k -> shr: wrong for negative x under truncating division)
  dce        dead-code elimination (global liveness), unreachable blocks,
             empty-block bypass
  licm       loop-invariant code motion into a loop preheader

Correctness rules shared by all passes:
  * `print` and possibly-trapping `div`/`mod` are never removed or moved;
  * reading an unassigned variable is a runtime error in Tiny, so constant
    propagation treats "unassigned" as its own lattice value, never as a constant.
"""
from __future__ import annotations

from .ir import COMMUTATIVE, OPFUNC, PURE_BINOPS, Block, Function, Instr

NAC = "NAC"          # not a constant
UNINIT = "UNINIT"    # never assigned on some path


# ------------------------------------------------------------------ analyses
def rpo(fn: Function):
    seen, order = set(), []

    def dfs(l):
        seen.add(l)
        for s in fn.blocks[l].succs():
            if s not in seen:
                dfs(s)
        order.append(l)
    dfs(fn.entry)
    return order[::-1]


def liveness(fn: Function):
    live_in = {l: set() for l in fn.blocks}
    live_out = {l: set() for l in fn.blocks}
    changed = True
    while changed:
        changed = False
        for l in reversed(rpo(fn)):
            b = fn.blocks[l]
            out = set().union(*(live_in[s] for s in b.succs())) if b.succs() else set()
            live = set(out) | set(b.term.uses())
            for i in reversed(b.instrs):
                live -= set(i.defs())
                live |= set(i.uses())
            if out != live_out[l] or live != live_in[l]:
                live_out[l], live_in[l] = out, live
                changed = True
    return live_in, live_out


def dominators(fn: Function):
    order = rpo(fn)
    preds = fn.preds()
    dom = {l: set(order) for l in order}
    dom[fn.entry] = {fn.entry}
    changed = True
    while changed:
        changed = False
        for l in order[1:]:
            ps = [p for p in preds[l] if p in dom]
            new = set.intersection(*(dom[p] for p in ps)) | {l} if ps else {l}
            if new != dom[l]:
                dom[l] = new
                changed = True
    return dom


def natural_loops(fn: Function):
    """[(header, set_of_blocks)], innermost (smallest) first; loops sharing a header merged."""
    dom = dominators(fn)
    preds = fn.preds()
    loops = {}
    for b in dom:
        for h in fn.blocks[b].succs():
            if h in dom[b]:                         # back edge b -> h
                body, stack = {h, b}, [b]
                while stack:
                    x = stack.pop()
                    for p in preds[x]:
                        if p not in body and p in dom:
                            body.add(p)
                            stack.append(p)
                loops.setdefault(h, set()).update(body)
    return sorted(loops.items(), key=lambda kv: len(kv[1]))


# ------------------------------------------------------------------ helpers
def _fold(op, a, b=None):
    """Fold a pure computation on constants; None if it would trap or is not foldable."""
    if op == "const":
        return a
    if op == "copy":
        return a
    if op == "neg":
        return -a
    if op in ("div", "mod") and b == 0:
        return None
    return OPFUNC[op](a, b)


def _canon(op, a, b):
    if op in COMMUTATIVE and repr(a) > repr(b):
        a, b = b, a
    return (op, a, b)


# ------------------------------------------------------------------ constant propagation
def _meet(states):
    states = [s for s in states if s is not None]
    if not states:
        return None
    keys = set().union(*states)
    out = {}
    for k in keys:
        vals = {s.get(k, UNINIT) for s in states}
        out[k] = vals.pop() if len(vals) == 1 else NAC
    return out


def _transfer(state, instrs):
    st = dict(state)

    def val(x):
        return x if isinstance(x, int) else st.get(x, UNINIT)
    for i in instrs:
        if i.dst is None:
            continue
        if i.op == "const":
            st[i.dst] = i.a
        elif i.op in ("copy", "neg"):
            v = val(i.a)
            st[i.dst] = (v if i.op == "copy" else -v) if isinstance(v, int) else NAC
        else:
            a, b = val(i.a), val(i.b)
            r = _fold(i.op, a, b) if isinstance(a, int) and isinstance(b, int) else None
            st[i.dst] = r if r is not None else NAC
    return st


def constprop(fn: Function) -> bool:
    order = rpo(fn)
    preds = fn.preds()
    IN = {l: None for l in order}
    OUT = {l: None for l in order}
    IN[fn.entry] = {}
    changed = True
    while changed:
        changed = False
        for l in order:
            if l != fn.entry:
                IN[l] = _meet([OUT[p] for p in preds[l] if p in OUT])
            if IN[l] is None:
                continue
            new = _transfer(IN[l], fn.blocks[l].instrs)
            if new != OUT[l]:
                OUT[l] = new
                changed = True
    did = False
    for l in order:
        if IN[l] is None:
            continue
        st = dict(IN[l])
        b = fn.blocks[l]
        new_instrs = []
        for i in b.instrs:
            for f in ("a", "b"):
                x = getattr(i, f)
                if isinstance(x, str) and isinstance(st.get(x), int):
                    setattr(i, f, st[x])
                    did = True
            if i.dst is not None and i.op != "const":
                if i.op in ("copy", "neg") and isinstance(i.a, int):
                    i.a, i.op = (i.a if i.op == "copy" else -i.a), "const"
                    did = True
                elif i.op not in ("copy", "neg") and isinstance(i.a, int) and isinstance(i.b, int):
                    r = _fold(i.op, i.a, i.b)
                    if r is not None:
                        i.op, i.a, i.b = "const", r, None
                        did = True
            st = _transfer(st, [i])
            new_instrs.append(i)
        b.instrs = new_instrs
        t = b.term
        if t.op == "cbr":
            if isinstance(t.a, str) and isinstance(st.get(t.a), int):
                t.a = st[t.a]
                did = True
            if isinstance(t.a, int):
                b.term = Instr("jmp", targets=(t.targets[0] if t.a != 0 else t.targets[1],))
                did = True
    return did


# ------------------------------------------------------------------ copy propagation + coalescing
def copyprop(fn: Function) -> bool:
    did = False
    for b in fn.blocks.values():
        copies = {}
        for i in b.instrs + [b.term]:
            for f in ("a", "b"):
                x = getattr(i, f)
                if isinstance(x, str) and x in copies:
                    setattr(i, f, copies[x])
                    did = True
            for d in i.defs():
                copies = {k: v for k, v in copies.items() if k != d and v != d}
            if i.op == "copy" and isinstance(i.a, str) and i.a != i.dst:
                copies[i.dst] = i.a
    # coalescing: `%t = op ...` immediately followed by `v = copy %t`, %t used nowhere else
    uses = {}
    for b in fn.blocks.values():
        for i in b.instrs + [b.term]:
            for u in i.uses():
                uses[u] = uses.get(u, 0) + 1
    for b in fn.blocks.values():
        k = 0
        while k + 1 < len(b.instrs):
            i, j = b.instrs[k], b.instrs[k + 1]
            if (i.dst is not None and i.dst.startswith("%t") and j.op == "copy" and j.a == i.dst
                    and uses.get(i.dst, 0) == 1):
                i.dst = j.dst
                del b.instrs[k + 1]
                did = True
            else:
                k += 1
    return did


# ------------------------------------------------------------------ CSE (local value numbering)
def cse(fn: Function) -> bool:
    did = False
    for b in fn.blocks.values():
        table = {}
        for i in b.instrs:
            key = None
            if i.op in OPFUNC:                    # includes div/mod: a repeat cannot trap anew
                key = _canon(i.op, i.a, i.b)
            elif i.op == "neg":
                key = ("neg", i.a, None)
            if key is not None and key in table and table[key] != i.dst:
                i.op, i.a, i.b = "copy", table[key], None
                did = True
                key = None
            for d in i.defs():
                table = {k: v for k, v in table.items() if v != d and d not in (k[1], k[2])}
            if key is not None and i.dst not in (key[1], key[2]):
                table[key] = i.dst
    return did


# ------------------------------------------------------------------ strength reduction
def _pow2(n):
    return isinstance(n, int) and n > 1 and n & (n - 1) == 0


def strength(fn: Function) -> bool:
    did = False
    for b in fn.blocks.values():
        for i in b.instrs:
            op, a, c = i.op, i.a, i.b
            new = None
            if op == "mul" and isinstance(a, int) and not isinstance(c, int):
                a, c = c, a                                   # constant on the right
                i.a, i.b = a, c
            if op == "mul" and c == 0:
                new = ("const", 0, None)
            elif op == "mul" and c == 1:
                new = ("copy", a, None)
            elif op == "mul" and c == -1:
                new = ("neg", a, None)
            elif op == "mul" and _pow2(c):
                new = ("shl", a, c.bit_length() - 1)
            elif op == "add" and c == 0:
                new = ("copy", a, None)
            elif op == "add" and a == 0:
                new = ("copy", c, None)
            elif op == "sub" and c == 0:
                new = ("copy", a, None)
            elif op == "sub" and isinstance(a, str) and a == c:
                new = ("const", 0, None)
            elif op == "div" and c == 1:
                new = ("copy", a, None)
            elif op == "mod" and c in (1, -1):
                new = ("const", 0, None)
            if new:
                i.op, i.a, i.b = new
                did = True
    return did


# ------------------------------------------------------------------ dead-code elimination
def dce(fn: Function) -> bool:
    did = False
    reach = fn.reachable()
    for l in list(fn.blocks):
        if l not in reach:
            del fn.blocks[l]
            did = True
    # bypass empty blocks that just jump on (not the entry)
    while True:
        target = {l: b.term.targets[0] for l, b in fn.blocks.items()
                  if l != fn.entry and not b.instrs and b.term.op == "jmp" and b.term.targets[0] != l}
        if not target:
            break
        def final(l, seen=()):
            while l in target and l not in seen:
                seen += (l,)
                l = target[l]
            return l
        moved = False
        for b in fn.blocks.values():
            new = tuple(final(t) for t in b.term.targets)
            if new != b.term.targets:
                b.term.targets = new
                moved = True
        for l in list(fn.blocks):
            if l not in fn.reachable():
                del fn.blocks[l]
                moved = True
        did |= moved
        if not moved:
            break
    for b in fn.blocks.values():                    # cbr to the same block twice -> jmp
        if b.term.op == "cbr" and b.term.targets[0] == b.term.targets[1]:
            b.term = Instr("jmp", targets=(b.term.targets[0],))
            did = True
    while True:
        _, live_out = liveness(fn)
        removed = False
        for l, b in fn.blocks.items():
            live = set(live_out[l]) | set(b.term.uses())
            keep = []
            for i in reversed(b.instrs):
                if i.dst is not None and i.dst not in live and not i.has_side_effect():
                    removed = True
                    continue
                if i.op == "copy" and i.a == i.dst:
                    removed = True
                    continue
                keep.append(i)
                live -= set(i.defs())
                live |= set(i.uses())
            b.instrs = keep[::-1]
        did |= removed
        if not removed:
            return did


# ------------------------------------------------------------------ LICM
def _preheader(fn: Function, header: str, body: set) -> Block:
    preds = [p for p in fn.preds()[header] if p not in body]
    if len(preds) == 1:
        p = fn.blocks[preds[0]]
        if p.term.op == "jmp":
            return p
    name = f"{header}_pre"
    k = 0
    while name in fn.blocks:
        k += 1
        name = f"{header}_pre{k}"
    pre = Block(name, [], Instr("jmp", targets=(header,)))
    for p in preds:
        t = fn.blocks[p].term
        t.targets = tuple(name if x == header else x for x in t.targets)
    blocks = {}
    for l, b in fn.blocks.items():                 # keep the new block just before the header
        if l == header:
            blocks[name] = pre
        blocks[l] = b
    fn.blocks.clear()
    fn.blocks.update(blocks)
    return pre


def licm(fn: Function) -> bool:
    did = False
    for header, _ in natural_loops(fn):
        loops = dict(natural_loops(fn))             # re-analyse: earlier hoists change the CFG
        if header not in loops:
            continue
        body = loops[header]
        live_in, _ = liveness(fn)
        exits = {s for l in body for s in fn.blocks[l].succs() if s not in body}
        defs_in_loop = {}
        for l in body:
            for i in fn.blocks[l].instrs:
                for d in i.defs():
                    defs_in_loop[d] = defs_in_loop.get(d, 0) + 1
        hoisted = []
        order = [l for l in rpo(fn) if l in body]
        progress = True
        while progress:
            progress = False
            for l in order:
                b = fn.blocks[l]
                for i in list(b.instrs):
                    pure = i.op in PURE_BINOPS or i.op in ("const", "copy", "neg") or \
                        (i.op in ("div", "mod") and not i.may_trap())
                    if not pure or i.dst is None:
                        continue
                    if defs_in_loop.get(i.dst, 0) != 1 or i.dst in live_in[header]:
                        continue
                    if any(i.dst in live_in[e] for e in exits):
                        continue
                    if any(isinstance(u, str) and defs_in_loop.get(u, 0) > 0 for u in i.uses()):
                        continue
                    b.instrs.remove(i)
                    hoisted.append(i)
                    defs_in_loop[i.dst] = 0           # now defined outside the loop
                    progress = True
        if hoisted:
            pre = _preheader(fn, header, body)
            pre.instrs.extend(hoisted)
            did = True
    return did


# ------------------------------------------------------------------ pipeline
PASSES = {"constprop": constprop, "copyprop": copyprop, "cse": cse,
          "strength": strength, "dce": dce, "licm": licm}
DEFAULT_ORDER = ["constprop", "copyprop", "cse", "strength", "licm", "dce"]


def optimize(fn: Function, passes=None, max_rounds: int = 10, stats=None, cleanup: bool = True) -> Function:
    """Run `passes` repeatedly until nothing changes. `cleanup` runs a final DCE
    (unreachable/empty blocks, dead code); the ablation study turns it off so
    that "without dce" really means without it."""
    passes = DEFAULT_ORDER if passes is None else passes
    for _ in range(max_rounds):
        changed = False
        for name in passes:
            c = PASSES[name](fn)
            if stats is not None and c:
                stats[name] = stats.get(name, 0) + 1
            changed |= c
        if not changed:
            break
    if cleanup:
        dce(fn)
    return fn
