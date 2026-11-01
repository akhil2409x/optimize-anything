"""Random Tiny programs for differential testing.

Every generated program
  * re-uses earlier expressions and a small pool of variable names, so that
    the same expression appears again after one of its operands changed (the
    case CSE must handle), and variable dividends are often negative and odd
    (the case a wrong x/2^k -> shift rewrite gets wrong). An earlier version
    without this missed two deliberately injected bugs (failure log F10);
  * terminates: loops are `cK = 0; while (cK < N) { ...; cK = cK + 1; }` and the
    body never assigns its own counter;
  * reads only variables that are definitely assigned on every path;
  * freely uses / and % with divisors that may be zero, negative numbers,
    repeated sub-expressions, constants and loop-invariant expressions, so
    every pass has something to do and every pass's trap rules are exercised.
"""
from __future__ import annotations

import random
import re


def _names(expr: str):
    return set(re.findall(r"[a-z]\w*", expr))


class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.nloop = 0
        self.seen = []                 # earlier expressions, re-emitted on purpose

    def expr(self, vars_, depth=0):
        r = self.r
        if depth == 0 and self.seen and r.random() < 0.35:
            cand = [e for e in self.seen if all(v in vars_ for v in _names(e))]
            if cand:
                return r.choice(cand)
        e = self._expr(vars_, depth)
        if depth == 0:
            self.seen.append(e)
        return e

    def _expr(self, vars_, depth):
        r = self.r
        if depth > 2 or r.random() < 0.3:
            if vars_ and r.random() < 0.6:
                return r.choice(sorted(vars_))
            return str(r.choice([0, 1, 2, 3, 4, 7, 8, 16, -1, -5, -7, -9, 10, 100]))
        op = r.choice(["+", "-", "*", "*", "/", "%", "<", "==", "!=", ">="])
        a, b = self._expr(vars_, depth + 1), self._expr(vars_, depth + 1)
        if op in "/%" and r.random() < 0.8:
            b = str(r.choice([2, 3, 4, 7, 8, -3]))      # mostly safe divisors, sometimes risky ones
        if r.random() < 0.15:
            return f"-({a})"
        return f"({a} {op} {b})"

    def block(self, vars_, counters, depth, n):
        out, assigned = [], set(vars_)
        for _ in range(n):
            k = self.r.random()
            names = ["a", "b", "c"]
            if k < 0.45 or not assigned:
                v = self.r.choice(names)
                out.append(f"{v} = {self.expr(assigned | counters)};")
                assigned.add(v)
            elif k < 0.65:
                out.append(f"print {self.expr(assigned | counters)};")
            elif k < 0.8 and depth < 2:
                then, _ = self.block(assigned, counters, depth + 1, self.r.randint(1, 3))
                other, _ = self.block(assigned, counters, depth + 1, self.r.randint(0, 2))
                out.append(f"if ({self.expr(assigned | counters)}) {{ {' '.join(then)} }} "
                           f"else {{ {' '.join(other)} }}")
                # only variables assigned before the if are definitely assigned after it
            elif depth < 2:
                self.nloop += 1
                c = f"c{self.nloop}"
                body, _ = self.block(assigned, counters | {c}, depth + 1, self.r.randint(1, 4))
                n_iter = self.r.randint(0, 6)
                out.append(f"{c} = 0; while ({c} < {n_iter}) {{ {' '.join(body)} {c} = {c} + 1; }}")
                # loop may run zero times: body assignments are not definite
        return out, assigned


def program(seed: int) -> str:
    g = Gen(seed)
    stmts, assigned = g.block(set(), set(), 0, g.r.randint(3, 9))
    if assigned:
        stmts.append(f"print {sorted(assigned)[0]};")
    return "\n".join(stmts) + "\n"
