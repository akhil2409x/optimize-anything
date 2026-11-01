"""Intermediate representation: three-address code in a control-flow graph.

Operands are either a variable name (str: a source variable or a temporary
`%tN`) or an integer constant. Instructions (non-terminators):
    const  dst, imm           dst = imm
    copy   dst, a             dst = a
    BINOP  dst, a, b          BINOP in add sub mul div mod shl lt le gt ge eq ne
    neg    dst, a
    print  a
Terminators (exactly one, last in each block):
    jmp L | cbr a, Ltrue, Lfalse  (a != 0 -> Ltrue) | halt

Side effects that passes must respect: `print` (output) and `div`/`mod`
(they can raise division by zero, so they may not be deleted or moved unless
the divisor is a known non-zero constant).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import frontend as F

BINOPS = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod", "<": "lt", "<=": "le",
          ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
OPFUNC = {v: F.ARITH[k] for k, v in BINOPS.items()}
OPFUNC["shl"] = lambda a, b: a << b
PURE_BINOPS = set(OPFUNC) - {"div", "mod"}
COMMUTATIVE = {"add", "mul", "eq", "ne"}


@dataclass
class Instr:
    op: str
    dst: str | None = None
    a: object = None
    b: object = None
    targets: tuple = ()           # jump targets for jmp/cbr

    def uses(self):
        return [x for x in (self.a, self.b) if isinstance(x, str)]

    def defs(self):
        return [self.dst] if self.dst is not None else []

    def may_trap(self):
        return self.op in ("div", "mod") and not (isinstance(self.b, int) and self.b != 0)

    def has_side_effect(self):
        return self.op == "print" or self.may_trap()

    def __str__(self):
        if self.op == "jmp":
            return f"jmp {self.targets[0]}"
        if self.op == "cbr":
            return f"cbr {self.a}, {self.targets[0]}, {self.targets[1]}"
        if self.op == "halt":
            return "halt"
        if self.op == "print":
            return f"print {self.a}"
        if self.op in ("const", "copy", "neg"):
            return f"{self.dst} = {self.op} {self.a}"
        return f"{self.dst} = {self.op} {self.a}, {self.b}"


@dataclass
class Block:
    label: str
    instrs: list = field(default_factory=list)
    term: Instr | None = None

    def succs(self):
        return list(self.term.targets) if self.term else []


@dataclass
class Function:
    blocks: dict                  # label -> Block, insertion ordered; first is entry
    entry: str = "entry"

    def preds(self):
        p = {l: [] for l in self.blocks}
        for b in self.blocks.values():
            for s in b.succs():
                p[s].append(b.label)
        return p

    def reachable(self):
        seen, stack = set(), [self.entry]
        while stack:
            l = stack.pop()
            if l in seen:
                continue
            seen.add(l)
            stack.extend(self.blocks[l].succs())
        return seen

    def instr_count(self):
        return sum(len(b.instrs) + 1 for b in self.blocks.values())

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __str__(self):
        lines = []
        for b in self.blocks.values():
            lines.append(f"{b.label}:")
            lines += [f"    {i}" for i in b.instrs]
            lines.append(f"    {b.term}")
        return "\n".join(lines)


# ------------------------------------------------------------------ AST -> IR
class IRGen:
    def __init__(self):
        self.blocks = {}
        self.ntemp = 0
        self.nlabel = 0
        self.cur = self.new_block("entry")

    def new_block(self, label=None):
        if label is None:
            self.nlabel += 1
            label = f"L{self.nlabel}"
        b = Block(label)
        self.blocks[label] = b
        return b

    def temp(self):
        self.ntemp += 1
        return f"%t{self.ntemp}"

    def emit(self, ins):
        self.cur.instrs.append(ins)

    def end(self, term):
        self.cur.term = term

    def expr(self, e):
        if isinstance(e, F.Num):
            t = self.temp()
            self.emit(Instr("const", t, e.value))
            return t
        if isinstance(e, F.Var):
            return e.name
        if isinstance(e, F.Neg):
            a = self.expr(e.operand)
            t = self.temp()
            self.emit(Instr("neg", t, a))
            return t
        a = self.expr(e.left)
        b = self.expr(e.right)
        t = self.temp()
        self.emit(Instr(BINOPS[e.op], t, a, b))
        return t

    def stmts(self, ss):
        for s in ss:
            self.stmt(s)

    def stmt(self, s):
        if isinstance(s, F.Assign):
            v = self.expr(s.expr)
            self.emit(Instr("copy", s.name, v))
        elif isinstance(s, F.Print):
            self.emit(Instr("print", a=self.expr(s.expr)))
        elif isinstance(s, F.If):
            c = self.expr(s.cond)
            then_b, else_b, join = self.new_block(), self.new_block(), self.new_block()
            self.end(Instr("cbr", a=c, targets=(then_b.label, else_b.label)))
            self.cur = then_b; self.stmts(s.then); self.end(Instr("jmp", targets=(join.label,)))
            self.cur = else_b; self.stmts(s.other); self.end(Instr("jmp", targets=(join.label,)))
            self.cur = join
        elif isinstance(s, F.While):
            head, body, exit_b = self.new_block(), self.new_block(), self.new_block()
            self.end(Instr("jmp", targets=(head.label,)))
            self.cur = head
            c = self.expr(s.cond)
            self.end(Instr("cbr", a=c, targets=(body.label, exit_b.label)))
            self.cur = body; self.stmts(s.body); self.end(Instr("jmp", targets=(head.label,)))
            self.cur = exit_b


def lower(prog) -> Function:
    g = IRGen()
    g.stmts(prog)
    g.end(Instr("halt"))
    return Function(g.blocks)


# ------------------------------------------------------------------ IR interpreter
def run_ir(fn: Function, max_steps: int = 10_000_000):
    """Execute IR directly (used by tests to check each pass in isolation)."""
    env, out, steps = {}, [], 0

    def val(x):
        if isinstance(x, int):
            return x
        if x not in env:
            raise F.TinyRuntimeError(f"variable {x} used before assignment")
        return env[x]
    label = fn.entry
    while True:
        b = fn.blocks[label]
        for i in b.instrs:
            steps += 1
            if i.op == "const":
                env[i.dst] = i.a
            elif i.op == "copy":
                env[i.dst] = val(i.a)
            elif i.op == "neg":
                env[i.dst] = -val(i.a)
            elif i.op == "print":
                out.append(val(i.a))
            else:
                env[i.dst] = OPFUNC[i.op](val(i.a), val(i.b))
        steps += 1
        if steps > max_steps:
            raise F.TinyRuntimeError("step limit exceeded")
        t = b.term
        if t.op == "halt":
            return out
        if t.op == "jmp":
            label = t.targets[0]
        else:
            label = t.targets[0] if val(t.a) != 0 else t.targets[1]
