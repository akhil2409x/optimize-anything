"""Front end of the Tiny compiler: lexer, recursive-descent parser, AST, and a
reference AST interpreter (the ground truth every compiled program is checked
against).

Tiny language
    program  := stmt*
    stmt     := ID '=' expr ';'
              | 'print' expr ';'
              | 'if' '(' expr ')' block ['else' block]
              | 'while' '(' expr ')' block
              | block
    block    := '{' stmt* '}'
    expr     := sum [('<'|'<='|'>'|'>='|'=='|'!=') sum]
    sum      := term (('+'|'-') term)*
    term     := unary (('*'|'/'|'%') unary)*
    unary    := '-' unary | primary
    primary  := INT | ID | '(' expr ')'
    comments : '#' to end of line

Semantics: unbounded integers; comparisons give 0 or 1; any non-zero value
is true; `/` and `%` truncate toward zero like C (-7 / 2 = -3, -7 % 2 = -1);
dividing by zero or reading an unassigned variable is a runtime error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class TinyError(Exception):
    pass


class TinyRuntimeError(TinyError):
    pass


# ------------------------------------------------------------------ lexer
TOKEN_RE = re.compile(r"""
    (?P<ws>\s+|\#[^\n]*)
  | (?P<int>\d+)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op><=|>=|==|!=|[-+*/%<>=;(){}])
""", re.X)
KEYWORDS = {"print", "if", "else", "while"}


@dataclass
class Token:
    kind: str      # int, id, kw, op, eof
    text: str
    line: int


def lex(src: str) -> list[Token]:
    out, pos, line = [], 0, 1
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            raise TinyError(f"line {line}: unexpected character {src[pos]!r}")
        kind = m.lastgroup
        text = m.group()
        if kind != "ws":
            if kind == "id" and text in KEYWORDS:
                kind = "kw"
            out.append(Token(kind, text, line))
        line += text.count("\n")
        pos = m.end()
    out.append(Token("eof", "", line))
    return out


# ------------------------------------------------------------------ AST
@dataclass
class Num:
    value: int


@dataclass
class Var:
    name: str


@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class Neg:
    operand: object


@dataclass
class Assign:
    name: str
    expr: object


@dataclass
class Print:
    expr: object


@dataclass
class If:
    cond: object
    then: list
    other: list


@dataclass
class While:
    cond: object
    body: list


# ------------------------------------------------------------------ parser
class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self):
        return self.t[self.i]

    def take(self, text=None, kind=None):
        tok = self.t[self.i]
        if (text is not None and tok.text != text) or (kind is not None and tok.kind != kind):
            want = text or kind
            raise TinyError(f"line {tok.line}: expected {want!r}, got {tok.text or 'end of file'!r}")
        self.i += 1
        return tok

    def program(self):
        stmts = []
        while self.peek().kind != "eof":
            stmts.append(self.stmt())
        return stmts

    def block(self):
        self.take("{")
        stmts = []
        while self.peek().text != "}":
            if self.peek().kind == "eof":
                raise TinyError("unterminated block")
            stmts.append(self.stmt())
        self.take("}")
        return stmts

    def stmt(self):
        tok = self.peek()
        if tok.text == "print":
            self.take()
            e = self.expr()
            self.take(";")
            return Print(e)
        if tok.text == "if":
            self.take(); self.take("(")
            c = self.expr()
            self.take(")")
            then = self.block()
            other = []
            if self.peek().text == "else":
                self.take()
                other = self.block()
            return If(c, then, other)
        if tok.text == "while":
            self.take(); self.take("(")
            c = self.expr()
            self.take(")")
            return While(c, self.block())
        if tok.text == "{":
            return If(Num(1), self.block(), [])         # a bare block is just a scope-less group
        name = self.take(kind="id").text
        self.take("=")
        e = self.expr()
        self.take(";")
        return Assign(name, e)

    def expr(self):
        left = self.sum()
        if self.peek().text in ("<", "<=", ">", ">=", "==", "!="):
            op = self.take().text
            return BinOp(op, left, self.sum())
        return left

    def sum(self):
        left = self.term()
        while self.peek().text in ("+", "-"):
            op = self.take().text
            left = BinOp(op, left, self.term())
        return left

    def term(self):
        left = self.unary()
        while self.peek().text in ("*", "/", "%"):
            op = self.take().text
            left = BinOp(op, left, self.unary())
        return left

    def unary(self):
        if self.peek().text == "-":
            self.take()
            return Neg(self.unary())
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok.kind == "int":
            self.take()
            return Num(int(tok.text))
        if tok.kind == "id":
            self.take()
            return Var(tok.text)
        if tok.text == "(":
            self.take()
            e = self.expr()
            self.take(")")
            return e
        raise TinyError(f"line {tok.line}: unexpected {tok.text or 'end of file'!r}")


def parse(src: str):
    return Parser(lex(src)).program()


# ------------------------------------------------------------------ shared arithmetic
def c_div(a: int, b: int) -> int:
    if b == 0:
        raise TinyRuntimeError("division by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def c_mod(a: int, b: int) -> int:
    return a - b * c_div(a, b)


ARITH = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "/": c_div, "%": c_mod,
    "<": lambda a, b: int(a < b), "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b), ">=": lambda a, b: int(a >= b),
    "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
}


# ------------------------------------------------------------------ reference interpreter
def interpret(prog, max_steps: int = 10_000_000):
    env, out = {}, []
    steps = [0]

    def ev(e):
        if isinstance(e, Num):
            return e.value
        if isinstance(e, Var):
            if e.name not in env:
                raise TinyRuntimeError(f"variable {e.name} used before assignment")
            return env[e.name]
        if isinstance(e, Neg):
            return -ev(e.operand)
        return ARITH[e.op](ev(e.left), ev(e.right))

    def run(stmts):
        for s in stmts:
            steps[0] += 1
            if steps[0] > max_steps:
                raise TinyRuntimeError("step limit exceeded")
            if isinstance(s, Assign):
                env[s.name] = ev(s.expr)
            elif isinstance(s, Print):
                out.append(ev(s.expr))
            elif isinstance(s, If):
                run(s.then if ev(s.cond) != 0 else s.other)
            elif isinstance(s, While):
                while ev(s.cond) != 0:
                    run(s.body)
                    steps[0] += 1
                    if steps[0] > max_steps:
                        raise TinyRuntimeError("step limit exceeded")
    try:
        run(prog)
    except TinyRuntimeError as e:
        e.partial_output = out              # what was printed before the error
        raise
    return out
