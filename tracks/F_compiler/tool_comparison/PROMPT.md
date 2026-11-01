Paste everything below the line, unchanged, into a NEW chat in each AI tool.
Save each tool's code as `tool_comparison/submission_<tool>.py`
(for example `submission_chatgpt.py`, `submission_claude.py`) and run
`python tool_comparison/check_submission.py tool_comparison/submission_<tool>.py`.

---

I am writing a small optimizing compiler in Python. Please implement one
optimization pass: **local common sub-expression elimination (CSE) by value
numbering**, working inside each basic block separately.

The IR (you may NOT change these classes; they are already defined in my code):

```python
from dataclasses import dataclass, field

@dataclass
class Instr:
    op: str                 # see below
    dst: str | None = None  # variable written (None for print and terminators)
    a: object = None        # operand: variable name (str) or integer constant (int)
    b: object = None        # second operand (binary ops only)
    targets: tuple = ()     # jump targets (terminators only)

@dataclass
class Block:
    label: str
    instrs: list            # non-terminator instructions, in order
    term: Instr             # jmp / cbr / halt

@dataclass
class Function:
    blocks: dict            # label -> Block
```

Instruction kinds in `Block.instrs`:
- `const dst, a` (dst = integer a), `copy dst, a` (dst = a), `neg dst, a` (dst = -a)
- binary `dst = a OP b` with OP in: add sub mul div mod shl lt le gt ge eq ne
  (div/mod truncate toward zero like C and raise an error when b == 0)
- `print a` (dst is None)

Variables are NOT in SSA form: the same variable can be assigned many times.

Write a function

```python
def cse(fn: Function) -> bool:
    """Eliminate repeated computations inside each block. Replace a repeated
    computation with `copy dst, <earlier result>`. Mutate fn in place and return
    True if anything changed."""
```

Requirements: the program's printed output and its runtime errors must not
change in any case. Treat add, mul, eq and ne as commutative. Only use the
Python standard library. Reply with just the code.
