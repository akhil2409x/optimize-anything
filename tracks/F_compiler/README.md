# Track F — Compiler optimization: the Tiny compiler

A complete mini-compiler written for this track. Source code goes through a
lexer and a recursive-descent parser to an AST, then to three-address IR on a
control-flow graph, then six optimization passes, register allocation, and
code for a small register VM that counts instructions and cycles. A separate
AST interpreter is the ground truth for every correctness check.

```
tinyc/frontend.py  lexer, parser, AST, reference interpreter
tinyc/ir.py        IR + CFG, AST -> IR lowering, IR interpreter
tinyc/passes.py    6 optimization passes, liveness, dominators, natural loops
tinyc/backend.py   interference graph, greedy + ILP register allocation, codegen, VM
tinyc/driver.py    O0-O3 pipelines;  tinyc/fuzz.py: random programs
programs/*.tiny    8 benchmark programs
```

**The Tiny language:** integers, variables, `+ - * / %`, comparisons,
`if`/`else`, `while` and `print`. Division truncates toward zero like C.
Division by zero and reading an unassigned variable are runtime errors, and
optimizations must preserve them.

## Optimizations (six; four needed for full credit)

| Pass | What it does | Correctness rule it must respect |
|---|---|---|
| Constant propagation + folding | Global dataflow over the CFG; folds expressions and branches on constants | Never folds `x/0`. "Unassigned on some path" is its own lattice value, never a constant. |
| Copy propagation + coalescing | Replaces uses of copies; merges `%t = op …; v = copy %t` | Invalidates on redefinition |
| CSE (local value numbering) | Reuses identical computations in a block; add/mul/eq/ne are commutative | Invalidates when an operand or the holder is redefined |
| Strength reduction | `x*2^k → x shl k`, `x*1`, `x*0`, `x+0`, `x-x`, `x/1`, `x%1`, `x*-1 → neg` | **No** `x/2^k → shr`: −7/4 = −1, but −7 >> 2 = −2 |
| Dead-code elimination | Liveness-based; also removes unreachable blocks and bypasses empty ones | Never removes `print` or a division that might trap |
| Loop-invariant code motion | Hoists invariant pure computations into a loop preheader; inner loops first | Never hoists a possibly-trapping division, or a value read before its definition or after the loop (zero-trip loops) |

## Formulation: register allocation as an optimization problem

**The problem:** each variable must go in one of K registers or be spilled to
memory; variables live at the same time (an edge in the interference graph)
can't share a register. Minimise the spill cost, the sum over spilled
variables of their uses and definitions weighted by 10^(loop depth).

- **Greedy:** Chaitin-Briggs simplify/select with optimistic colouring.
  Polynomial time, no optimality guarantee.
- **ILP:** binary x_{v,r} (v in register r) and spill_v. Each variable gets one
  register or is spilled; neighbours can't share a register. Minimise
  Σ cost_v · spill_v, solved exactly by HiGHS branch-and-bound.

Pass ordering is also explored as a search problem: all 720 orders of the six
passes vs a greedy ordering heuristic (Exp 4).

## Correctness

| Check | Result |
|---|---|
| 8 benchmarks × 11 configurations (O0–O3, each pass alone, reversed order) × 2 register counts | Output identical to the AST interpreter |
| 300 random programs × 11 configurations, 2–6 registers (`tests/test_tinyc.py`) | 0 mismatches; output *and* runtime errors identical |
| Unit tests per pass, including each known hazard | 32/32 tests pass |
| Register allocations independently checked (no two interfering variables share a register) | Every compile |
| **Mutation testing** (`tests/mutation_check.py`): 5 injected optimizer bugs | **Each caught by a targeted test.** A 6th, equivalent mutant (can't change any Tiny program's behaviour) is documented. |

An earlier fuzzer reported 0 mismatches while being blind to two of these
bugs. See failure log F10 for how that was found and fixed.

## Results

Cycle model: mul 3, div/mod 20, ld/st 3, everything else 1. K = 6
allocatable registers unless stated. Full tables are in `results/SUMMARY.md`.

**Exp 1: optimization levels (geometric mean over 8 programs, vs O0)**

| Level | What it adds | Cycle speed-up | Instructions executed | Code size | Loads + stores (total) | Compile time |
|---|---|---|---|---|---|---|
| O0 | everything in memory | 1.00× | 1.00× | 1.00× | 565,146 | 0.7 ms |
| O1 | + greedy register allocation | 3.44× | 3.06× fewer | 2.76× smaller | 7,334 | 1.0 ms |
| O2 | + all six IR passes | **4.89×** | 4.95× fewer | 4.70× smaller | 2,108 | 2.4 ms |
| O3 | + ILP register allocation | 4.91× | 5.00× fewer | 4.73× smaller | 1,808 | 14.2 ms |

- **Range:** per program, speed-ups run from 2.1× (pressure) to 9.6× (fib,
  poly).
- **Register allocation does most of the work:** it removes 99% of memory
  traffic.
- **The IR passes add another 1.42×** on top of register allocation.

**Exp 2: ablation (speed-up vs no IR passes, greedy allocation)**

| Configuration | Geomean speed-up |
|---|---|
| All six passes | **1.42×** |
| All except constant propagation | 1.09× (the most important pass) |
| All except DCE | 1.28× |
| All except LICM | 1.31× |
| All except copy propagation, or except CSE | 1.34× |
| All except strength reduction | 1.37× |
| Any single pass alone | 0.97–1.03× |

**Passes work together.** Each pass alone does almost nothing, but together
they give 1.42×:
- constant propagation turns loads into immediates, so CSE sees identical keys;
- copy propagation exposes invariants to LICM;
- DCE then deletes the leftovers.

**LICM alone is a real trade-off,** not a bug. It makes grid 21% faster but
five other programs 4–16% slower. Hoisting constants out of loops keeps them
in registers across the whole loop, which causes spills with 6 registers.
With constant propagation first, the constants become immediates and the
problem disappears.

**Exp 3: register allocation, greedy vs ILP (8 programs, after all IR passes)**

| K | Greedy spill cost | ILP spill cost | Programs where ILP is better | Cycles, greedy → ILP | Allocation time, greedy vs ILP |
|---|---|---|---|---|---|
| 2 | 5,224 | 5,114 | 2 / 8 | 882,571 → 876,031 (−0.7%) | 1.7 ms vs 63 ms |
| 4 | 892 | 872 | 1 / 8 | 699,130 → 698,080 (−0.2%) | 2.0 ms vs 75 ms |
| 6 | 147 | 127 | 1 / 8 | 674,368 → 673,318 (−0.2%) | 2.1 ms vs 91 ms |
| 8 | 83 | 63 | 1 / 8 | 671,656 → 670,606 (−0.2%) | 2.0 ms vs 106 ms |

- **The ILP is always optimal and never worse:** it never needed more cycles
  than greedy in any case.
- **Greedy is near-optimal on these programs:** within 0.7%, at about 1/40 of
  the allocation time. That matches why production compilers use heuristic
  colouring.

**Exp 4: pass ordering (each pass applied once; 720 orders per program)**
- **The default order** (constprop, copyprop, cse, strength, licm, dce) was
  as good as the best order on 7 of 8 programs.
- **Order matters most for grid:** 46,206 cycles for the best order vs 80,853
  for the worst, a 1.75× difference with 22 distinct outcomes.
- **The greedy ordering heuristic fails on grid** (77,406 cycles). It applies
  whichever pass helps most *right now*, and LICM's payoff only shows after
  constant propagation has run.
- **Iterating to a fixpoint beats every single order:** poly drops to 3,106
  cycles vs 3,706 for the best single pass of each, and grid to 46,166. So
  the default pipeline iterates until nothing changes.

## AI tool comparison (Section 8)

`tool_comparison/` holds a self-contained prompt (implement the CSE pass) and
`check_submission.py`, which scores any tool's answer on identical tests:
10 hazard unit tests, 300 random programs in 4 pipeline configurations, and
the instructions executed on the 8 benchmarks. Paste the prompt into fresh
chats with two tools and score both; `docs/tool_comparison.md` has the table
to fill in. The scorer was itself checked: the in-repo pass scores 10/10 with
0 wrong programs, and a deliberately naive version scores 5/10 and fails a
random program.

## How to run

```bash
cd tracks/F_compiler
pytest -q                           # 32 tests (~15 s)
python tests/mutation_check.py      # inject 6 bugs, check each is caught (~2 min)
python run_experiments.py           # all 4 experiments (~6 min)
python -c "from tinyc import driver as D, backend as B; c = D.compile_source(open('programs/grid.tiny').read(), 'O2'); print(c.fn); print(B.run_vm(c.code))"
```

## Limitations

- **The language is small:** no functions, arrays or input, so every program's
  inputs are compile-time constants. That flatters constant propagation.
  Loops keep the measurements meaningful, because loop variables are never
  constant.
- **CSE is local** (one basic block); global value numbering would find more.
- **No live-range splitting:** register allocation spills a whole variable.
  Spill code uses two reserved scratch registers.
- **A simple cycle model:** no pipeline, cache or branch-prediction effects.
  VM wall-clock times are reported too, but they are Python interpreter times.
