# F · 001 · Tiny compiler: six passes, register allocation, pass ordering

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-29
- **Kind of work:** planning, code generation, testing (fuzzing, mutation testing), benchmarking, documentation

## Original prompt

After Track H, Claude said F was next and included the two-tool comparison.
Reply:

```
yes start F
```

## AI response (summary)

Claude wrote the Tiny compiler in `tinyc/`:

- `frontend.py`: lexer, recursive-descent parser, AST and the reference AST
  interpreter (ground truth).
- `ir.py`: three-address code on a CFG, AST → IR lowering and an IR interpreter.
- `passes.py`: global constant propagation and folding, copy propagation with
  coalescing, local CSE, strength reduction, DCE, and LICM with preheaders.
- `backend.py`: interference graph, greedy (Chaitin-Briggs) and ILP register
  allocation, code generation, and a register VM that counts cycles.
- `fuzz.py`: random terminating programs. `driver.py`: levels O0–O3.
- `tests/test_tinyc.py` (32 tests) and `tests/mutation_check.py`.
- `tool_comparison/PROMPT.md` and `check_submission.py`, for the Section 8
  tool comparison.

## Revisions during the session

1. **Errors hid printed output.** The correctness check compared only the
   error when a program printed and then failed. It now compares output and
   error together.
2. **The fuzzer was blind to two injected bugs** (failure log F10). The
   generator was improved, single-pass and reversed-order configurations were
   added, and targeted tests now kill each mutant.
3. **A test was vacuous.** The LICM test's assertion couldn't fail; it was
   rewritten to check that `i*i` leaves the inner loop.
4. **The DCE ablation was impossible** (failure log F11). The final clean-up
   is now optional, and the experiment was rerun.
5. **"Only LICM" looked like no effect.** Investigation showed a genuine
   register-pressure trade-off (+21% on grid, −4–8% on five other programs),
   which is reported as a finding.

## Resulting code change

Commit `[AI:claude] Track F: Tiny compiler, 6 passes, greedy vs ILP register allocation`
= local commit `9c9a64e` (on GitHub: `https://github.com/akhil2409x/optimize-anything/commit/9c9a64e`)

## Verification note

To do by the student:
- Run `pytest -q` and `python tests/mutation_check.py`.
- Compile `programs/grid.tiny` at O0 and O2 and read the IR before and after
  (`print(fn)` in `tinyc/driver.py`).
- Explain why `x/4 → x>>2` is wrong for x = −7, and why LICM can make code
  slower with few registers.
- **Do the two-tool comparison:** paste `tool_comparison/PROMPT.md` into fresh
  chats with two tools and score both answers with `check_submission.py`.
