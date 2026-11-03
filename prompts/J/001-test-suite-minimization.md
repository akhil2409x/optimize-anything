# J · 001 · Test-suite minimization and prioritization on more-itertools

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-31
- **Kind of work:** formulation, code generation, measurement, mutation testing, documentation

## Original prompt

After Track D, Claude proposed test-suite minimization as Track J. Reply:

```
yes complete it
```

## AI response (summary)

Claude chose more-itertools v11.2.1 as the subject: a public open-source
library with ~7,300 lines of library code and ~9,200 lines of tests. It wrote:

- `setup.sh`, which clones the subject at that tag.
- `tsmin/collect.py`, which measures per-test time (junit XML, 3 runs,
  without coverage) and per-test statement and branch coverage (coverage.py
  dynamic contexts, read from its SQLite file). It keeps only requirements
  that every coverage run hit.
- `tsmin/select.py`, the formulation:
  - x_t ∈ {0,1}; minimise Σ c_t x_t subject to every requirement being covered.
  - Methods: greedy (count and per-second), redundancy removal, an exact ILP
    (HiGHS), an LP lower bound, and a random-order baseline.
  - Prioritization orders (total, additional, additional per second) scored
    with APFD and APFDc.
- `tsmin/mutate.py`, mutation testing:
  - Its own AST mutation operators, one node and one line per mutant.
  - Each mutant runs only against the tests that execute its line, on 2
    processes.
  - The result is a kill matrix, so any subset or ordering can be scored
    without rerunning.
- `patches/shrink-slow-tests.patch`, which shrinks the two bottleneck tests
  to the smallest inputs found that keep their coverage.
- `tests/test_tsmin.py` (15 tests) and `run_experiments.py` (6 experiments).

## Revisions during the session

1. **No statement coverage at first.** With branch coverage on, coverage.py
   stores only arcs, so the first collection found zero statement
   requirements. Statements are now derived from arc endpoints, as
   coverage.py itself does.
2. **Path mismatch in the patched copy.** Coverage of the patched copy was
   compared using paths relative to the wrong directory, so every
   requirement looked lost. The subject root is now passed through.
3. **ILP added free tests.** With raw junit times (0.000 s for many tests),
   the time-minimising ILP added hundreds of "free" tests. A 1 ms per-test
   overhead was added to the cost.
4. **First shrink went too far.** A first shrink to 10³ items lost the
   lock-contention branch of `concurrent_tee` and a Miller–Rabin loop arc;
   the sizes were raised to 10⁴ after probing.

## Resulting code change

Commit `[AI:claude] Track J: test-suite minimization (set cover: greedy vs ILP) on more-itertools` = local commit `020818a` (on GitHub: `https://github.com/akhil2409x/optimize-anything/commit/020818a`)

## Verification note

To do by the student:
- Write the set-cover ILP on paper with three tests and four requirements and
  solve it by hand.
- Open `results/kill_matrix.json` and confirm one missed mutant by applying it
  by hand and running the reduced suite.
- Rerun `./setup.sh && python run_experiments.py --remeasure` and compare the
  numbers.
