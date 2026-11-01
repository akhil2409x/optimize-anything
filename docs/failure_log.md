# Failure log — AI-generated errors found and fixed

The brief requires at least three documented cases. Each entry: what the AI
produced, how the error was caught, the fix, and the lesson.

---

## F1 · Track E · ILP reported a cost four times higher than the real one

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the first version of `min_proximity` in `timetabling/ilp.py`
  returned the solver's own objective value. On a 30-exam instance with a
  60 s time limit it reported **200.59**, while the independent evaluator
  (`evaluate.proximity_cost`) gave **47.59** for the same timetable.
- **Cause:** the auxiliary variables `z[i,j,d]` are only bounded from below.
  At the optimum they sit on that bound, but a time-limited incumbent can leave
  them higher than needed, so the solver's objective over-states the true cost.
- **How it was caught:** comparing solver output with the evaluator on every
  run instead of trusting either one.
- **Fix:** `ILPResult.objective` is now always recomputed from the timetable by
  `evaluate.py`; the comment in the code explains why.
- **Lesson:** never report a solver's objective for a non-optimal run without
  re-evaluating the decision variables.

## F2 · Track E · Warm start silently ignored (two layers)

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the generated code called `setInitialValue` on the
  variables but never told the solver to use them. After adding
  `warmStart=True`, results with and without the warm start were still
  identical — on one instance the "warm-started" ILP (23.28) was even worse than
  the heuristic it started from (22.36), which is impossible if the start was used.
- **Cause:** PuLP 2.9's highspy-based `HiGHS` class accepts `warmStart` but
  only the command-line `HiGHS_CMD` actually passes it on. Also, a start
  that sets only the x variables violates the z constraints and is rejected.
- **How it was caught:** a sanity rule — a warm-started minimizer can never end
  worse than its start.
- **Fix:** `_HiGHSWarm` in `ilp.py` hands a complete solution (x **and** z) to
  highspy with `setSolution` before `run()`. The same instance now reaches
  20.05, better than the heuristic's 22.36.
- **Lesson:** check that an option has an observable effect; "accepted"
  parameters are not always used.

## F3 · Environment · Generated code broke on the installed library version

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the ILP code used `pulp.LpVariable(name, cat="Binary")`,
  which fails with `TypeError` on PuLP 4.2 (the latest release at the time),
  whose variable API was rewritten.
- **Fix:** pinned `pulp==2.9.0` in `requirements.txt`, the version the code
  and tests were written against.
- **Lesson:** AI tools write code for the library versions they were trained
  on. Pin versions and record them.

---

*(Add entries from Tracks A–J below as they happen — xv6 and SQL rewrites usually supply several.)*
