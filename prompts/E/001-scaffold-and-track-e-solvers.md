# E · 001 · Repository scaffold and Track E solvers

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-28
- **Kind of work:** planning, code generation, testing, debugging, documentation

## Original prompt

The assignment brief (PDF) was attached with the message "suggest". Claude
proposed a lean plan for all ten tracks and four Track E options. Follow-up
prompt, choosing the first option:

```
yes this   [replying to: "Exam timetabling (my top pick)"]
```

## AI response (summary)

Claude scaffolded the repository and wrote the Track E package:

- `timetabling/data.py` — Carter `.stu` loader, seeded synthetic generator
- `timetabling/evaluate.py` — clash check and Carter proximity cost
- `timetabling/greedy.py` — first-fit baseline, largest-degree, DSatur, DSatur-proximity
- `timetabling/local_search.py` — feasibility-preserving hill climbing
- `timetabling/ilp.py` — min-slots and min-proximity ILPs in PuLP + HiGHS
- `tests/test_timetabling.py` — 18 tests incl. brute-force comparison
- `run_experiments.py` — Experiments 1 and 2, CSVs and plot

## Revisions during the session

1. PuLP 4.2 API break → pinned `pulp==2.9.0` (failure log F3).
2. ILP objective disagreed with the evaluator → objective recomputed
   independently (F1).
3. Warm start ignored by PuLP's HiGHS wrapper → custom subclass passing a full
   x+z start (F2).
4. Tiny ILP test instance turned out infeasible; brute force confirmed it, so the
   test was extended to check infeasible cases too.
5. Small Experiment 2 instances were infeasible for every method → slot budget
   set to (optimal slot count from the min-slots ILP) + 3.

## Resulting code change

Commit: `[AI:claude] Track E: exam timetabling solvers, tests, experiments` = local commit `7b97d6d` (on GitHub: `<repo URL>/commit/7b97d6d`)

## Verification note

To do by the student before submitting: read every file, rerun `pytest -q` and
`python run_experiments.py`, hand-check one proximity cost, and be ready to
explain DSatur, the ILP linearisation and why the LP bound is zero in the viva.
Add a chat export or screenshots of this conversation to `prompts/E/exports/`.
