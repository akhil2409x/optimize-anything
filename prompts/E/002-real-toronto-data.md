# E · 002 · Real Toronto data, ILP-repair hybrid

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-11-01
- **Kind of work:** data acquisition, experiments, code (small), documentation

## Original prompt

After Track J, Claude listed the remaining student-side tasks (Carter data,
tool comparison, exports, publishing, report). Reply:

```
do it all for me
```

## AI response (summary)

- **Data:**
  - Found a public copy of the Toronto benchmark `.stu` files on GitHub and
    added `get_data.sh`, which fetches them at a pinned commit.
  - Checked that the instance sizes match the published ones.
- **Experiment changes:**
  - Experiment 1 now runs the real instances with the same 5 seeds as the
    synthetic ones.
  - SUMMARY.md gained a per-instance table (`--summary-only` rebuilds it
    from the CSVs).
- **New scripts:**
  - `hybrid_hec.py`: every greedy construction fails on hec-s-92, so the
    min-slots ILP supplies a clash-free timetable and hill climbing then
    lowers its cost.
  - `solve.py`, a one-instance demo.
- **Write-up:** updated the README with the real-data results, comparing
  with the best results listed on the Nottingham benchmark page.

## Revisions during the session

None needed to the solvers. The run surfaced a real result instead:
- First fit is clash-free on only 1 of 7 real institutions.
- hec-s-92 needs the ILP.

## Resulting code change

Commit `[AI:claude] Final deliverables: real Toronto data, tool comparison, 4-core rerun, report, demo` (the latest commit; see `git log`)

## Verification note

To do by the student:
- Run `./get_data.sh && python solve.py data/yor-f-83.stu` and compare
  with the table in the README.
- Explain why DSatur fails on hec-s-92 while the ILP succeeds.
