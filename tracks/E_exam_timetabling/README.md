# Track E — Main case study: University exam timetabling

## 1. The process and why it needs optimizing

Every term a university must place each exam into a time slot. Two exams that
share even one student cannot go in the same slot (a *clash*), and students
who get exams in back-to-back slots have little time to prepare. A common
manual approach is to walk down the course list and drop each exam into the
first free slot. That produces a legal timetable most of the time, but it
ignores how close together each student's exams end up, and on tight calendars
it can fail outright.

Evidence (Experiment 1): the first-fit baseline is the cheapest method to run,
but it fails to find a clash-free timetable on some instances where better
orderings succeed, and its spacing cost is clearly higher than the optimized
methods. See `results/` for the numbers.

## 2. Formulation

Sets: exams *E*, students *S*, slots *T = {0, …, n_slots−1}*.
Data: *w_ij* = number of students taking both exam *i* and exam *j*.

Decision variables: *x_et ∈ {0,1}* = 1 if exam *e* is held in slot *t*.

Constraints (hard):
- each exam in exactly one slot: Σ_t x_et = 1
- no clashes: x_it + x_jt ≤ 1 for every conflicting pair (i, j) and every slot t

Objective — Carter proximity cost (the standard measure for this benchmark):

  min (1/|S|) · Σ_{i<j} w_ij · 2^(5−|t_i − t_j|)  for 1 ≤ |t_i − t_j| ≤ 5

Exams one slot apart cost 16 per shared student, five apart cost 1, six or more
apart cost nothing.

Secondary problem — minimum number of slots (graph colouring), solved as an ILP
in `timetabling/ilp.py`, gives the tightest calendar the data allows.

Success metric: feasibility (zero clashes) first, then lower proximity cost,
then runtime. All values are recomputed by `timetabling/evaluate.py`, which is
independent of the solvers.

## 3. Strategy selection

| Factor | This problem | Consequence |
|---|---|---|
| Variable type | binary (exam × slot) | discrete methods; gradient methods do not apply |
| Size | 50–400 exams synthetic; real Carter sets up to ~2,400 exams | exact ILP only for small cases, heuristics at scale |
| Exact or heuristic | NP-hard (contains graph colouring) | ILP + branch-and-bound for small instances and bounds; greedy + local search at scale |
| Hardware | one laptop CPU | single-threaded runs; restarts could be parallelised (Track D link) |
| Parallelism | independent seeds / restarts | embarrassingly parallel, not needed yet |

## 4. Methods compared

| Method | File | Chapter 6 link |
|---|---|---|
| First-fit by course id (**baseline**) | `greedy.py` | greedy |
| Largest-degree first | `greedy.py` | greedy |
| DSatur | `greedy.py` | greedy |
| DSatur + least-cost slot choice | `greedy.py` | greedy |
| DSatur-proximity + hill climbing | `local_search.py` | local improvement heuristic |
| Min-slots ILP | `ilp.py` | ILP, branch-and-bound (HiGHS) |
| Proximity ILP, cold and warm-started | `ilp.py` | ILP, branch-and-bound (HiGHS) |

## 5. How to run

```bash
pip install -r ../../requirements.txt
cd tracks/E_exam_timetabling
pytest -q                          # correctness checks
python run_experiments.py --quick  # ~30 s smoke run
python run_experiments.py          # full run, ~15 min
```

To add real benchmark data, download the Toronto/Carter `.stu` files from
https://www.cs.nott.ac.uk/~pszrq/data.htm into `data/` (for example
`hec-s-92.stu`, `sta-f-83.stu`, `yor-f-83.stu`). They are picked up
automatically with their standard slot counts.

## 6. Correctness checks (`tests/`)

- proximity cost and clash count checked against hand-computed values
- per-exam incremental cost agrees with the full objective
- every greedy method returns clash-free timetables on easy instances
- hill climbing never breaks feasibility and never makes the cost worse
- min-slots ILP finds 3 colours for a 5-cycle (known answer)
- proximity ILP equals brute-force enumeration on six tiny instances,
  including correctly reporting infeasible ones

## 7. Results

See `results/SUMMARY.md` (generated tables and plot) and the discussion in the report.

## 8. Known limitations

- Synthetic instances are denser than most real Carter instances; the real
  benchmark files should be added before final conclusions.
- The proximity ILP has a very weak LP relaxation (fractional x can make every z
  zero), so its proven lower bound stays at 0 and the optimality gap cannot be
  closed within the time limit. It is useful as an improver of a heuristic
  solution, not as a proof of optimality.
- Hill climbing stops at the first local minimum; Kempe-chain moves or
  simulated annealing are natural next steps.
