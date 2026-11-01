# Track E — Main case study: University exam timetabling

## 1. The process and why it needs optimizing

Every term a university must place each exam into a time slot. Two exams that
share even one student cannot go in the same slot (a *clash*), and students
who get exams in back-to-back slots have little time to prepare. A common
manual approach is to walk down the course list and drop each exam into the
first free slot. That produces a legal timetable most of the time, but it
ignores how close together each student's exams end up, and on tight calendars
it can fail outright.

Evidence (Experiment 1):
- **Synthetic data:** the first-fit baseline found no clash-free timetable
  in 7 of 20 runs. When it succeeded, its spacing cost was 22–47% higher
  than DSatur-proximity + hill climbing.
- **Real data:** on the seven real Toronto institutions it produced a
  clash-free timetable for **only one** (sta-f-83), and that one cost 25%
  more than the optimized method.

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
| Size | 50–400 exams synthetic; real Toronto sets used here 81–381 exams (the largest Toronto set has ~2,400) | exact ILP only for small cases, heuristics at scale |
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

**Real benchmark data.** `./get_data.sh` fetches the Toronto (Carter,
Laporte and Lee 1996) `.stu` files into `data/`, from a pinned commit of a
public copy. The original description is at
<https://people.cs.nott.ac.uk/pszrq/data.htm>. `run_experiments.py` picks
up every `data/*.stu` with its standard slot count. The files are not
committed: `.gitignore` excludes them.

The seven instances used:

| Instance | Exams | Students | Slots |
|---|---|---|---|
| ear-f-83 | 190 | 1,125 | 24 |
| hec-s-92 | 81 | 2,823 | 18 |
| lse-f-91 | 381 | 2,726 | 18 |
| sta-f-83 | 139 | 611 | 13 |
| tre-s-92 | 261 | 4,360 | 23 |
| ute-s-92 | 184 | 2,749 | 10 |
| yor-f-83 | 181 | 941 | 21 |

The sizes match the published ones, which checks the loader.

`python hybrid_hec.py` runs the min-slots ILP, then hill climbing, on
hec-s-92 (see §7).

## 6. Correctness checks (`tests/`)

- proximity cost and clash count checked against hand-computed values
- per-exam incremental cost agrees with the full objective
- every greedy method returns clash-free timetables on easy instances
- hill climbing never breaks feasibility and never makes the cost worse
- min-slots ILP finds 3 colours for a 5-cycle (known answer)
- proximity ILP equals brute-force enumeration on six tiny instances,
  including correctly reporting infeasible ones

## 7. Results

Full tables are in `results/SUMMARY.md` (5 seeds per instance). Cost is the
proximity cost per student, recomputed by `evaluate.py`.

**Synthetic instances:**

| Exams | First fit (baseline) | DSatur | DSatur-proximity + hill climb |
|---|---|---|---|
| 50 | 48.95, feasible 3/5 | 53.47 | **33.56 ± 0.83** |
| 100 | 31.96, 4/5 | 33.90 | **25.32 ± 0.77** |
| 200 | 31.24, 1/5 | 35.15 | **25.65 ± 0.41** |
| 400 | 28.87, 5/5 | 32.91 | **19.66 ± 0.25** |

**Real Toronto instances:**

| Instance | First fit (baseline) | Largest degree | DSatur | DSatur-proximity + hill climb | Best known (Nottingham page) |
|---|---|---|---|---|---|
| ear-f-83 | infeasible | infeasible | 67.38 | **40.83 ± 0.24** | — |
| hec-s-92 | infeasible | infeasible | infeasible | infeasible; ILP + hill climb **15.79 ± 1.04** | — |
| lse-f-91 | infeasible | infeasible | infeasible | **13.00 ± 0.04** | — |
| sta-f-83 | 211.93 | 194.40 | 194.40 | **168.95 ± 0.29** | — |
| tre-s-92 | infeasible | 15.90 | 15.92 | **10.09** | 7.75 |
| ute-s-92 | infeasible | infeasible | 58.82 | **31.70 ± 0.31** | — |
| yor-f-83 | infeasible | infeasible | 61.66 | **46.30 ± 0.06** | 34.84 (version not stated) |

- **Feasibility is the hard part on real data.** First fit is clash-free on
  1 of 7 instances, plain DSatur on 5 of 7, and DSatur-proximity on 6 of 7.
  Choosing the cheapest legal slot also lets DSatur finish lse-f-91.
- **hec-s-92 defeats every greedy method** in 18 slots. The min-slots ILP
  finds an 18-slot clash-free timetable in 20 s (cost 25.25), and hill
  climbing then lowers that to 15.79 ± 1.04 (best 14.27). This hybrid, exact
  for feasibility and heuristic for cost, is a direct Chapter 6 lesson.
- **Against published results:** the Nottingham benchmark page lists best
  penalties of 7.75 for tre-s-92 and 34.84 for yor-f-83. Ours are 30–33%
  higher. State-of-the-art methods spend minutes to hours per instance with
  much stronger neighbourhoods; ours takes under 0.2 s.
- **Small instances (12–18 exams):** the proximity ILP improved the
  heuristic by 4–31% within 30 s, but its lower bound stayed at 0, so
  optimality was never proven. Once (synth-e18 r0) the cold-started ILP
  ended worse than the heuristic. Its warm-started run did not.

## 8. Known limitations

- **Single-exam moves only:** hill climbing uses one-exam moves. That is
  why it stays 30%+ above published Toronto results. Kempe-chain moves or
  simulated annealing are the next step.
- **hec-s-92 needs the ILP:** none of the greedy constructions is
  clash-free on it within its 18 slots.
- The proximity ILP has a very weak LP relaxation (fractional x can make every z
  zero), so its proven lower bound stays at 0 and the optimality gap cannot be
  closed within the time limit. It is useful as an improver of a heuristic
  solution, not as a proof of optimality.
- Hill climbing stops at the first local minimum.
