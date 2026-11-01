# Optimize Anything — Assignment 8 (14-Day Vibe-Coding Challenge)

Optimization Techniques lab · all ten tracks A–J · AI-assisted development with full disclosure.

**Main case study (Track E): university exam timetabling** — assign exams to time
slots with no student clashes while spreading each student's exams apart.

## Collaborators

| GitHub user | Role |
|---|---|
| [@akhil2409x](https://github.com/akhil2409x) | student |
| [@akhilkulal4050](https://github.com/akhilkulal4050) | professor (reviewer) |

## Repository layout

```
tracks/
  A_continuous/                      GD … Adam, Newton, Gauss–Newton, backprop, landscapes
  B_discrete/                        greedy, DP, branch-and-bound, ILP, QUBO/Ising
  C_quantum_optimization/            QAOA, VQE, parameter-shift, SPSA, QNG, noise, barren plateaus
  D_submodular_parallel_distributed/ submodular greedy, speedup/Amdahl, FedAvg, MGDA, GradNorm
  E_exam_timetabling/                main case study (working code, tests, results)
  F_compiler/                        mini-compiler + optimization passes
  G_xv6/                             xv6 kernel patch + benchmarks
  H_dbms/                            PostgreSQL TPC-H index / join-order selection
  I_quantum_algorithms_qml/          DJ, BV, Grover, QFT, QPE, Shor, annealing, QML vs classical
  J_software_engineering/            test-suite minimization on an open-source project
prompts/A … prompts/J/               prompt logs, one folder per track (exports/ inside each)
prompts/_session/                    verbatim export of the stored part of the AI session
docs/failure_log.md                  AI-generated errors found and fixed (F1–F16)
docs/tool_comparison.md              two AI tools compared on one task
report/REPORT.md, report/REPORT.pdf  technical report (one section per track, final questions answered)
demo/                                demonstration deck (source) and live-demo commands
tools/export_transcript.py           converts an AI session transcript to a redacted Markdown export
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd tracks/E_exam_timetabling && pytest -q && python run_experiments.py --quick
```

External code and data are fetched by scripts rather than committed:

| Track | Script | What it fetches |
|---|---|---|
| E | `tracks/E_exam_timetabling/get_data.sh` | Toronto benchmark files |
| G | `tracks/G_xv6/setup.sh` | xv6-riscv at a pinned commit; the patch is applied on top |
| H | `tracks/H_dbms/setup.sh` | TPC-H dbgen (also needs PostgreSQL 16 + hypopg) |
| J | `tracks/J_software_engineering/setup.sh` | more-itertools v11.2.1 |

## Status

| Track | Topic | Status |
|---|---|---|
| A | Continuous optimization | 13 optimizers from scratch, hand-written backprop, curvature methods, 7 experiments; 19 tests vs autograd |
| B | Discrete optimization | knapsack ×5 methods, QUBO/Ising, feature selection, 31 tests, results |
| C | Quantum optimization | QAOA on Track B Max-Cut, VQE (H₂), GD/Adam/SPSA/QNG, shots, noise, barren plateaus; 13 tests (simulation only) |
| D | Submodular / parallel / distributed | submodularity tests, greedy vs ILP, speed-up/Amdahl/Gustafson (2-core cloud + 4-core laptop VM), load balancing + work stealing, FedAvg, MGDA/GradNorm; 22 tests |
| E | Exam timetabling (main case study) | baseline + 5 heuristics + 2 ILPs + ILP-repair hybrid; synthetic and **real Toronto data (7 institutions)**; 18 tests |
| F | Compiler optimization | Tiny compiler: 6 passes, greedy vs ILP register allocation, pass ordering; 32 tests + mutation testing; tool-comparison kit |
| G | xv6 OS optimization | MLFQ scheduler + copy-on-write fork; usertests pass on 4 kernels × 1/3 CPUs; patch + benchmarks |
| H | DBMS optimization | TPC-H SF 0.1 index selection: knapsack DP, greedy, what-if greedy, ILP vs default; 9 tests; results identical |
| I | Quantum algorithms and QML | DJ, BV, Grover, QFT, QPE, Shor (hand-written), annealing + embedding, QML vs classical; 33 tests (simulation only) |
| J | Software-engineering optimization | more-itertools test-suite minimization as weighted set cover (greedy vs ILP), mutation testing (1,116 mutants), bottleneck fix, prioritization; 15 tests |

## AI tools register

Every AI tool used anywhere in this repository is listed here (Section 8 of the brief).

| Tool | Model / version | Tracks | Dates | Kind of work |
|---|---|---|---|---|
| Claude (claude.ai) | Claude Opus 5.5 | A–J, repo scaffold, report, demo deck | 2026-10-28 – 11-01 | planning, code generation, testing, debugging, documentation |
| Claude (fresh-context subagent) | Claude Sonnet 5.5 | F (tool comparison only) | 2026-11-01 | code generation for the CSE comparison task |
| Claude (fresh-context subagent) | Claude Haiku 5.5 | F (tool comparison only) | 2026-11-01 | code generation for the CSE comparison task |

## Publishing (student)

The brief requires a public GitHub repository. From this folder, with the
GitHub CLI logged in (`gh auth login`):

```bash
gh repo create optimize-anything --public --source=. --push
gh api -X PUT repos/{owner}/optimize-anything/collaborators/akhil-gitcoder -f permission=push
gh api -X PUT repos/{owner}/optimize-anything/collaborators/akhilkulal4050 -f permission=pull
```

`gh api` fills in `{owner}` from the current repository. Each
`prompts/*/00*.md` names its commit hash. After pushing, replace
`<repo URL>` there with the repository's address, so the links resolve.

## Commit convention

- `[AI:claude] …` — change generated by an AI tool (name the tool)
- `[manual] …` — change written or corrected by hand
- `[AI:claude+manual] …` — AI draft with hand edits; say what was edited in the body

Every AI commit should have a matching entry in `prompts/<track>/`.

## Chapter 6 coverage checklist

Fill in the code location and result as each item is completed. The list must
match the exact Chapter 6 syllabus taught in class — add or remove rows to match.

| Area | Method / concept | Track | Code location | Result |
|---|---|---|---|---|
| Foundations | ML as optimization (params, loss, regularization) | A | `tracks/A_continuous/contopt/objectives.py` (`LogisticRegression`) | A Exp. 3 (λ sweep) |
| Foundations | Continuous vs discrete optimization | A, B | `tracks/A_continuous/`, `tracks/B_discrete/` | A all, B all |
| Foundations | Convex vs non-convex; local/global minima; saddle points | A | `tracks/A_continuous/contopt/objectives.py` | A Exp. 1–2 |
| Foundations | Quantum computing basics; optimization in QML | C, I | `tracks/C_quantum_optimization/README.md`, `tracks/I_quantum_algorithms_qml/qalgo/qml_models.py` | C Exp. 1–2, I Exp. 6 |
| Continuous | Gradient descent | A | `tracks/A_continuous/contopt/optimizers.py` (`GD`) | A Exp. 1, 3, 6 |
| Continuous | SGD, mini-batch SGD | A | `tracks/A_continuous/contopt/optimizers.py` (`run_first_order(batch=…)`) | A Exp. 3, 5, 7 |
| Continuous | Momentum | A | `tracks/A_continuous/contopt/optimizers.py` | A Exp. 1, 3, 5, 6 |
| Continuous | Nesterov (NAG) | A | `tracks/A_continuous/contopt/optimizers.py` | A Exp. 1, 3, 5, 6 |
| Continuous | AdaGrad | A | `tracks/A_continuous/contopt/optimizers.py` | A Exp. 3, 5, 6 |
| Continuous | RMSProp | A | `tracks/A_continuous/contopt/optimizers.py` | A Exp. 3, 5, 6 |
| Continuous | Adam | A, C | `tracks/A_continuous/contopt/optimizers.py` | A Exp. 3, 5–7 |
| Continuous | Newton's method | A | `tracks/A_continuous/contopt/optimizers.py` (`newton`) | A Exp. 1–4 |
| Continuous | Gauss–Newton | A | `tracks/A_continuous/contopt/optimizers.py` (`gauss_newton`, `levenberg_marquardt`) | A Exp. 4 |
| Continuous | Backpropagation (vs optimizer) | A | `tracks/A_continuous/contopt/mlp.py` (`backprop`) | A Exp. 5 |
| Continuous | Fisher / diagonal / low-rank curvature, Hessian-vector products | A | `tracks/A_continuous/contopt/optimizers.py` (`natural_gradient`, `diag_newton`, `lbfgs`, `newton_cg`) | A Exp. 3 |
| Continuous | Convergence: convex, strongly convex, non-convex | A | `tracks/A_continuous/run_experiments.py` (`exp1`) | A Exp. 1 |
| Continuous | Step-size / optimizer tuning; noisy objectives | A, C | `tracks/A_continuous/run_experiments.py` (`exp6`, `exp7`) | A Exp. 6–7, C Exp. 2–3 |
| Discrete | Greedy | B, E, F, H, J | `tracks/B_discrete/discrete/knapsack.py`, `tracks/E_exam_timetabling/timetabling/greedy.py`, `tracks/H_dbms/tpchopt/selection.py`, `tracks/F_compiler/tinyc/backend.py` (greedy colouring), `tracks/J_software_engineering/tsmin/select.py` (`greedy`) | B Exp. 1, E Exp. 1, H, F Exp. 3, J Exp. 2 |
| Discrete | Integer Linear Programming | B, E, F, H, J | `tracks/B_discrete/discrete/knapsack.py` (`ilp`), `tracks/E_exam_timetabling/timetabling/ilp.py`, `tracks/H_dbms/tpchopt/selection.py` (`ilp`), `tracks/F_compiler/tinyc/backend.py` (`allocate(..., 'ilp')`), `tracks/J_software_engineering/tsmin/select.py` (`ilp`) | B Exp. 1, E Exp. 1–2, H, F Exp. 3, J Exp. 2 |
| Discrete | Branch-and-bound | B, E | `tracks/B_discrete/discrete/knapsack.py` (`branch_and_bound`, own code) | B Exp. 1–2 |
| Discrete | Dynamic programming | B, H | `tracks/B_discrete/discrete/knapsack.py` (`dp`), `tracks/H_dbms/tpchopt/selection.py` (`knapsack_dp`) | B Exp. 1–2, H |
| Discrete | Feature selection / combinatorial applications | B, D, J | `tracks/B_discrete/discrete/feature_selection.py`, `tracks/D_submodular_parallel_distributed/distopt/submodular.py` (`R2Selection`), `tracks/J_software_engineering/tsmin/select.py` (test selection = set cover) | B Exp. 5, D Exp. 1–2, J Exp. 2–6 |
| Discrete | QUBO / Ising, binary-to-spin, penalty terms | B | `tracks/B_discrete/discrete/qubo.py` | B Exp. 3–4 |
| Quantum | QAOA | C | `tracks/C_quantum_optimization/qopt/maxcut.py` | C Exp. 1–4 |
| Quantum | VQE | C | `tracks/C_quantum_optimization/qopt/vqe.py` | C Exp. 2, 6 |
| Quantum | PQCs / QNNs | C, I | `tracks/C_quantum_optimization/qopt/vqe.py` (HEA), `tracks/C_quantum_optimization/qopt/barren.py` | C Exp. 2, 5 |
| Quantum | Parameter-shift rule | C | `tracks/C_quantum_optimization/qopt/train.py`, `tracks/C_quantum_optimization/qopt/barren.py` | C Exp. 2–3, 5 |
| Quantum | SPSA | C | `tracks/C_quantum_optimization/qopt/train.py` | C Exp. 2 |
| Quantum | Quantum natural gradient | C | `tracks/C_quantum_optimization/qopt/train.py` | C Exp. 2 |
| Quantum | Barren plateaus, finite shots, hardware noise | C | `tracks/C_quantum_optimization/qopt/barren.py`, `tracks/C_quantum_optimization/qopt/maxcut.py` | C Exp. 3–5 |
| Quantum | Deutsch–Jozsa | I | `tracks/I_quantum_algorithms_qml/qalgo/algorithms.py` (hand-written) | I Exp. 1 |
| Quantum | Bernstein–Vazirani | I | `tracks/I_quantum_algorithms_qml/qalgo/algorithms.py` (hand-written) | I Exp. 1, 3 |
| Quantum | Grover | I | `tracks/I_quantum_algorithms_qml/qalgo/algorithms.py` (hand-written) | I Exp. 1–3 |
| Quantum | Quantum Fourier Transform | I | `tracks/I_quantum_algorithms_qml/qalgo/algorithms.py` (hand-written) | I Exp. 1 |
| Quantum | Quantum Phase Estimation | I | `tracks/I_quantum_algorithms_qml/qalgo/algorithms.py` (hand-written) | I Exp. 1 |
| Quantum | Shor | I | `tracks/I_quantum_algorithms_qml/qalgo/algorithms.py` (order finding, N = 15) | I Exp. 1 |
| Quantum | Quantum annealing; QUBO embedding limits | I | `tracks/I_quantum_algorithms_qml/qalgo/annealing.py` | I Exp. 4–5 |
| Quantum | Variational classifier; quantum kernels | I | `tracks/I_quantum_algorithms_qml/qalgo/qml_models.py` | I Exp. 6 |
| Submodular | Diminishing returns test; greedy; guarantee conditions | D | `tracks/D_submodular_parallel_distributed/distopt/submodular.py` (`test_properties`, `greedy`, `lazy_greedy`) | D Exp. 1–2 |
| Parallel | Load balancing, makespan, work stealing | D, G | `tracks/D_submodular_parallel_distributed/distopt/parallel.py` (`load_balance`, `simulate`, `optimal_makespan`); G: `tracks/G_xv6/patches/xv6-optimize.patch` (MLFQ) | D Exp. 4, G sched benchmark |
| Parallel | Locality, synchronization, communication overhead | D | `tracks/D_submodular_parallel_distributed/distopt/parallel.py` (`locality`, `synchronization`, `overhead`) | D Exp. 5 |
| Parallel | Data / model / pipeline parallelism | D | `tracks/D_submodular_parallel_distributed/distopt/parallel.py` (`data_parallel`, `model_parallel`, `pipeline`) | D Exp. 6 |
| Parallel | Speedup, efficiency, Amdahl, Gustafson | D | `tracks/D_submodular_parallel_distributed/distopt/parallel.py` (`run_parallel`, `amdahl`, `gustafson`) | D Exp. 3 |
| Strategy | Method selection by variable type, size, hardware … | E, J | `tracks/E_exam_timetabling/README.md` §3, `tracks/J_software_engineering/README.md` (profile first: minimization vs fixing the bottleneck) | table, J Exp. 1, 5 |
| Distributed | FedAvg vs centralized | D | `tracks/D_submodular_parallel_distributed/distopt/learning.py` (`fedavg`, `centralized`) | D Exp. 7 |
| Multi-task | MGDA | D | `tracks/D_submodular_parallel_distributed/distopt/learning.py` (`mgda_alpha`, `train_multitask`) | D Exp. 8 |
| Multi-task | GradNorm | D | `tracks/D_submodular_parallel_distributed/distopt/learning.py` (`train_multitask('gradnorm')`) | D Exp. 8 |
