# Track G — xv6 operating-system optimization

Two kernel features, each benchmarked on at least three workloads before and
after, on the same machine and the same QEMU configuration:

1. **A multilevel feedback queue (MLFQ) scheduler** to replace round robin
   (concept: process scheduling; objectives: response time and turnaround).
2. **Copy-on-write (COW) fork** to replace eager copying (concept: virtual
   memory and page faults; objectives: fork latency, memory use, pages copied).

**What to submit:** `patches/xv6-optimize.patch`, which is 1,257 lines
against upstream xv6-riscv commit `e93dabeb` (2026-10-26). `setup.sh`
re-creates the patched tree from a fresh clone. This was verified in the
session.

## Correctness: xv6 still boots and passes usertests

Both features are compile-time switches, giving four kernels.

| Variant | Build | CPUS=1 | CPUS=3 |
|---|---|---|---|
| Original | `SCHED=RR COW=0` | cowtest OK, **usertests ALL TESTS PASSED** | cowtest OK, **ALL TESTS PASSED** |
| MLFQ | `SCHED=MLFQ COW=0` | OK, **ALL TESTS PASSED** | OK, **ALL TESTS PASSED** |
| COW | `SCHED=RR COW=1` | OK, **ALL TESTS PASSED** | OK, **ALL TESTS PASSED** |
| Both | `SCHED=MLFQ COW=1` | OK, **ALL TESTS PASSED** | OK, **ALL TESTS PASSED** |

- **usertests:** these are the full runs (not `-q`), about 3–4 minutes each.
- **cowtest:** my own test program, written so it passes on the original
  kernel too. It checks that:
  - a child sees its parent's data, and its writes stay private
  - the parent's writes after fork don't reach the child
  - a `read()` *by the kernel* into a shared page in the child leaves the
    parent's page unchanged (`copyout` breaks COW)
  - pages are shared correctly across three generations
  - no physical page leaks across 30 forks (free-page count before = after)

## Feature 1: MLFQ scheduler

**Design** (`kernel/proc.c`, `kernel/trap.c`):
- **Levels:** three priority levels with quanta of 1, 2 and 4 timer ticks
  (one tick ≈ 0.1 s). New processes start at level 0.
- **Demotion:** using a whole quantum moves a process down one level. Time used
  at a level is kept across sleeps, so a process can't stay on top by
  yielding just before its quantum ends.
- **Preemption:** on every tick, a process yields if a higher-priority process
  is runnable. Without this, an interactive process that wakes up would wait
  out a CPU hog's 4-tick quantum.
- **Boost:** every 50 ticks all processes return to level 0, so CPU-bound jobs
  can't starve.
- **Selection:** the scheduler finds the best level with an unlocked scan,
  then re-checks under `p->lock` and runs round robin within that level.

**Benchmark** (`user/schedbench.c`, CPUS=1, 5 repeats). Every kernel runs
identical work: loop iterations calibrated once on the original kernel.

| Workload | Metric | Round robin | MLFQ | Change |
|---|---|---|---|---|
| **mixed**: 3 CPU-bound + 2 interactive | interactive turnaround | 10.02 ± 0.11 s | 4.45 ± 0.01 s | **−56%** |
| | interactive wake-up latency after `pause(1)` | 242 ± 3 ms | 104 ± 0.1 ms | **−57%** (1 tick ≈ 100 ms is the floor) |
| | CPU-bound turnaround | 9.03 ± 0.20 s | 8.73 ± 0.34 s | −3% |
| **shortlong**: 4 short jobs arrive 0.5 s after 2 long ones | short-job turnaround | 1.82 ± 0.26 s | 1.29 ± 0.38 s | **−29%** |
| | short-job response | 351 ms | 151 ms | −57% |
| **shortlate**: same, short jobs arrive after 1.5 s | short-job turnaround | 1.82 ± 0.23 s | 1.06 ± 0.32 s | **−42%** |
| | long-job turnaround | 9.11 s | 8.91 s | not hurt |
| **cpu**: 4 identical CPU-bound jobs | makespan | 8.02 ± 0.10 s | 8.07 ± 0.18 s | no difference |
| | times each job was scheduled | 20.9 | 7.9 | −62% context switches |

**What this shows**
- **MLFQ approximates "shortest job first" without knowing job lengths.**
  Interactive and short jobs stay at high priority while CPU hogs sink.
- **It helps more when the hogs have already sunk** (−42% vs −29%). Reaching
  the bottom level takes 1 + 2 = 3 ticks of CPU; after a 0.5 s head start
  shared by two jobs, they have used only about 2.5 each.
- **Pure batch work gains nothing in total time.** It does gain 62% fewer
  context switches, which would matter more on real hardware with caches.
- **Short-job variance is higher under MLFQ** (±0.3–0.4 s vs ±0.25 s). The
  priority boost fires every 5 s of uptime regardless of when a workload
  starts, so sometimes it lifts the long jobs back to the top just as the short
  ones arrive.

## Feature 2: copy-on-write fork

**Design** (`kernel/vm.c`, `kernel/kalloc.c`, `kernel/trap.c`):
- **Reference counts:** each physical page has a count, protected by
  `kmem.lock`. `kfree` frees a page only when the count drops to 0.
- **`uvmcopy`:** maps the parent's pages into the child instead of copying
  them. Writable pages become read-only + `PTE_COW` (RSW bit 8) in **both**
  page tables, and the TLB is flushed.
- **`cowfault()`:** on a store page fault to a COW page, copy it, or just
  re-enable writing if this process is the last sharer.
- **`copyout()`:** breaks COW before the kernel writes into a user page. This
  is the case a naive COW implementation misses.
- **Lazy allocation still works:** unmapped pages still go through `vmfault`.

**Benchmark** (`user/cowbench.c`, 4 MB parent heap, 20 forks per run, 5 runs):

| Workload | Eager copy (original) | Copy-on-write | Change |
|---|---|---|---|
| fork → child exits → wait | 25.0 ± 1.9 ms | 1.73 ± 0.27 ms | **14× faster** |
| fork → child execs `nop` (shell pattern) | 26.2 ± 1.4 ms | 3.76 ± 0.43 ms | **7× faster** |
| fork → child reads every page | 24.9 ± 1.8 ms | 2.14 ± 0.39 ms | **12× faster** |
| fork → child writes every page | 24.7 ± 1.7 ms | 56.1 ± 3.8 ms | **2.3× slower** |
| Pages copied by 20 forks | 20,580 | 0 at fork, 21 on faults (exit case) | −99.9% |
| Children alive at once, 16 MB parent, 128 MB RAM | **6** (memory exhausted) | **40** (benchmark limit) | ≥ 6.7× |
| Minimum free pages during capacity test | 3,702 | 27,730 | |

**What this shows**
- **COW wins whenever the child writes little before exiting or exec'ing,**
  which is the common Unix pattern.
- **When the child writes everything it loses.** It pays 1,024 page faults
  (trap entry and exit, a lookup, a copy and a TLB flush each) instead of one
  tight copy loop: about 55 µs per page fault vs about 24 µs per page for the
  bulk copy. That is an honest negative result, and the reason real kernels
  pair COW with tricks such as pre-copying pages that are about to be written.
- **COW saves memory as well as time.** It held 40 live children of a 16 MB
  process where eager copying ran out of RAM at 6.

## How to run

```bash
# Linux or WSL
sudo apt-get install -y qemu-system-misc gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu make git
pip install pexpect matplotlib
cd tracks/G_xv6
./setup.sh                            # clone upstream at the pinned commit, apply the patch, build
python run_benchmarks.py              # ~45 min: benchmarks + full usertests on 4 kernels x 2 CPU counts
python run_benchmarks.py --only sched # just the scheduler benchmarks (~10 min)

# by hand
cd xv6-riscv && make SCHED=MLFQ COW=1 CPUS=1 qemu
$ cowtest
$ schedbench calib
$ schedbench mixed 39489513
$ cowbench forkexit
```

## Files

| Path | What it is |
|---|---|
| `patches/xv6-optimize.patch` | The submission: all kernel and user changes |
| `setup.sh` | Clone, pin, patch and build |
| `run_benchmarks.py` | Build each variant, drive QEMU with pexpect, parse, plot |
| `results/SUMMARY.md`, `*.csv`, `*.png` | Generated results |
| `results/qemu_console.log` | Raw console output of every run, including usertests |
| `results/superseded/` | The two scheduler runs made before fixing the calibration bug (failure log F8) |

## Limitations

- **QEMU, not hardware:** TCG emulation doesn't model caches or TLB costs
  realistically, so COW fault cost and context-switch savings would look
  different on real silicon.
- **Coarse timer:** the 100 ms tick makes MLFQ quanta coarse. Linux uses about
  1–4 ms ticks.
- **CPUS=1 for benchmarks:** this is on purpose, so the policy decides the
  order of execution. The correctness runs also use CPUS=3.
- **Approximate counters:** COW statistics are unlocked counters, exact on 1
  CPU and approximate under concurrency. They are only reported from 1-CPU runs.
