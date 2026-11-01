# G · 001 · MLFQ scheduler and copy-on-write fork in xv6

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-28
- **Kind of work:** planning, kernel code generation, testing, benchmarking, debugging, documentation

## Original prompt

Claude proposed the order G, H, F, A, D, J for the remaining tracks, saying G is
the riskiest (kernel C, QEMU), and asked whether to go straight into G. Reply:

```
yes
```

## AI response (summary)

Claude installed QEMU and the RISC-V cross-compiler in the cloud workspace,
cloned upstream xv6-riscv (commit e93dabeb, 2026-10-26), read the current
kernel sources (proc.c, vm.c, trap.c, kalloc.c), then wrote:

- **Feature 1:** an MLFQ scheduler. Three levels with quanta of 1, 2 and 4
  ticks, demotion when a quantum is used up, preemption by higher-priority
  processes, and a boost every 50 ticks.
- **Feature 2:** copy-on-write fork. Per-page reference counts, a PTE_COW
  software bit, a write-fault handler, and COW breaking in `copyout`.
- **Instrumentation:** the `waitstat`, `memstat` and `rtime` system calls, and
  per-process timing statistics.
- **User programs:** `cowtest` (correctness), `schedbench` and `cowbench`
  (benchmarks), and `nop`.
- **Scripts:** `run_benchmarks.py`, which builds the 4 variants, boots QEMU,
  runs the benchmarks and full usertests, and parses the output; and
  `setup.sh`, which recreates the tree from the patch.

Design points Claude identified before coding:
- **MLFQ needs preemption.** Without it, an interactive process that wakes up
  waits behind a 4-tick bottom-level quantum.
- **COW must coexist with lazy sbrk.** This xv6 version already has lazily
  allocated pages through `vmfault`.
- **`copyout` must break COW.** Otherwise `read()` into a shared page would
  change another process's memory.

## Revisions during the session

1. **Short jobs showed no MLFQ benefit at first.** Claude hypothesised that
   0.5 s of head start was too short for the long jobs to sink, added a 1.5 s
   variant (`shortlate`), and reports both.
2. **MLFQ looked 7–8% slower on CPU-bound jobs.** The cause was a benchmark
   bug: per-run calibration gave MLFQ 9% more work. Calibration is now done once
   on the original kernel (failure log F8), and the superseded runs are kept in
   `results/superseded/`.

## Resulting code change

`tracks/G_xv6/patches/xv6-optimize.patch`, commit
`[AI:claude] Track G: xv6 MLFQ scheduler and copy-on-write fork`
= local commit `b5d38d9` (on GitHub: `<repo URL>/commit/b5d38d9`)

## Verification note

Done in the session: `cowtest` and full `usertests` passed on all 4 kernel
variants, each with 1 and 3 CPUs (8/8).

To do by the student:
- Run `./setup.sh` and `make SCHED=MLFQ COW=1 qemu` yourself on Linux or WSL.
- Read `cowfault()` and the COW branch in `copyout()` line by line.
- Explain why `kfree` must decrement a reference count instead of freeing.
- Explain why the parent's PTEs are also made read-only in `uvmcopy`.
- Explain why the scheduler re-checks state under the lock after its unlocked
  scan.
