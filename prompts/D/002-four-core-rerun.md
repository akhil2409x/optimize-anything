# D · 002 · Rerun on 4 cores; strided-shard bug (F16)

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-11-01

## Original prompt

```
do it all for me
```

This was the reply to the reminder to rerun Track D's speed-up experiments
on a laptop with more cores.

## AI response (summary)

**Setup:**
- With the student's permission, Claude got access to `~/Documents` on the
  student's MacBook Air through the Claude desktop app.
- It copied the repository into the app's 4-core Linux VM and installed the
  dependencies.
- It ran Track D's tests (22 pass) and Exp 3–6 there.

**Results:**
- Strong scaling: 3.40× on 4 cores.
- Load balancing: the same pattern as on 2 cores.

**A bug the rerun exposed:**
- *Symptom:* data parallelism with data resident in the workers was slower
  than serial.
- *Cause:* each worker took the strided view `X[i::p]` (failure log F16).
  Timing a strided vs a contiguous quarter showed ~9×.
- *Fix:* contiguous blocks.

**After the fix:**
- The 2-core cloud machine shows 1.2–2.4× over serial.
- The 4-core VM is still slower than serial, because the kernels are
  memory-bound. This is reported as a negative result.

## Resulting code change

`distopt/parallel.py` (contiguous shards), `results/mac_4core/`, README §3
and §6; same commit as E/002.

## Verification note

To do by the student: explain why `X[i::p]` is slower than
`np.array_split(X, p)[i]` for a matrix product, using what Exp 5 measures
about memory access.
