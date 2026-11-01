"""Parallel program optimization: measured on real processes, plus a scheduler
simulator driven by measured task times.

Real measurements use `multiprocessing` (separate processes, so Python's GIL does
not serialise them) with BLAS/OpenMP threads pinned to 1 so NumPy cannot secretly
use extra cores. On a machine with C cores, speed-up cannot exceed C; running
more processes than cores (oversubscription) is measured on purpose.

Concepts and where they are shown
  speed-up S(p) = T(1)/T(p), efficiency E(p) = S(p)/p          strong_scaling
  Amdahl:     S(p) = 1 / ((1 - f) + f / p)   (fixed problem; f = parallel fraction)
  Gustafson:  S(p) = (1 - f) + f p           (problem grows with p)   weak_scaling
  load balancing, makespan                                       load_balance, simulate
  work stealing                                                  simulate (policy 'steal')
  granularity / communication overhead                           overhead
  locality                                                       locality
  synchronization                                                synchronization
  data / model / pipeline parallelism                            data_parallel, model_parallel, pipeline
"""
from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
import random
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np   # noqa: E402

CTX = mp.get_context("fork")
CORES = len(os.sched_getaffinity(0))


# ------------------------------------------------------------------ CPU-bound work
def work(n: int) -> int:
    """Deterministic CPU-bound task: sum of Collatz sequence lengths for a range.
    Pure Python, so it is limited by one core per process."""
    total = 0
    for x in range(1, n + 1):
        m, steps = x, 0
        while m != 1:
            m = m // 2 if m % 2 == 0 else 3 * m + 1
            steps += 1
        total += steps
    return total


def _work_range(args):
    lo, hi = args
    total = 0
    for x in range(lo, hi):
        m, steps = x, 0
        while m != 1:
            m = m // 2 if m % 2 == 0 else 3 * m + 1
            steps += 1
        total += steps
    return total


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


# ------------------------------------------------------------------ strong and weak scaling
def run_parallel(n_items: int, procs: int, serial_s: float = 0.0, chunks_per_proc: int = 4):
    """Serial part (a busy loop of `serial_s` seconds) + parallel Collatz over 1..n_items."""
    t0 = time.perf_counter()
    end = time.perf_counter() + serial_s
    while time.perf_counter() < end:            # stands for unparallelisable set-up / I/O
        pass
    n_chunks = procs * chunks_per_proc
    edges = np.linspace(1, n_items + 1, n_chunks + 1).astype(int)
    tasks = list(zip(edges[:-1], edges[1:]))
    if procs == 1:
        total = sum(_work_range(t) for t in tasks)
    else:
        with CTX.Pool(procs) as pool:
            total = sum(pool.map(_work_range, tasks, chunksize=1))
    return total, time.perf_counter() - t0


def amdahl(p, f):
    return 1.0 / ((1 - f) + f / p)


def gustafson(p, f):
    return (1 - f) + f * p


def fit_parallel_fraction(ps, speedups):
    """Least-squares fit of Amdahl's f on the measured speed-ups (p <= cores)."""
    fs = np.linspace(0, 1, 10001)
    err = [sum((amdahl(p, f) - s) ** 2 for p, s in zip(ps, speedups)) for f in fs]
    return float(fs[int(np.argmin(err))])


# ------------------------------------------------------------------ load balancing (real)
def heterogeneous_tasks(n_tasks=48, seed=0):
    """Task sizes that grow along the list (later ranges have longer Collatz chains
    and are larger), so contiguous blocks are badly unbalanced."""
    rng = np.random.default_rng(seed)
    sizes = (np.linspace(200, 6000, n_tasks) * rng.uniform(0.7, 1.3, n_tasks)).astype(int)
    starts = np.cumsum(np.concatenate([[1], sizes[:-1]]))
    return [(int(s), int(s + z)) for s, z in zip(starts, sizes)]


def _timed_task(t):
    t0 = time.perf_counter()
    r = _work_range(t)
    return r, time.perf_counter() - t0, os.getpid()


def load_balance(tasks, procs, policy):
    """policy: 'block' (contiguous static), 'cyclic' (round-robin static),
    'dynamic' (shared queue, chunksize 1), 'dynamic-big' (chunksize = n/p).
    Returns makespan and per-worker busy time."""
    if policy in ("block", "cyclic"):
        if policy == "block":
            k = math.ceil(len(tasks) / procs)
            groups = [tasks[i * k:(i + 1) * k] for i in range(procs)]
        else:
            groups = [tasks[i::procs] for i in range(procs)]
        t0 = time.perf_counter()
        with CTX.Pool(procs) as pool:
            res = pool.map(_run_group, groups, chunksize=1)
        makespan = time.perf_counter() - t0
        busy = [b for _, b in res]
        total = sum(r for r, _ in res)
    else:
        cs = 1 if policy == "dynamic" else math.ceil(len(tasks) / procs)
        t0 = time.perf_counter()
        with CTX.Pool(procs) as pool:
            res = pool.map(_timed_task, tasks, chunksize=cs)
        makespan = time.perf_counter() - t0
        per = {}
        for _, dt, pid in res:
            per[pid] = per.get(pid, 0) + dt
        busy = list(per.values()) + [0.0] * (procs - len(per))
        total = sum(r for r, _, _ in res)
    return dict(policy=policy, makespan=makespan, busy=busy, total=total,
                imbalance=max(busy) / (sum(busy) / procs) if sum(busy) else 1.0)


def _run_group(group):
    t0 = time.perf_counter()
    r = sum(_work_range(t) for t in group)
    return r, time.perf_counter() - t0


# ------------------------------------------------------------------ scheduler simulator
def simulate(durations, p, policy, steal_seed=0, overhead=0.0):
    """Makespan of scheduling `durations` on p identical workers.
      block   : contiguous blocks, static
      cyclic  : task i -> worker i mod p, static
      dynamic : list scheduling in the given order (each idle worker takes the next task)
      lpt     : longest processing time first, then list scheduling (4/3-approximation)
      steal   : work stealing; every worker starts with a contiguous block in a deque,
                takes from its own head; when empty it steals from the TAIL of a random
                non-empty victim (cost `overhead` per steal)
    """
    d = list(durations)
    if policy in ("block", "cyclic"):
        k = math.ceil(len(d) / p)
        groups = [d[i * k:(i + 1) * k] for i in range(p)] if policy == "block" else [d[i::p] for i in range(p)]
        return max(sum(g) for g in groups), {"steals": 0}
    if policy in ("dynamic", "lpt"):
        order = sorted(d, reverse=True) if policy == "lpt" else d
        heap = [0.0] * p
        for t in order:
            heapq.heapreplace(heap, heap[0] + t + overhead)
        return max(heap), {"steals": 0}
    if policy == "steal":
        rng = random.Random(steal_seed)
        k = math.ceil(len(d) / p)
        deques = [list(d[i * k:(i + 1) * k]) for i in range(p)]
        clock = [0.0] * p
        steals = 0
        events = [(0.0, w) for w in range(p)]
        heapq.heapify(events)
        while events:
            now, w = heapq.heappop(events)
            if deques[w]:
                t = deques[w].pop(0)                       # own work from the head
                clock[w] = now + t
                heapq.heappush(events, (clock[w], w))
                continue
            victims = [v for v in range(p) if deques[v]]
            if not victims:
                clock[w] = max(clock[w], now)
                continue
            v = rng.choice(victims)
            stolen = deques[v][len(deques[v]) // 2:]      # steal the far half
            deques[v] = deques[v][:len(deques[v]) // 2]
            deques[w].extend(stolen)
            steals += 1
            heapq.heappush(events, (now + overhead, w))
        return max(clock), {"steals": steals}
    raise ValueError(policy)


def optimal_makespan(durations, p, time_limit=30):
    """P || Cmax as an ILP (exact) for small instances."""
    import pulp
    n = len(durations)
    prob = pulp.LpProblem("cmax", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x{i}_{w}", cat="Binary") for w in range(p)] for i in range(n)]
    C = pulp.LpVariable("C", lowBound=0)
    prob += C
    for i in range(n):
        prob += pulp.lpSum(x[i]) == 1
    for w in range(p):
        prob += pulp.lpSum(durations[i] * x[i][w] for i in range(n)) <= C
    prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0.0))
    return float(C.value()), pulp.LpStatus[prob.status]


# ------------------------------------------------------------------ overheads
def _tiny(x):
    return x * x


def _sum_array(a):
    return float(a.sum())


def overhead(procs=2):
    """(1) per-task scheduling overhead vs chunk size; (2) shipping data vs computing on it."""
    rows = []
    n = 200_000
    t_serial = timed(lambda: [_tiny(i) for i in range(n)])[1]
    rows.append(dict(test="200k tiny tasks", setting="serial loop", seconds=t_serial))
    with CTX.Pool(procs) as pool:
        for cs in (1, 100, 10_000):
            _, t = timed(pool.map, _tiny, range(n), cs)
            rows.append(dict(test="200k tiny tasks", setting=f"{procs} processes, chunksize {cs}", seconds=t))
        arrays = [np.ones(2_000_000) for _ in range(8)]                 # 8 x 16 MB
        _, t_local = timed(lambda: [float(a.sum()) for a in arrays])
        _, t_ship = timed(pool.map, _sum_array, arrays, 1)
        rows.append(dict(test="sum 8 arrays of 16 MB", setting="serial (no copying)", seconds=t_local))
        rows.append(dict(test="sum 8 arrays of 16 MB", setting=f"{procs} processes (arrays pickled to workers)",
                         seconds=t_ship))
    return rows


# ------------------------------------------------------------------ locality
def locality(n=3000, reps=3):
    a = np.random.default_rng(0).random((n, n))           # C order: rows are contiguous
    rows = []
    for name, fn in [("row by row (contiguous)", lambda: sum(float(a[i, :].sum()) for i in range(n))),
                     ("column by column (stride n*8 bytes)", lambda: sum(float(a[:, j].sum()) for j in range(n)))]:
        ts = [timed(fn)[1] for _ in range(reps)]
        rows.append(dict(access=name, seconds=min(ts), ns_per_element=round(1e9 * min(ts) / n / n, 2)))
    return rows


# ------------------------------------------------------------------ synchronization
def _count_locked(args):
    counter, lock, n = args
    for _ in range(n):
        with lock:
            counter.value += 1


def _count_local(n):
    c = 0
    for _ in range(n):
        c += 1
    return c


def synchronization(n=40_000, procs=2):
    counter = CTX.Value("q", 0, lock=False)
    lock = CTX.Lock()
    t0 = time.perf_counter()
    ps = [CTX.Process(target=_count_locked, args=((counter, lock, n),)) for _ in range(procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    t_lock = time.perf_counter() - t0
    with CTX.Pool(procs) as pool:
        t0 = time.perf_counter()
        local = sum(pool.map(_count_local, [n] * procs))
        t_local = time.perf_counter() - t0
    return [dict(method="shared counter, lock per increment", total=counter.value, seconds=t_lock),
            dict(method="local counts, one reduction at the end", total=local, seconds=t_local)]


# ------------------------------------------------------------------ data / model / pipeline parallelism
def _grad_shard(args):
    X, y, w = args
    return X.T @ (X @ w - y), len(y)


_RESIDENT = {}          # data placed here BEFORE the pool forks stays in every worker (copy-on-write)


def _grad_resident(args):
    i, procs, w = args
    X, y = _RESIDENT["X"][i::procs], _RESIDENT["y"][i::procs]
    return X.T @ (X @ w - y), len(y)


def _matvec(args):
    W, x = args
    return W @ x


def _matvec_resident(args):
    i, procs, x = args
    return np.array_split(_RESIDENT["W"], procs)[i] @ x


def data_parallel(n=400_000, d=50, procs=2, steps=20):
    """Least-squares gradient computed on p data shards and averaged == the full gradient.
    'shipped': each call sends the shards to the workers (pickling ~160 MB);
    'resident': shards stay in the workers, only w (d numbers) and gradients travel,
    as in real data-parallel training. Timed over `steps` gradient evaluations."""
    rng = np.random.default_rng(0)
    X, w = rng.standard_normal((n, d)), rng.standard_normal(d)
    y = X @ rng.standard_normal(d)
    full = X.T @ (X @ w - y) / n
    _, t1 = timed(lambda: [X.T @ (X @ w - y) / n for _ in range(steps)])
    shards = [(X[i::procs], y[i::procs], w) for i in range(procs)]
    with CTX.Pool(procs) as pool:
        parts, t_ship = timed(lambda: [pool.map(_grad_shard, shards) for _ in range(steps)][-1])
    g_ship = sum(p for p, _ in parts) / n
    _RESIDENT.update(X=X, y=y)
    with CTX.Pool(procs) as pool:
        parts, t_res = timed(lambda: [pool.map(_grad_resident, [(i, procs, w) for i in range(procs)])
                                      for _ in range(steps)][-1])
    _RESIDENT.clear()
    g_res = sum(p for p, _ in parts) / n
    return dict(max_abs_diff=float(max(np.abs(g_ship - full).max(), np.abs(g_res - full).max())),
                steps=steps, serial_s=t1, shipped_s=t_ship, resident_s=t_res,
                bytes_per_step_shipped=int(X.nbytes + y.nbytes), bytes_per_step_resident=int((procs + procs) * d * 8))


def model_parallel(n=6000, procs=2, steps=20):
    """A layer y = W x split by rows across processes (each holds part of the model)."""
    rng = np.random.default_rng(1)
    W, x = rng.standard_normal((n, n)), rng.standard_normal(n)
    full = W @ x
    _, t1 = timed(lambda: [W @ x for _ in range(steps)])
    parts = np.array_split(W, procs)
    with CTX.Pool(procs) as pool:
        out, t_ship = timed(lambda: [pool.map(_matvec, [(Wp, x) for Wp in parts]) for _ in range(steps)][-1])
    y_ship = np.concatenate(out)
    _RESIDENT.update(W=W)
    with CTX.Pool(procs) as pool:
        out, t_res = timed(lambda: [pool.map(_matvec_resident, [(i, procs, x) for i in range(procs)])
                                    for _ in range(steps)][-1])
    _RESIDENT.clear()
    y_res = np.concatenate(out)
    return dict(max_abs_diff=float(max(np.abs(y_ship - full).max(), np.abs(y_res - full).max())),
                steps=steps, serial_s=t1, shipped_s=t_ship, resident_s=t_res,
                weights_per_process_mb=round(parts[0].nbytes / 2 ** 20, 1))


def _stage(inq, outq, n_items, fn_n):
    for _ in range(n_items):
        item = inq.get()
        outq.put(item + work(fn_n) * 0)


def pipeline(micro_batches=16, stage_work=2500):
    """Two-stage pipeline (each stage = CPU work), sequential vs pipelined on 2 processes.
    Ideal pipelined time ~ (M + S - 1) * t_stage for S stages and M micro-batches."""
    _, t_stage = timed(work, stage_work)
    _, t_seq = timed(lambda: [work(stage_work) + work(stage_work) for _ in range(micro_batches)])
    q01, q12, q2o = CTX.Queue(), CTX.Queue(), CTX.Queue()
    s1 = CTX.Process(target=_stage, args=(q01, q12, micro_batches, stage_work))
    s2 = CTX.Process(target=_stage, args=(q12, q2o, micro_batches, stage_work))
    s1.start(); s2.start()
    t0 = time.perf_counter()
    for i in range(micro_batches):
        q01.put(i)
    out = [q2o.get() for _ in range(micro_batches)]
    t_pipe = time.perf_counter() - t0
    s1.join(); s2.join()
    ideal = (micro_batches + 1) * t_stage
    return dict(micro_batches=micro_batches, t_stage=t_stage, sequential_s=t_seq, pipelined_s=t_pipe,
                ideal_pipelined_s=ideal, bubble_fraction=1 / (micro_batches + 1),
                in_order=out == list(range(micro_batches)))
