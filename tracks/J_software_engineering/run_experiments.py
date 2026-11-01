"""Track J experiments: test-suite minimization and prioritization on more-itertools.

    ./setup.sh                          # clone the subject at v11.2.1
    python run_experiments.py           # ~60 min on 2 cores (mutation testing, coverage of a threaded test)
    python run_experiments.py --only 3 5   # rerun some experiments
    python run_experiments.py --remeasure   # also redo timing + coverage (~8 min more)

Cached measurements (results/measurements.json, results/kill_matrix.json) are reused
unless --remeasure is given.

Exp 1  Profile the baseline: per-test time distribution, the bottleneck tests.
Exp 2  Minimization as weighted set cover: random-order baseline vs greedy vs ILP,
       for statement coverage and statement+branch coverage; LP lower bound.
Exp 3  Verify for real: rerun each reduced suite under coverage (fresh process) and
       time it against the full suite.
Exp 4  The cost: mutation testing (1,100+ mutants) - faults the reduced suites miss,
       compared with random subsets of the same size.
Exp 5  Fix the bottleneck itself: shrink the two slowest tests to the smallest inputs
       that keep their coverage; check coverage and mutant kills; 2-worker CI makespan.
Exp 6  Prioritization: APFD / APFDc and time to detect 50 / 90 / 100 % of faults.
"""
from __future__ import annotations

import argparse
import heapq
import json
import platform
import random
import statistics as st
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsmin import collect as C, mutate as M, select as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
MEAS = RESULTS / "measurements.json"
KILLS = RESULTS / "kill_matrix.json"
PATCH = HERE / "patches" / "shrink-slow-tests.patch"
OVERHEAD = 0.001          # s per selected test added to its cost (pytest bookkeeping; also breaks 0.000 s ties)
SLOW = ["tests/test_more.py::TestConcurrentTee::test_concurrent_consumers",
        "tests/test_recipes.py::PrimeFunctionTests::test_primes"]
SECTIONS = RESULTS / "sections"
SUMMARY: list[str] = []
STATE: dict = {}


def out(line=""):
    print(line, flush=True)
    SUMMARY.append(line)


def flush_section(n):
    """Each experiment's text goes to results/sections/expN.md; SUMMARY.md joins them, so
    `--only` can rerun one experiment without losing the others."""
    SECTIONS.mkdir(parents=True, exist_ok=True)
    (SECTIONS / f"exp{n}.md").write_text("\n".join(SUMMARY) + "\n")
    SUMMARY.clear()


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    out("| " + " | ".join(cols) + " |")
    out("|" + "---|" * len(cols))
    for r in rows:
        out("| " + " | ".join(fmt[c].format(r[c]) if c in fmt and r[c] != "" else
                               (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    out()


def save(name, rows):
    (RESULTS / name).write_text(json.dumps(rows, indent=1, default=str))


def suite_time(tests, times):
    return sum(times[t] for t in tests)


def lpt_makespan(durations, p=2):
    heap = [0.0] * p
    for d in sorted(durations, reverse=True):
        heapq.heapreplace(heap, heap[0] + d)
    return max(heap)


# ------------------------------------------------------------------ Exp 1
def exp1(d):
    out("## Exp 1 · Profiling the baseline")
    out()
    times = d["times"]
    total = sum(times.values())
    srt = sorted(times.items(), key=lambda kv: -kv[1])
    out(f"Subject: more-itertools {d['commit'][:10]} (v11.2.1): {len(times)} test functions "
        f"(plus thousands of subTests), library ~7,300 lines, tests ~9,200 lines.")
    out(f"Sum of per-test times (median of 3 runs, no coverage): **{total:.2f} s**; "
        f"full `pytest` wall time per run: {', '.join(f'{w:.1f}' for w in d['full_suite_wall'])} s "
        f"(includes start-up and collection; the spread comes mostly from one threaded test).")
    out()
    rows, acc = [], 0.0
    for i, (t, s) in enumerate(srt[:8]):
        acc += s
        rows.append(dict(rank=i + 1, test=t.split("::", 1)[1], seconds=s, share=s / total, cumulative=acc / total))
    table(rows, ["rank", "test", "seconds", "share", "cumulative"],
          {"seconds": "{:.3f}", "share": "{:.1%}", "cumulative": "{:.1%}"})
    under = sum(1 for s in times.values() if s < 0.01)
    out(f"- {under} of {len(times)} tests ({under / len(times):.0%}) take under 10 ms. "
        f"The two slowest tests take {(srt[0][1] + srt[1][1]) / total:.0%} of all test time: **they are the bottleneck**.")
    out()
    save("exp1_profile.json", rows)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(srt) + 1), [sum(s for _, s in srt[:k]) / total for k in range(1, len(srt) + 1)])
    ax.set_xscale("log")
    ax.set_xlabel("tests, slowest first (log scale)")
    ax.set_ylabel("cumulative share of test time")
    fig.tight_layout()
    fig.savefig(RESULTS / "time_profile.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ Exp 2
def exp2(d):
    out("## Exp 2 · Minimization as weighted set cover")
    out()
    times = d["times"]
    cost = {t: times[t] + OVERHEAD for t in times}
    full_t = suite_time(times, times)
    rows, suites = [], {}
    for crit, kinds in (("statements", ("L",)), ("statements+branches", ("L", "A"))):
        cov = S.requirements(d["coverage"], kinds)
        U = S.universe(cov)
        ess = S.essential_tests(cov)
        lb = S.ilp(cov, cost, relax=True)[0]
        methods = []
        rnd = [S.random_until_cover(cov, seed) for seed in range(10)]
        methods.append(("random order until covered (10 seeds, mean)", None, rnd, 0.0))
        for name, fn in [("greedy, most new requirements", lambda: S.greedy(cov)),
                         ("greedy, new requirements per second", lambda: S.greedy(cov, cost)),
                         ("greedy per second + redundancy removal",
                          lambda: S.remove_redundant(S.greedy(cov, cost), cov, cost)),
                         ("ILP, minimise number of tests", lambda: S.ilp(cov)[0]),
                         ("ILP, minimise time", lambda: S.ilp(cov, cost)[0])]:
            t0 = time.perf_counter()
            sel = fn()
            methods.append((name, sel, None, time.perf_counter() - t0))
        for name, sel, many, secs in methods:
            if many:
                n = st.mean(len(s) for s in many)
                tt = st.mean(suite_time(s, times) for s in many)
                ok = all(S.is_cover(s, cov, U) for s in many)
            else:
                n, tt, ok = len(sel), suite_time(sel, times), S.is_cover(sel, cov, U)
                suites[(crit, name)] = sel
            rows.append(dict(criterion=crit, method=name, tests=n, tests_pct=n / len(times),
                             test_time_s=tt, time_pct=tt / full_t, covers_all=ok, solve_s=secs,
                             both_slow_tests_kept=(all(s in sel for s in SLOW) if sel else "")))
        ilp_obj = S.ilp(cov, cost)[1]
        rows.append(dict(criterion=crit, method=f"LP relaxation: lower bound {lb:.3f} vs ILP objective "
                         f"{ilp_obj:.3f} (both incl. 1 ms/test)", tests="", tests_pct="",
                         test_time_s="", time_pct="", covers_all="", solve_s="", both_slow_tests_kept=""))
        out(f"- **{crit}**: {len(U)} requirements; {len(ess)} essential tests (sole cover of some "
            f"requirement); both slow tests essential: {all(s in ess for s in SLOW)}.")
    out()
    out(f"Full suite: {len(times)} tests, {full_t:.2f} s of test time. Costs include {OVERHEAD * 1000:.0f} ms "
        "per test; the time ILP's objective counts that too.")
    out()
    table(rows, ["criterion", "method", "tests", "tests_pct", "test_time_s", "time_pct", "covers_all", "solve_s"],
          {"tests": "{:.0f}", "tests_pct": "{:.0%}", "test_time_s": "{:.2f}", "time_pct": "{:.1%}",
           "solve_s": "{:.2f}"})
    save("exp2_minimization.json", rows)
    STATE["suites"] = suites
    return suites


# ------------------------------------------------------------------ Exp 3
def exp3(d, suites, reps):
    out("## Exp 3 · Verification: rerun each reduced suite for real")
    out()
    full_U = S.universe(d["coverage"])
    rows = []
    picks = [("full suite", sorted(d["times"]))] + [
        (f"{c}: {m}", suites[(c, m)]) for c, m in [
            ("statements+branches", "ILP, minimise number of tests"),
            ("statements+branches", "ILP, minimise time"),
            ("statements+branches", "greedy per second + redundancy removal"),
            ("statements", "ILP, minimise time")]]
    cover = {}
    for name, sel in picks:
        U, _ = C.coverage_of(sel)
        cover[name] = U
    # timing runs interleaved (A B C D E, A B C D E, ...) so slow drifts hit every suite alike
    walls = {name: [] for name, _ in picks}
    sums = {name: [] for name, _ in picks}
    passed = {name: True for name, _ in picks}
    for _ in range(reps):
        for name, sel in picks:
            tm, ok, w = C.run_junit(sel)
            walls[name].append(w)
            sums[name].append(sum(tm.values()))
            passed[name] &= all(ok.values()) and len(ok) == len(sel)
    for name, sel in picks:
        U = cover[name]
        rows.append(dict(suite=name, tests=len(sel), wall_s_median=st.median(walls[name]),
                         wall_s_min=min(walls[name]), wall_s_max=max(walls[name]),
                         test_time_s_median=st.median(sums[name]), all_pass=passed[name],
                         missing_requirements=len(full_U - U),
                         missing_branches=len({r for r in full_U - U if r[0] == "A"})))
    table(rows, ["suite", "tests", "wall_s_median", "wall_s_min", "wall_s_max", "test_time_s_median", "all_pass",
                 "missing_requirements", "missing_branches"],
          {"wall_s_median": "{:.2f}", "wall_s_min": "{:.2f}", "wall_s_max": "{:.2f}", "test_time_s_median": "{:.2f}"})
    out(f"{reps} timing runs per suite, interleaved. `missing_*` compares a fresh coverage run of the reduced "
        "suite with the requirements covered by the full suite. Wall time includes ~0.5 s of pytest start-up "
        "and collection; `test_time_s` is the sum of junit per-test times.")
    out()
    save("exp3_verify.json", rows)


# ------------------------------------------------------------------ Exp 4
def build_kill_matrix(d, procs=2):
    lines = M.mutable_lines(d["coverage"], d["import_time"])
    mutants = M.generate(C.SUBJECT, sorted(lines), lines)
    covering = M.covering_tests(mutants, d["coverage"])
    dirs = [M.make_workdir(WORK / f"orig{i}") for i in range(procs)]
    t0 = time.time()
    kills = M.kill_matrix(mutants, covering, dirs, d["times"], WORK / "kills_orig.jsonl")
    took = time.time() - t0
    # the shrunk versions of the two slow tests, on the mutants they execute
    sdirs = [M.make_workdir(WORK / f"shrunk{i}", patch=PATCH) for i in range(procs)]
    cov_shr = {m.id: {t for t in covering[m.id] if t in SLOW} for m in mutants}
    kills_shr = M.kill_matrix(mutants, cov_shr, sdirs, d["times"], WORK / "kills_shrunk.jsonl")
    data = dict(mutants=[m.__dict__ for m in mutants], kills={k: sorted(v) if v is not None else None
                                                              for k, v in kills.items()},
                kills_shrunk={k: sorted(v) if v is not None else None for k, v in kills_shr.items()},
                seconds=took, commit=d["commit"])
    KILLS.write_text(json.dumps(data))
    return data


def load_kills():
    k = json.loads(KILLS.read_text())
    k["kills"] = {int(a): (set(b) if b is not None else None) for a, b in k["kills"].items()}
    k["kills_shrunk"] = {int(a): (set(b) if b is not None else None) for a, b in k["kills_shrunk"].items()}
    return k


def exp4(d, suites, k):
    out("## Exp 4 · What the reduced suites give up: mutation testing")
    out()
    muts = {m["id"]: m for m in k["mutants"]}
    kills = {m: v for m, v in k["kills"].items() if v}        # killed by the full suite
    survived = sum(1 for v in k["kills"].values() if v == set())
    out(f"{len(muts)} mutants on lines the suite executes; each was run against every test that executes "
        f"its line ({k['seconds'] / 60:.0f} min on 2 processes; a run over 3× its normal time + 10 s is "
        f"retried test by test and a test that still overruns counts as failing). Killed by the full suite: **{len(kills)}** "
        f"({len(kills) / len(muts):.1%}); survived: {survived} (missed faults or equivalent mutants).")
    out()
    rows = []
    rng = random.Random(0)
    alltests = sorted(d["times"])
    for (crit, name), sel in sorted(suites.items()):
        sset = set(sel)
        killed = sum(1 for v in kills.values() if v & sset)
        rnd = []
        for _ in range(20):                                   # random subsets of the same size
            r = set(rng.sample(alltests, len(sel)))
            rnd.append(sum(1 for v in kills.values() if v & r) / len(kills))
        missed = [m for m, v in kills.items() if not v & sset]
        rows.append(dict(criterion=crit, method=name, tests=len(sel), killed=killed,
                         kill_rate=killed / len(kills), random_same_size=st.mean(rnd), missed=len(missed),
                         missed_ops=", ".join(f"{op}: {n}" for op, n in sorted(
                             _count(muts[m]["operator"].split()[0] for m in missed).items()))))
    table(rows, ["criterion", "method", "tests", "kill_rate", "random_same_size", "missed", "missed_ops"],
          {"kill_rate": "{:.1%}", "random_same_size": "{:.1%}"})
    out("`kill_rate` = share of the full suite's killed mutants that the reduced suite also kills. "
        "`random_same_size` = mean over 20 random subsets with the same number of tests.")
    out()
    save("exp4_mutation.json", rows)
    best = max((r for r in rows if r["criterion"] == "statements+branches"), key=lambda r: r["kill_rate"])
    ex = [muts[m] for m, v in kills.items() if not v & set(suites[(best["criterion"], best["method"])])][:5]
    if ex:
        out(f"Examples of faults missed by `{best['method']}` (statements+branches):")
        out()
        for m in ex:
            out(f"- `{m['file']}:{m['line']}` `{m['original']}` → `{m['replacement']}`")
        out()


def _count(it):
    c = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


# ------------------------------------------------------------------ Exp 5
def exp5(d, suites, k, reps):
    out("## Exp 5 · Fixing the bottleneck: shrink the two slowest tests")
    out()
    out("`patches/shrink-slow-tests.patch`: `test_concurrent_consumers` 10⁵ items × 100 threads → 10⁴ × 10; "
        "`test_primes` checks primes below 10⁵ → below 10⁴. Sizes were chosen as the smallest tried that "
        "keep each test's statement and branch coverage (10³ items lost the lock-contention branch "
        "`more.py:5571→5570` and the Miller–Rabin loop arc `recipes.py:1270→1269`).")
    out()
    wd = M.make_workdir(WORK / "shrunk_t", patch=PATCH)
    _, per = C.coverage_of(SLOW, wd)
    rows = []
    for t in SLOW:
        old = d["coverage"][t]
        tms = [C.run_junit([t], subject=wd)[0][t] for _ in range(reps)]
        rows.append(dict(test=t.split("::", 1)[1], before_s=d["times"][t], after_s=st.median(tms),
                         requirements_before=len(old), lost=len(old - per[t]), gained=len(per[t] - old)))
    table(rows, ["test", "before_s", "after_s", "requirements_before", "lost", "gained"],
          {"before_s": "{:.3f}", "after_s": "{:.3f}"})
    # fault detection on the mutants these two tests execute
    kills, kshr = k["kills"], k["kills_shrunk"]
    rows2 = []
    for t in SLOW:
        ms = [m for m in kills if kills[m] is not None and t in _covering(d, k, m)]
        before = sum(1 for m in ms if t in kills[m])
        after = sum(1 for m in ms if kshr.get(m) and t in kshr[m])
        only_this = sum(1 for m in ms if kills[m] == {t})
        rows2.append(dict(test=t.split("::", 1)[1], mutants_executed=len(ms), killed_before=before,
                          killed_after=after, killed_only_by_this_test=only_this))
    table(rows2, ["test", "mutants_executed", "killed_before", "killed_after", "killed_only_by_this_test"])
    # combined effect, measured
    times_new = dict(d["times"])
    for r, t in zip(rows, SLOW):
        times_new[t] = r["after_s"]
    full_new = sorted(d["times"])
    sel = suites[("statements+branches", "ILP, minimise time")]
    variants = [("original full suite", full_new, C.SUBJECT, d["times"]),
                ("ILP-minimised (statements+branches), original tests", sel, C.SUBJECT, d["times"]),
                ("full suite, two tests shrunk", full_new, wd, times_new),
                ("ILP-minimised + two tests shrunk", sel, wd, times_new)]
    walls = {v[0]: [] for v in variants}
    for _ in range(reps):                                    # interleaved, as in Exp 3
        for name, tests, subj, _ in variants:
            walls[name].append(C.run_junit(tests, subject=subj)[2])
    rows3 = [dict(suite=name, tests=len(tests), test_time_s=suite_time(tests, tm),
                  makespan_2_workers_lpt=lpt_makespan([tm[t] for t in tests]),
                  measured_wall_s_median=st.median(walls[name]), measured_wall_s_min=min(walls[name]),
                  measured_wall_s_max=max(walls[name]))
             for name, tests, subj, tm in variants]
    table(rows3, ["suite", "tests", "test_time_s", "makespan_2_workers_lpt", "measured_wall_s_median",
                  "measured_wall_s_min", "measured_wall_s_max"],
          {"test_time_s": "{:.2f}", "makespan_2_workers_lpt": "{:.2f}", "measured_wall_s_median": "{:.2f}",
           "measured_wall_s_min": "{:.2f}", "measured_wall_s_max": "{:.2f}"})
    out(f"`test_time_s` uses per-test medians; measured wall times are {reps} interleaved runs each. "
        "`makespan_2_workers_lpt`: simulated CI time on 2 parallel workers with longest-first scheduling "
        "(Track D) - with the original tests it can never go below the 7 s threaded test.")
    out()
    save("exp5_bottleneck.json", dict(tests=rows, faults=rows2, suites=rows3))


def _covering(d, k, mid):
    if "covering" not in STATE:
        muts = [M.Mutant(**m) for m in k["mutants"]]
        STATE["covering"] = M.covering_tests(muts, d["coverage"])
    return STATE["covering"][mid]


# ------------------------------------------------------------------ Exp 6
def exp6(d, k):
    out("## Exp 6 · Prioritization: find faults early (all tests still run)")
    out()
    cov = S.requirements(d["coverage"], ("L", "A"))
    cost = {t: d["times"][t] + OVERHEAD for t in d["times"]}
    kills = {m: v for m, v in k["kills"].items() if v}
    rows, curves = [], {}
    for meth in ("original", "random", "total", "additional", "additional/s"):
        orders = [S.prioritize(cov, meth, cost, seed=s) for s in (range(10) if meth == "random" else [0])]
        r = dict(order=meth + (" (10 seeds)" if meth == "random" else ""),
                 APFD=st.mean(S.apfd(o, kills) for o in orders),
                 APFDc=st.mean(S.apfd_c(o, kills, cost) for o in orders))
        for f in (0.5, 0.9, 1.0):
            r[f"s_to_{int(f * 100)}pct"] = st.mean(S.time_to_find(o, kills, cost, f) for o in orders)
        rows.append(r)
        o = orders[0]
        pos = {t: i for i, t in enumerate(o)}
        first = sorted(min(pos[t] for t in v) for v in kills.values())
        cum, acc = [], 0.0
        for t in o:
            acc += cost[t]
            cum.append(acc)
        curves[meth] = ([cum[i] for i in first], [(j + 1) / len(first) for j in range(len(first))])
    table(rows, ["order", "APFD", "APFDc", "s_to_50pct", "s_to_90pct", "s_to_100pct"],
          {"APFD": "{:.3f}", "APFDc": "{:.3f}", "s_to_50pct": "{:.2f}", "s_to_90pct": "{:.2f}",
           "s_to_100pct": "{:.2f}"})
    out("APFD counts tests; APFDc weighs each test by its time (1 = every fault found instantly). "
        "`s_to_X` = seconds of test time until X% of the killable mutants are detected.")
    out()
    save("exp6_prioritization.json", rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, (x, y) in curves.items():
        ax.step(x, y, where="post", label=m)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("seconds of test time spent")
    ax.set_ylabel("fraction of faults (mutants) detected")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "prioritization.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 timing repetition instead of 7")
    ap.add_argument("--only", type=int, nargs="*", help="experiments to (re)run; Exp 2 always runs")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    reps = 1 if a.quick else 7
    if a.remeasure or not MEAS.exists():
        C.collect(MEAS)
    d = C.load(MEAS)
    only = set(a.only or range(1, 7))
    if 1 in only:
        exp1(d)
        flush_section(1)
    suites = exp2(d)
    flush_section(2)
    if 3 in only:
        exp3(d, suites, reps)
        flush_section(3)
    if only & {4, 5, 6}:
        if a.remeasure or not KILLS.exists():
            build_kill_matrix(d)
        k = load_kills()
        for n, fn in ((4, lambda: exp4(d, suites, k)), (5, lambda: exp5(d, suites, k, reps)), (6, lambda: exp6(d, k))):
            if n in only:
                fn()
                flush_section(n)
    head = (f"# Track J results\n\nMachine: {platform.platform()}, Python {platform.python_version()}, "
            f"2 CPU cores. Last run {time.strftime('%Y-%m-%d %H:%M')}.\n\n")
    parts = [(SECTIONS / f"exp{n}.md").read_text() for n in range(1, 7) if (SECTIONS / f"exp{n}.md").exists()]
    (RESULTS / "SUMMARY.md").write_text(head + "\n".join(parts))


if __name__ == "__main__":
    main()
