"""Track G: build each xv6 variant, boot it in QEMU, run tests and benchmarks.

    python run_benchmarks.py                 # everything (~1 h: usertests dominate)
    python run_benchmarks.py --quick         # 2 repeats, usertests -q
    python run_benchmarks.py --only tests    # just correctness (cowtest + usertests)
    python run_benchmarks.py --only sched cow

Variants (same source tree, compile-time switches in the Makefile):
    original : SCHED=RR   COW=0   (upstream xv6 behaviour + statistics syscalls)
    mlfq     : SCHED=MLFQ COW=0
    cow      : SCHED=RR   COW=1
    both     : SCHED=MLFQ COW=1

Benchmarks run with CPUS=1 so the scheduling policy, not parallelism, decides
the order of execution. Correctness runs use the default CPUS=3 as well, because
races only show up with several CPUs.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import subprocess
import time
from pathlib import Path

import pexpect

HERE = Path(__file__).parent
XV6 = HERE / "xv6-riscv"
RESULTS = HERE / "results"
VARIANTS = {"original": ("RR", 0), "mlfq": ("MLFQ", 0), "cow": ("RR", 1), "both": ("MLFQ", 1)}
PROMPT = r"\n\$ "


def build(variant: str):
    sched, cow = VARIANTS[variant]
    subprocess.run(["make", "clean"], cwd=XV6, check=True, capture_output=True)
    r = subprocess.run(["make", f"SCHED={sched}", f"COW={cow}", "kernel/kernel", "fs.img"],
                       cwd=XV6, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build {variant} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


class Machine:
    """One booted xv6 in QEMU, driven through its serial console."""

    def __init__(self, cpus: int, log):
        cmd = (f"qemu-system-riscv64 -machine virt -bios none -kernel kernel/kernel -m 128M "
               f"-smp {cpus} -nographic -global virtio-mmio.force-legacy=false "
               f"-drive file=fs.img,if=none,format=raw,id=x0 "
               f"-device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0")
        self.p = pexpect.spawn(cmd, cwd=str(XV6), encoding="utf-8", timeout=120, logfile=log)
        self.p.expect(PROMPT)

    def run(self, command: str, timeout: int = 600) -> str:
        self.p.sendline(command)
        self.p.expect(PROMPT, timeout=timeout)
        out = self.p.before
        if "panic" in out:
            raise RuntimeError(f"kernel panic during '{command}':\n{out[-1500:]}")
        return out

    def close(self):
        self.p.sendcontrol("a")
        self.p.send("x")
        self.p.expect(pexpect.EOF, timeout=30)


def kv(line: str) -> dict:
    return dict(re.findall(r"(\w+)=([\w.?-]+)", line))


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


# ------------------------------------------------------------------ correctness
def correctness(quick: bool, log):
    rows = []
    for variant in VARIANTS:
        build(variant)
        for cpus in (1, 3):
            m = Machine(cpus, log)
            t0 = time.time()
            cow = m.run("cowtest", timeout=300)
            ut = m.run("usertests -q" if quick else "usertests", timeout=3600)
            m.close()
            failed = re.findall(r"test (\w+): FAILED", ut) or re.findall(r"FAILED", ut)
            rows.append(dict(variant=variant, cpus=cpus,
                             cowtest="OK" if "cowtest: OK" in cow else "FAIL",
                             usertests="ALL TESTS PASSED" if "ALL TESTS PASSED" in ut else f"FAILED {failed}",
                             usertests_mode="-q" if quick else "full",
                             minutes=round((time.time() - t0) / 60, 1)))
            print(f"   {variant} cpus={cpus}: cowtest {rows[-1]['cowtest']}, usertests {rows[-1]['usertests']}",
                  flush=True)
    return rows


# ------------------------------------------------------------------ scheduler
def calibrate(log) -> int:
    """Measure loop iterations per 0.1 s tick once, on the ORIGINAL kernel, and use
    the same number for both kernels so they run identical work (failure log F8)."""
    build("original")
    m = Machine(1, log)
    vals = []
    for _ in range(3):
        out = m.run("schedbench calib", timeout=120)
        vals += [int(x) for x in re.findall(r"CALIB iters_per_tick=(\d+)", out)]
    m.close()
    return sorted(vals)[len(vals) // 2]


def sched_bench(repeats: int, log):
    rows = []
    ipt = calibrate(log)
    print(f"   calibrated: {ipt} iterations per tick (median of 3, original kernel)", flush=True)
    for variant in ("original", "mlfq"):
        build(variant)
        for rep in range(repeats):
            m = Machine(1, log)
            for wl in ("cpu", "mixed", "shortlong", "shortlate"):
                out = m.run(f"schedbench {wl} {ipt}", timeout=900)
                for line in out.splitlines():
                    line = line.strip()
                    if line[:2] in ("R ", "L ", "T "):
                        d = kv(line)
                        d.update(kind=line[0], variant=variant, rep=rep)
                        rows.append(d)
            m.close()
            print(f"   {variant} repeat {rep + 1}/{repeats} done", flush=True)
    return rows


def sched_summary(rows):
    out = []
    for wl in ("cpu", "mixed", "shortlong", "shortlate"):
        for variant in ("original", "mlfq"):
            rs = [r for r in rows if r["wl"] == wl and r["variant"] == variant]
            for cls in dict.fromkeys(r["cls"] for r in rs if r["kind"] == "R"):
                c = [r for r in rs if r["kind"] == "R" and r["cls"] == cls]
                turn = [int(r["turn"]) / 1e6 for r in c]
                resp = [int(r["resp"]) / 1e3 for r in c]
                out.append(dict(workload=wl, variant=variant, metric=f"{cls} turnaround (s)",
                                mean=round(st.mean(turn), 3), std=round(st.pstdev(turn), 3), n=len(turn)))
                out.append(dict(workload=wl, variant=variant, metric=f"{cls} response (ms)",
                                mean=round(st.mean(resp), 1), std=round(st.pstdev(resp), 1), n=len(resp)))
                if cls != "interactive":
                    sw = [int(r["nsched"]) for r in c]
                    out.append(dict(workload=wl, variant=variant, metric=f"{cls} times scheduled",
                                    mean=round(st.mean(sw), 1), std=round(st.pstdev(sw), 1), n=len(sw)))
            lat = [int(r["lat_mean"]) / 1e3 for r in rs if r["kind"] == "L"]
            if lat:
                out.append(dict(workload=wl, variant=variant, metric="interactive sleep(1) latency (ms)",
                                mean=round(st.mean(lat), 1), std=round(st.pstdev(lat), 1), n=len(lat)))
                mx = [int(r["lat_max"]) / 1e3 for r in rs if r["kind"] == "L"]
                out.append(dict(workload=wl, variant=variant, metric="interactive worst latency (ms)",
                                mean=round(st.mean(mx), 1), std=round(st.pstdev(mx), 1), n=len(mx)))
            ms = [int(r["makespan"]) / 1e6 for r in rs if r["kind"] == "T"]
            out.append(dict(workload=wl, variant=variant, metric="makespan (s)",
                            mean=round(st.mean(ms), 3), std=round(st.pstdev(ms), 3), n=len(ms)))
    return out


# ------------------------------------------------------------------ COW
def cow_bench(repeats: int, log):
    rows = []
    for variant in ("original", "cow"):
        build(variant)
        for rep in range(repeats):
            m = Machine(1, log)
            for wl in ("forkexit", "forkexec", "forkread", "forkwrite", "capacity"):
                out = m.run(f"cowbench {wl}", timeout=900)
                for line in out.splitlines():
                    if line.strip().startswith("C "):
                        d = kv(line.strip())
                        d.update(variant=variant, rep=rep)
                        rows.append(d)
            m.close()
            print(f"   {variant} repeat {rep + 1}/{repeats} done", flush=True)
    return rows


def cow_summary(rows):
    out = []
    for wl in ("forkexit", "forkexec", "forkread", "forkwrite", "capacity"):
        for variant in ("original", "cow"):
            rs = [r for r in rows if r["wl"] == wl and r["variant"] == variant]
            if not rs:
                continue
            t = [int(r["mean_us"]) for r in rs]
            out.append(dict(workload=wl, variant=variant, runs=len(rs),
                            fork_cycle_us=f"{st.mean(t):.0f} ± {st.pstdev(t):.0f}" if wl != "capacity" else "",
                            pages_copied_at_fork=round(st.mean(int(r["copied"]) for r in rs)),
                            cow_faults=round(st.mean(int(r["cow_faults"]) for r in rs)),
                            cow_copies=round(st.mean(int(r["cow_copies"]) for r in rs)),
                            min_free_pages=round(st.mean(int(r["free_min"]) for r in rs)),
                            live_children=round(st.mean(int(r["children"]) for r in rs)) if wl == "capacity" else ""))
    return out


# ------------------------------------------------------------------ plots + summary
def plots(sched_rows, cow_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if sched_rows:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [("mixed", "interactive sleep(1) latency (ms)", "Interactive wake-up latency"),
                  ("shortlong", "short turnaround (s)", "Short jobs, 0.5 s after long jobs"),
                  ("shortlate", "short turnaround (s)", "Short jobs, 1.5 s after long jobs"),
                  ("cpu", "makespan (s)", "4 CPU-bound jobs: makespan")]
        S = sched_summary(sched_rows)
        for ax, (wl, metric, title) in zip(axes, panels):
            vals = [next(r for r in S if r["workload"] == wl and r["variant"] == v and r["metric"] == metric)
                    for v in ("original", "mlfq")]
            ax.bar(["round robin\n(original)", "MLFQ"], [v["mean"] for v in vals],
                   yerr=[v["std"] for v in vals], capsize=4, color=["#999999", "#3b75af"])
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(metric)
        fig.suptitle("xv6 scheduler, CPUS=1, QEMU (mean ± std over repeats)")
        fig.tight_layout()
        fig.savefig(RESULTS / "sched.png", dpi=150)
    if cow_rows:
        C = cow_summary(cow_rows)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        wls = ["forkexit", "forkexec", "forkread", "forkwrite"]
        for k, v in enumerate(("original", "cow")):
            ys = [float(next(r for r in C if r["workload"] == w and r["variant"] == v)["fork_cycle_us"].split()[0]) / 1000
                  for w in wls]
            axes[0].bar([i + 0.4 * k for i in range(4)], ys, 0.4,
                        label="eager copy (original)" if v == "original" else "copy-on-write")
        axes[0].set_xticks([i + 0.2 for i in range(4)], wls)
        axes[0].set_ylabel("fork → child exit → wait (ms)")
        axes[0].set_title("Fork cycle time, 4 MB parent heap")
        axes[0].legend()
        cap = [next(r for r in C if r["workload"] == "capacity" and r["variant"] == v) for v in ("original", "cow")]
        axes[1].bar(["eager copy\n(original)", "copy-on-write"], [c["live_children"] for c in cap],
                    color=["#999999", "#3b75af"])
        axes[1].set_ylabel("children alive at once (limit 40)")
        axes[1].set_title("Capacity: 16 MB parent in 128 MB RAM")
        fig.tight_layout()
        fig.savefig(RESULTS / "cow.png", dpi=150)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def read_csv(name):
    p = RESULTS / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def write_summary():
    parts = ["# Track G results (generated by run_benchmarks.py)\n",
             "xv6-riscv under QEMU 8.2 (TCG emulation, 128 MB RAM). Times come from the RISC-V `time` CSR "
             "(10 MHz). Benchmarks use CPUS=1.\n"]
    t = read_csv("correctness.csv")
    if t:
        parts += ["## Correctness\n", md_table(t, ["variant", "cpus", "cowtest", "usertests", "usertests_mode",
                                                    "minutes"]), ""]
    s = read_csv("sched_summary.csv")
    if s:
        parts += ["## Feature 1 — MLFQ scheduler vs round robin\n",
                  md_table(s, ["workload", "variant", "metric", "mean", "std", "n"]), "\n![sched](sched.png)\n"]
    c = read_csv("cow_summary.csv")
    if c:
        parts += ["## Feature 2 — copy-on-write fork vs eager copy\n",
                  md_table(c, ["workload", "variant", "runs", "fork_cycle_us", "pages_copied_at_fork",
                               "cow_faults", "cow_copies", "min_free_pages", "live_children"]),
                  "\n![cow](cow.png)\n"]
    (RESULTS / "SUMMARY.md").write_text("\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", choices=["tests", "sched", "cow"])
    args = ap.parse_args()
    todo = set(args.only or ["tests", "sched", "cow"])
    repeats = 2 if args.quick else 5
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "qemu_console.log", "a") as log:
        if "sched" in todo:
            print("Scheduler benchmarks")
            r = sched_bench(repeats, log)
            write_csv(RESULTS / "sched_raw.csv", r)
            write_csv(RESULTS / "sched_summary.csv", sched_summary(r))
        if "cow" in todo:
            print("COW benchmarks")
            r = cow_bench(repeats, log)
            write_csv(RESULTS / "cow_raw.csv", r)
            write_csv(RESULTS / "cow_summary.csv", cow_summary(r))
        if "tests" in todo:
            print("Correctness: cowtest + usertests on every variant")
            write_csv(RESULTS / "correctness.csv", correctness(args.quick, log))
    sched_rows = read_csv("sched_raw.csv")
    cow_rows = read_csv("cow_raw.csv")
    plots(sched_rows, cow_rows)
    write_summary()
    print("Done. See results/")


if __name__ == "__main__":
    main()
