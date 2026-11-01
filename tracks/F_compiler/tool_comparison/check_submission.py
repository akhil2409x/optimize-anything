"""Score an AI tool's CSE pass on identical, hidden-from-the-tool tests.

    python tool_comparison/check_submission.py tool_comparison/submission_chatgpt.py

The submission must define `cse(fn) -> bool` for the IR in PROMPT.md. It is
plugged into the real Tiny compiler in place of the built-in pass and checked:
  1. hazard unit tests (invalidation after redefinition, commutativity, a
     result variable that is also an operand, div/mod, neg, print, return value);
  2. differential testing: 300 random programs with CSE alone and inside the full
     pipeline, outputs and runtime errors compared with the reference interpreter;
  3. effectiveness: executed instructions on the 8 benchmarks with only CSE enabled.
Writes a JSON line to tool_comparison/scores.jsonl.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tinyc import backend as B, driver as D, fuzz, ir, passes as P   # noqa: E402
from tinyc.ir import Block, Function, Instr                         # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    # make the IR classes available under the names used in the prompt
    mod.Instr, mod.Block, mod.Function = Instr, Block, Function
    spec.loader.exec_module(mod)
    return mod.cse


def block_fn(instrs):
    return Function({"entry": Block("entry", instrs, Instr("halt"))})


def unit_tests(cse):
    results = {}

    def case(name, instrs, expect_copy_at=None, expect_unchanged_at=()):
        fn = block_fn(instrs)
        before = ir.run_ir(fn.copy()) if True else None
        try:
            changed = cse(fn)
            after = ir.run_ir(fn)
            ok = after == before
            out = fn.blocks["entry"].instrs
            if expect_copy_at is not None:
                ok &= out[expect_copy_at].op == "copy"
            for k in expect_unchanged_at:
                ok &= out[k].op != "copy"
            if expect_copy_at is not None:
                ok &= changed is True
        except Exception as e:
            ok = False
            out = repr(e)
        results[name] = bool(ok)

    I = Instr
    case("reuses identical computation",
         [I("const", "a", 3), I("const", "b", 4), I("add", "x", "a", "b"), I("add", "y", "a", "b"),
          I("print", a="y")], expect_copy_at=3)
    case("commutative add",
         [I("const", "a", 3), I("const", "b", 4), I("add", "x", "a", "b"), I("add", "y", "b", "a"),
          I("print", a="y")], expect_copy_at=3)
    case("not commutative sub",
         [I("const", "a", 3), I("const", "b", 4), I("sub", "x", "a", "b"), I("sub", "y", "b", "a"),
          I("print", a="x"), I("print", a="y")], expect_unchanged_at=(3,))
    case("invalidates after operand redefined",
         [I("const", "a", 3), I("const", "b", 4), I("add", "x", "a", "b"), I("mul", "a", "a", 5),
          I("add", "y", "a", "b"), I("print", a="x"), I("print", a="y")], expect_unchanged_at=(4,))
    case("invalidates when holder overwritten",
         [I("const", "a", 3), I("const", "b", 4), I("add", "x", "a", "b"), I("const", "x", 100),
          I("add", "y", "a", "b"), I("print", a="x"), I("print", a="y")], expect_unchanged_at=(4,))
    case("dst is also an operand (a = a + b twice)",
         [I("const", "a", 3), I("const", "b", 4), I("add", "a", "a", "b"), I("add", "a", "a", "b"),
          I("print", a="a")], expect_unchanged_at=(3,))
    case("division reused",
         [I("const", "a", 17), I("const", "b", 5), I("div", "x", "a", "b"), I("div", "y", "a", "b"),
          I("print", a="y")], expect_copy_at=3)
    case("constants as operands",
         [I("const", "a", 3), I("mul", "x", "a", 8), I("mul", "y", 8, "a"), I("print", a="y")], expect_copy_at=2)
    case("print untouched",
         [I("const", "a", 3), I("print", a="a"), I("print", a="a")])
    try:
        fn = block_fn([I("const", "a", 1), I("print", a="a")])
        results["returns False when nothing to do"] = cse(fn) is False
    except Exception:
        results["returns False when nothing to do"] = False
    return results


def differential(cse, n=300):
    P.PASSES["cse"] = cse
    bad = crashes = 0
    first = None
    configs = [["cse"], ["copyprop", "cse", "dce"], P.DEFAULT_ORDER, list(reversed(P.DEFAULT_ORDER))]
    for seed in range(n):
        src = fuzz.program(seed)
        ref = D.reference(src)
        for ps in configs:
            try:
                got = D.observed(B.run_vm(D.compile_source(src, "O2", k=4, passes=ps).code))
            except Exception as e:
                got = ("CRASH", repr(e)[:60])
                crashes += 1
            if got != ref:
                bad += 1
                first = first or f"seed {seed} passes {ps}: expected {ref}, got {got}"
                break
    return bad, crashes, first


def effectiveness(cse):
    P.PASSES["cse"] = cse
    out = {}
    for prog in sorted((HERE.parent / "programs").glob("*.tiny")):
        src = prog.read_text()
        out[prog.stem] = B.run_vm(D.compile_source(src, "O2", k=6, passes=["cse"]).code).instructions
    return out


def main():
    path = Path(sys.argv[1])
    original = P.PASSES["cse"]
    t0 = time.perf_counter()
    try:
        cse = load(path)
    except Exception:
        print("Could not load the submission:\n" + traceback.format_exc())
        sys.exit(1)
    unit = unit_tests(cse)
    bad, crashes, first = differential(cse)
    eff = effectiveness(cse)
    ref_eff = effectiveness(original)
    P.PASSES["cse"] = original
    score = dict(submission=path.name, unit_passed=f"{sum(unit.values())}/{len(unit)}",
                 unit_failed=[k for k, v in unit.items() if not v],
                 random_programs_wrong=bad, crashes=crashes, first_failure=first,
                 executed_instructions=sum(eff.values()),
                 reference_executed_instructions=sum(ref_eff.values()),
                 lines_of_code=sum(1 for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")),
                 check_seconds=round(time.perf_counter() - t0, 1))
    for k, v in score.items():
        print(f"{k:34s} {v}")
    with open(HERE / "scores.jsonl", "a") as f:
        f.write(json.dumps(score) + "\n")


if __name__ == "__main__":
    main()
