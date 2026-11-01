"""Correctness checks for Track F. Run from tracks/F_compiler: `pytest -q`.

Three layers:
  1. unit tests of the front end, each pass and the register allocators;
  2. targeted programs for known optimizer hazards (each one kills a
     deliberately injected bug, see README "mutation testing");
  3. differential testing: random programs must behave identically (same
     output, same runtime error) at every optimization level, with every pass
     alone and in reversed order, and with 2..6 registers.
"""
import random

import pytest

from tinyc import backend as B, driver as D, frontend as F, fuzz, ir, passes as P
from tinyc.ir import Instr


def behaviour(src, level="O2", k=4, passes=None, ra=None):
    c = D.compile_source(src, level, k=k, passes=passes, ra=ra)
    return D.observed(B.run_vm(c.code))


ALL_CONFIGS = ([("O0", None), ("O1", None), ("O2", None), ("O3", None)]
               + [("O2", [p]) for p in P.PASSES]
               + [("O2", list(reversed(P.DEFAULT_ORDER)))])


def same_everywhere(src, ks=(2, 3, 6)):
    ref = D.reference(src)
    for lvl, ps in ALL_CONFIGS:
        for k in ks:
            got = behaviour(src, lvl, k, ps)
            assert got == ref, f"{lvl} {ps} k={k}: {got} != {ref}"
    return ref


# ------------------------------------------------------------------ front end
def test_c_division_semantics():
    assert F.interpret(F.parse("print -7 / 2; print -7 % 2; print 7 / -2; print 7 % -2;")) == [-3, -1, -3, 1]


def test_parse_errors_and_runtime_errors():
    with pytest.raises(F.TinyError):
        F.parse("x = ;")
    with pytest.raises(F.TinyError):
        F.parse("while (1) { x = 1;")
    with pytest.raises(F.TinyRuntimeError):
        F.interpret(F.parse("print y;"))
    assert D.reference("print 1; print 1 / 0; print 2;") == ([1], "division by zero")


def test_operator_precedence():
    assert F.interpret(F.parse("print 2 + 3 * 4 - 10 / 5; print (2 + 3) * 4; print - 2 * 3;")) == [12, 20, -6]


# ------------------------------------------------------------------ passes (IR level)
def ir_of(src):
    return ir.lower(F.parse(src))


def test_constprop_folds_and_propagates_across_blocks():
    fn = ir_of("x = 4; if (x > 2) { y = x * 10; } else { y = 1; } print y;")
    P.optimize(fn, ["constprop", "dce"])
    assert "print 40" in str(fn) and "cbr" not in str(fn)


def test_constprop_never_folds_division_by_zero():
    fn = ir_of("x = 0; print 5 / x;")
    P.optimize(fn)
    assert any(i.op == "div" for b in fn.blocks.values() for i in b.instrs)
    assert ir.run_ir.__name__ and D.reference("x = 0; print 5 / x;")[1] == "division by zero"


def test_constprop_unassigned_on_one_path_is_not_a_constant():
    # y is 5 on one path and unassigned on the other: must NOT become `print 5`
    src = "c = 0; if (c) { y = 5; } else { z = 1; } print y;"
    assert D.reference(src)[1] == "other"
    same_everywhere(src)


def test_cse_invalidates_when_operand_redefined():
    fn = ir.Function({"entry": ir.Block("entry", [
        Instr("add", "x", "a", "b"), Instr("mul", "a", "a", 3), Instr("add", "y", "a", "b"),
        Instr("print", a="x"), Instr("print", a="y")], Instr("halt"))})
    P.cse(fn)
    assert fn.blocks["entry"].instrs[2].op == "add"            # not replaced by a copy of x


def test_cse_reuses_and_is_commutative():
    fn = ir.Function({"entry": ir.Block("entry", [
        Instr("add", "x", "a", "b"), Instr("add", "y", "b", "a"), Instr("print", a="y")], Instr("halt"))})
    assert P.cse(fn) and fn.blocks["entry"].instrs[1].op == "copy"


def test_strength_reduction_rules():
    fn = ir.Function({"entry": ir.Block("entry", [
        Instr("mul", "p", "x", 8), Instr("mul", "q", 16, "x"), Instr("mul", "r", "x", 1),
        Instr("add", "s", "x", 0), Instr("div", "t", "x", 4), Instr("mul", "u", "x", -1)], Instr("halt"))})
    P.strength(fn)
    ops = [(i.op, i.b) for i in fn.blocks["entry"].instrs]
    assert ops[0] == ("shl", 3) and ops[1] == ("shl", 4) and ops[2][0] == "copy"
    assert ops[3][0] == "copy" and ops[4] == ("div", 4) and ops[5][0] == "neg"   # division untouched


def test_division_by_power_of_two_of_negative_numbers():
    # the classic wrong rewrite x / 4 -> x >> 2 gives -2 here instead of -1
    same_everywhere("x = 0 - 7; i = 0; while (i < 3) { print x / 4; print x / 2; x = x - 2; i = i + 1; }")


def test_dce_keeps_print_and_trapping_division():
    src = "x = 1; y = 0; z = x / y; w = 3 + 4; print x;"
    assert D.reference(src) == ([], "division by zero")
    same_everywhere(src)
    fn = ir_of("a = 5; b = a * 2; print a;")
    P.optimize(fn)
    assert "mul" not in str(fn) and "shl" not in str(fn)       # dead pure computation removed


def test_licm_hoists_invariant_out_of_inner_loop():
    src = ("n = 6; s = 0; i = 0; while (i < n) { j = 0; "
           "while (j < n) { t = i * i + 3; s = s + t * j; j = j + 1; } i = i + 1; } print s;")
    def i_squared_in_innermost_loop(fn):
        _, inner = P.natural_loops(fn)[0]                     # smallest loop = the j loop
        return any(ins.op == "mul" and ins.a == "i" and ins.b == "i"
                   for l in inner for ins in fn.blocks[l].instrs)
    before = ir_of(src)
    assert i_squared_in_innermost_loop(before)
    after = ir_of(src)
    P.optimize(after, ["copyprop", "licm", "dce"])
    assert not i_squared_in_innermost_loop(after)              # i*i now computed once per i
    assert any(ins.op == "mul" and ins.a == "i" for b in after.blocks.values() for ins in b.instrs)
    same_everywhere(src)


def test_licm_respects_zero_trip_loops_and_values_live_after_loop():
    # t is assigned in a loop that runs 0 times and is read after it: hoisting would define it
    same_everywhere("t = 1; n = 0; i = 0; while (i < n) { t = 7 * 3; i = i + 1; } print t;")
    # a trapping division inside a loop that never runs must not be hoisted
    same_everywhere("z = 0; i = 0; while (i < 0) { q = 5 / z; print q; i = i + 1; } print 1;")


# ------------------------------------------------------------------ register allocation
SPILLY = """a = 1; b = 2; c = 3; d = 4; e = 5; f = 6; i = 0;
while (i < 20) { a = a + b; b = b + c; c = c + d; d = d + e; e = e + f; f = f + a; i = i + 1; }
print a; print b; print c; print d; print e; print f;"""


@pytest.mark.parametrize("k", [2, 3, 4])
def test_ilp_never_worse_than_greedy_and_both_valid(k):
    fn = P.optimize(ir_of(SPILLY))
    g = B.allocate(fn, k, "greedy")
    o = B.allocate(fn, k, "ilp")
    B.check_allocation(fn, g)
    B.check_allocation(fn, o)
    assert o.status == "Optimal" and o.spill_cost <= g.spill_cost + 1e-9


def test_enough_registers_means_no_spills():
    fn = P.optimize(ir_of(SPILLY))
    assert B.allocate(fn, 16, "greedy").spilled == 0


def test_check_allocation_rejects_conflicts():
    fn = P.optimize(ir_of(SPILLY))
    a = B.allocate(fn, 8, "greedy")
    nodes, edges = B.interference(fn)
    u, v = next(iter(edges))
    a.loc[u] = a.loc[v] = "r0"
    with pytest.raises(AssertionError):
        B.check_allocation(fn, a)


# ------------------------------------------------------------------ benchmark programs
@pytest.mark.parametrize("path", sorted(D.__file__ and __import__("pathlib").Path(__file__).parents[1].joinpath("programs").glob("*.tiny")))
def test_benchmark_programs_agree(path):
    same_everywhere(path.read_text(), ks=(3, 6))


# ------------------------------------------------------------------ differential fuzzing
@pytest.mark.parametrize("chunk", range(6))
def test_random_programs(chunk):
    for seed in range(chunk * 50, chunk * 50 + 50):
        src = fuzz.program(seed)
        ref = D.reference(src)
        k = random.Random(seed).randint(2, 6)
        for lvl, ps in ALL_CONFIGS:
            assert behaviour(src, lvl, k, ps) == ref, f"seed {seed} {lvl} {ps} k={k}\n{src}"
