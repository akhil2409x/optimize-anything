"""Mutation testing: inject known optimizer bugs and confirm the test suite catches each.
Run from tracks/F_compiler:  python tests/mutation_check.py"""
import os
import subprocess
import sys

MUTANTS = {
    "dce_deletes_trapping_division":
        "import tinyc.ir as I; I.Instr.has_side_effect = lambda self: self.op == 'print'",
    "cse_without_invalidation":
        "import tinyc.passes as P, tinyc.ir as I\n"
        "def bad(fn):\n"
        "    did = False\n"
        "    for b in fn.blocks.values():\n"
        "        t = {}\n"
        "        for i in b.instrs:\n"
        "            if i.op in I.OPFUNC:\n"
        "                k = P._canon(i.op, i.a, i.b)\n"
        "                if k in t and t[k] != i.dst:\n"
        "                    i.op, i.a, i.b = 'copy', t[k], None; did = True\n"
        "                elif i.dst not in (k[1], k[2]): t[k] = i.dst\n"
        "    return did\n"
        "P.cse = bad; P.PASSES['cse'] = bad",
    "div_by_power_of_two_as_shift":
        "import tinyc.passes as P, tinyc.ir as I, tinyc.backend as B\n"
        "I.OPFUNC['shl'] = B.VMOPS['shl'] = lambda a, b: a << b if b >= 0 else a >> -b\n"
        "orig = P.strength\n"
        "def bad(fn):\n"
        "    d = orig(fn)\n"
        "    for b in fn.blocks.values():\n"
        "        for i in b.instrs:\n"
        "            if i.op == 'div' and P._pow2(i.b): i.op, i.b = 'shl', -(i.b.bit_length() - 1); d = True\n"
        "    return d\n"
        "P.strength = bad; P.PASSES['strength'] = bad",
    "licm_hoists_trapping_division":
        "import tinyc.passes as P\n"
        "src = __import__('inspect').getsource(P.licm).replace("
        "'(i.op in (\"div\", \"mod\") and not i.may_trap())', '(i.op in (\"div\", \"mod\"))')\n"
        "exec(compile(src, 'licm', 'exec'), P.__dict__); P.PASSES['licm'] = P.licm",
    # EQUIVALENT mutant (expected to survive): Tiny loops only exit from their header,
    # so anything live at an exit is live at the header, which the header check already
    # blocks. The exit check would matter if `break` existed.
    "licm_ignores_live_at_exit (equivalent)":
        "import tinyc.passes as P\n"
        "src = __import__('inspect').getsource(P.licm).replace("
        "'if any(i.dst in live_in[e] for e in exits):', 'if False:')\n"
        "exec(compile(src, 'licm', 'exec'), P.__dict__); P.PASSES['licm'] = P.licm",
    "constprop_treats_unassigned_as_constant":
        "import tinyc.passes as P\n"
        "orig = P._meet\n"
        "def bad(states):\n"
        "    states = [s for s in states if s is not None]\n"
        "    if not states: return None\n"
        "    out = {}\n"
        "    for k in set().union(*states):\n"
        "        vals = {s[k] for s in states if k in s}\n"
        "        out[k] = vals.pop() if len(vals) == 1 else P.NAC\n"
        "    return out\n"
        "P._meet = bad",
}

conftest = os.path.join(os.path.dirname(__file__), "conftest.py")
results = {}
for name, code in MUTANTS.items():
    with open(conftest, "w") as f:
        f.write(code + "\n")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests/test_tinyc.py"],
                       capture_output=True, text=True)
    failed = [l.split("::")[1].split(" ")[0] for l in r.stdout.splitlines() if l.startswith("FAILED")]
    results[name] = failed
    print(f"{name:42s} caught by {len(failed):2d} tests: {', '.join(sorted(set(f.split('[')[0] for f in failed)))}")
os.remove(conftest)
sys.exit(0 if all(v for k, v in results.items() if "equivalent" not in k) else 1)
