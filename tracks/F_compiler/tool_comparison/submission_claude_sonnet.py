from dataclasses import dataclass, field

COMMUTATIVE = {"add", "mul", "eq", "ne"}
BINARY = {"add", "sub", "mul", "div", "mod", "shl", "lt", "le", "gt", "ge", "eq", "ne"}


def cse(fn: Function) -> bool:
    changed = False

    for block in fn.blocks.values():
        counter = [0]
        var_vn = {}        # variable -> value number currently held
        holders = {}       # value number -> set of variables holding it
        const_vn = {}      # integer constant -> value number
        table = {}         # expression key -> value number

        def fresh():
            counter[0] += 1
            return counter[0]

        def vn_of(x):
            if isinstance(x, str):
                v = var_vn.get(x)
                if v is None:
                    v = fresh()
                    var_vn[x] = v
                    holders.setdefault(v, set()).add(x)
                return v
            v = const_vn.get(x)
            if v is None:
                v = fresh()
                const_vn[x] = v
            return v

        def assign(var, v):
            old = var_vn.get(var)
            if old is not None:
                holders[old].discard(var)
            var_vn[var] = v
            holders.setdefault(v, set()).add(var)

        for ins in block.instrs:
            op = ins.op
            if op == "print":
                continue
            if op == "const":
                assign(ins.dst, vn_of(ins.a))
                continue
            if op == "copy":
                assign(ins.dst, vn_of(ins.a))
                continue
            if op == "neg":
                key = ("neg", vn_of(ins.a))
            elif op in BINARY:
                va, vb = vn_of(ins.a), vn_of(ins.b)
                if op in COMMUTATIVE and va > vb:
                    va, vb = vb, va
                key = (op, va, vb)
            else:
                # unknown instruction: be conservative
                if ins.dst is not None:
                    assign(ins.dst, fresh())
                continue

            dst = ins.dst
            v = table.get(key)
            if v is not None:
                hs = holders.get(v)
                if hs:
                    h = dst if dst in hs else min(hs)
                    ins.op = "copy"
                    ins.a = h
                    ins.b = None
                    changed = True
                    assign(dst, v)
                    continue
                # value no longer held anywhere: recompute, same value number
                assign(dst, v)
                continue
            v = fresh()
            table[key] = v
            assign(dst, v)

    return changed
