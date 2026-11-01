COMMUTATIVE = frozenset({"add", "mul", "eq", "ne"})
BINARY_OPS = frozenset({"add", "sub", "mul", "div", "mod", "shl",
                        "lt", "le", "gt", "ge", "eq", "ne"})


def cse(fn: Function) -> bool:
    """Eliminate repeated computations inside each block. Replace a repeated
    computation with `copy dst, <earlier result>`. Mutate fn in place and return
    True if anything changed."""
    changed = False
    for block in fn.blocks.values():
        if _cse_block(block):
            changed = True
    return changed


def _cse_block(block: Block) -> bool:
    # Local value numbering over one straight-line block.
    # Every value gets a number; variables record which number they hold.
    # A repeated expression whose earlier result is still held by some
    # variable is replaced by a copy from that variable.
    changed = False
    counter = 0
    var_vn = {}    # variable -> value number it currently holds
    holders = {}   # value number -> {variable: None} for variables holding it now
    table = {}     # expression key -> value number

    def fresh():
        nonlocal counter
        counter += 1
        return counter

    def set_var(name, v):
        old = var_vn.get(name)
        if old == v:
            return
        if old is not None:
            holders[old].pop(name, None)
        var_vn[name] = v
        holders.setdefault(v, {})[name] = None

    def operand_vn(x):
        if isinstance(x, str):
            v = var_vn.get(x)
            if v is None:
                # Initial (unknown) value of this variable within the block.
                v = fresh()
                set_var(x, v)
            return v
        key = ("const", x)
        v = table.get(key)
        if v is None:
            v = fresh()
            table[key] = v
        return v

    for ins in block.instrs:
        op = ins.op
        if op == "print":
            continue
        if op == "copy":
            set_var(ins.dst, operand_vn(ins.a))
            continue
        if op == "const":
            key = ("const", ins.a)
        elif op == "neg":
            key = ("neg", operand_vn(ins.a))
        elif op in BINARY_OPS:
            va = operand_vn(ins.a)
            vb = operand_vn(ins.b)
            if op in COMMUTATIVE and va > vb:
                va, vb = vb, va
            key = (op, va, vb)
        else:
            # Unknown instruction: its result is unknown.
            if ins.dst is not None:
                set_var(ins.dst, fresh())
            continue

        dst = ins.dst
        v = table.get(key)
        if v is None:
            v = fresh()
            table[key] = v
            set_var(dst, v)
            continue

        if var_vn.get(dst) == v:
            # dst already holds this value; the instruction is a no-op.
            continue

        holder = None
        for h in holders.get(v, ()):
            if var_vn.get(h) == v:
                holder = h
                break

        if holder is not None:
            ins.op = "copy"
            ins.a = holder
            ins.b = None
            changed = True
        set_var(dst, v)

    return changed
