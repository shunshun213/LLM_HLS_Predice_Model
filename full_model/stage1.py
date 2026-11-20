"""Stage 1: design (parsed pragmas + clock) -> predicted hardware structure (HS), in the exact
format Stage 2 reads (the sanitized HS of sanitize.py). Every rule is written out below and
numbered S* (structure), O* (operators), P* (II / depth), M* (memories), A* (access pattern),
X* (aux cores)."""
import math

N = 64


def ku(cfg):
    """unroll factor of loop_k (64 if full)."""
    u = cfg['unroll_k']
    return N if u == 'full' else int(u or 1)


def ju(cfg):
    u = cfg['unroll_j']
    return N if u == 'full' else int(u or 1)


# ============================================================== S: structure
def structure(cfg):
    """Returns dict with modules (list of names), loops (list of loop dicts without II/depth),
    stages, top_loop_bodies, flattened flags."""
    pl = cfg['pipeline_loop']
    uk, uj = ku(cfg), ju(cfg)
    off = cfg['flatten'] == 'off'
    loops, mods, stages, bodies = [], [], [], []

    # ---- S4 copy_in: flattened unless a whole-row word (complete reshape along dim 2) of a
    # single-port array must be assembled by read-modify-write (the inner loop then carries a
    # dependence on the same word and cannot be merged with the outer loop)
    rowword_1p = any(cfg[a + '_mode'] == 'reshape' and cfg[a + '_type'] == 'complete' and cfg[a + '_dim'] == 2
                     and cfg[a + '_storage'] == 'ram_1p' for a in ('A', 'B'))
    if rowword_1p:
        mods.append('gemm_Pipeline_copy_in_j')
        loops.append(dict(module='gemm_Pipeline_copy_in_j', loop='copy_in_j', flattened_from=None, level=0,
                          trip_count=N, pipelined=True))
        loops.append(dict(module='gemm', loop='copy_in', flattened_from=None, level=0, trip_count=N, pipelined=False))
        stages.append(dict(stage='copy_in', kind='top_loop', phase='copy_in', trip_count=N))
        bodies.append(dict(loop='copy_in', trip_count=N, children=['gemm_Pipeline_copy_in_j']))
    else:
        mods.append('gemm_Pipeline_copy_in_copy_in_j')
        loops.append(dict(module='gemm_Pipeline_copy_in_copy_in_j', loop='copy_in_copy_in_j',
                          flattened_from=['copy_in', 'copy_in_j'], level=0, trip_count=N * N, pipelined=True))
        stages.append(dict(stage='gemm_Pipeline_copy_in_copy_in_j', kind='call', phase='copy_in'))

    # ---- compute
    C_part = cfg['C_mode'] == 'partition'
    C_rowword = cfg['C_mode'] == 'reshape' and cfg['C_type'] == 'complete' and cfg['C_dim'] == 2
    A_complete = cfg['A_mode'] in ('partition', 'reshape') and cfg['A_type'] == 'complete'

    def top_ij(children_of_j, j_trip):
        """top loops i (and j) around children; flatten i,j unless off or C row-word (S3)."""
        if off or C_rowword:
            loops.append(dict(module='gemm', loop='loop_i', flattened_from=None, level=0, trip_count=N, pipelined=False))
            loops.append(dict(module='gemm', loop='loop_j', flattened_from=None, level=1, trip_count=j_trip, pipelined=False))
            stages.append(dict(stage='loop_i', kind='top_loop', phase='compute', trip_count=N))
            bodies.append(dict(loop='loop_i', trip_count=N, children=['loop_j']))
            bodies.append(dict(loop='loop_j', trip_count=j_trip, children=children_of_j))
        else:
            loops.append(dict(module='gemm', loop='loop_i_loop_j', flattened_from=['loop_i', 'loop_j'], level=0,
                              trip_count=N * j_trip, pipelined=False))
            stages.append(dict(stage='loop_i_loop_j', kind='top_loop', phase='compute', trip_count=N * j_trip))
            bodies.append(dict(loop='loop_i_loop_j', trip_count=N * j_trip, children=children_of_j))

    if pl == 'k':
        tck = -(-N // uk)
        if uj == 1 and not off and not C_part:
            # S2: perfect nest -> i, j, k flattened into one pipeline
            m = 'gemm_Pipeline_loop_i_loop_j_loop_k'
            mods.append(m)
            loops.append(dict(module=m, loop='loop_i_loop_j_loop_k', flattened_from=['loop_i', 'loop_j', 'loop_k'],
                              level=0, trip_count=N * N * tck, pipelined=True))
            stages.append(dict(stage=m, kind='call', phase='compute'))
        else:
            # k pipeline modules, one per j copy (unroll_j), called from the top i/j loops
            ks = ['gemm_Pipeline_loop_k'] + [f'gemm_Pipeline_loop_k{t}' for t in range(1, uj)]
            for m in ks:
                mods.append(m)
                loops.append(dict(module=m, loop='loop_k', flattened_from=None, level=0, trip_count=tck, pipelined=True))
            top_ij(ks, -(-N // uj))
    elif pl == 'i':
        # S6 [extrapolated: no design in the data]: PIPELINE on loop_i fully unrolls loop_j and
        # loop_k; one pipeline over i (TC 64) called once from the top
        m = 'gemm_Pipeline_loop_i'
        mods.append(m)
        loops.append(dict(module=m, loop='loop_i', flattened_from=None, level=0, trip_count=N, pipelined=True))
        stages.append(dict(stage=m, kind='call', phase='compute'))
    elif pl == 'j':
        tcj = -(-N // uj)
        if A_complete and uj == 1 and not off:
            # S1: A completely partitioned / reshaped (either dim): every A element sits in its own
            # bank or lane, so no per-i row hoist into registers is needed -> i, j flattened
            # (13/13 designs; the reason is the absence of the hoist, not the dimension)
            m = 'gemm_Pipeline_loop_i_loop_j'
            mods.append(m)
            loops.append(dict(module=m, loop='loop_i_loop_j', flattened_from=['loop_i', 'loop_j'], level=0,
                              trip_count=N * tcj, pipelined=True))
            stages.append(dict(stage=m, kind='call', phase='compute'))
        else:
            m = 'gemm_Pipeline_loop_j'
            mods.append(m)
            loops.append(dict(module=m, loop='loop_j', flattened_from=None, level=0, trip_count=tcj, pipelined=True))
            loops.append(dict(module='gemm', loop='loop_i', flattened_from=None, level=0, trip_count=N, pipelined=False))
            stages.append(dict(stage='loop_i', kind='top_loop', phase='compute', trip_count=N))
            bodies.append(dict(loop='loop_i', trip_count=N, children=[m]))
    else:
        # no pipeline: all loops stay in the top FSM; k fully unrolled -> no loop_k
        tcj = -(-N // uj)
        loops.append(dict(module='gemm', loop='loop_i', flattened_from=None, level=0, trip_count=N, pipelined=False))
        loops.append(dict(module='gemm', loop='loop_j', flattened_from=None, level=1, trip_count=tcj, pipelined=False))
        stages.append(dict(stage='loop_i', kind='top_loop', phase='compute', trip_count=N))
        bodies.append(dict(loop='loop_i', trip_count=N, children=['loop_j']))
        if uk < N:
            kids = []
            for t in range(uj):
                loops.append(dict(module='gemm', loop='loop_k', flattened_from=None, level=2, trip_count=-(-N // uk),
                                  pipelined=False))
                kids.append('loop_k')
                bodies.append(dict(loop='loop_k', trip_count=-(-N // uk), children=[]))
            bodies.append(dict(loop='loop_j', trip_count=tcj, children=kids))
        else:
            bodies.append(dict(loop='loop_j', trip_count=tcj, children=[]))

    # ---- copy_out: always one flattened pipeline
    mods.append('gemm_Pipeline_copy_out_copy_out_j')
    loops.append(dict(module='gemm_Pipeline_copy_out_copy_out_j', loop='copy_out_copy_out_j',
                      flattened_from=['copy_out', 'copy_out_j'], level=0, trip_count=N * N, pipelined=True))
    stages.append(dict(stage='gemm_Pipeline_copy_out_copy_out_j', kind='call', phase='copy_out'))
    mods.append('gemm')
    return dict(modules=mods, loops=loops, stages=stages, bodies=bodies)


# ============================================================== O: operator latency / impl
# O2: default scheduled latency of each core by (op, impl, clock)   (HS_dataset finding 2)
DEFAULT_LAT = {
    ('fadd', 'fabric'): {10.0: 3, 5.0: 5, 3.33: 12, 2.5: 13},
    ('fadd', 'fulldsp'): {10.0: 4, 5.0: 9, 3.33: 12, 2.5: 12},
    ('fmul', 'fabric'): {10.0: 3, 5.0: 5, 3.33: 9, 2.5: 9},
    ('fmul', 'meddsp'): {10.0: 3, 5.0: 6, 3.33: 9, 2.5: 9},
    ('fmul', 'fulldsp'): {10.0: 3, 5.0: 6, 3.33: 7, 2.5: 9},
    ('fmul', 'maxdsp'): {10.0: 3, 5.0: 5, 3.33: 7, 2.5: 7},
}


class UnsupportedDesign(ValueError):
    """A design outside the modeled space (raised with a message naming the construct)."""


def clock(cfg):
    """Domain: the four characterized clocks. DEFAULT_LAT and the depth offsets are measured
    tool characteristics at these clocks only; other clocks are rejected, not snapped."""
    T = float(cfg['clock_ns'])
    c = min((10.0, 5.0, 3.33, 2.5), key=lambda c: abs(c - T))
    if abs(c - T) > 0.05:
        raise UnsupportedDesign(f'clock {T} ns outside the characterized set 10 / 5 / 3.33 / 2.5 ns')
    return c


def default_impl(cfg, op, recurrence):
    """O1: impl the tool picks when bind_op gives none.
    fmul: maxdsp, except fulldsp at 2.5 ns.
    fadd on the accumulation recurrence (loop_k pipelined): the impl with the smaller default
      latency (ties -> fabric): fabric at 10/5/3.33 ns, fulldsp at 2.5 ns.
    fadd off the recurrence (loop_j / no pipeline / unsafe-math tree): fulldsp at >= 5 ns,
      fabric below."""
    T = clock(cfg)
    if op == 'fmul':
        return 'fulldsp' if T == 2.5 else 'maxdsp'
    if recurrence:
        a, b = DEFAULT_LAT[('fadd', 'fabric')][T], DEFAULT_LAT[('fadd', 'fulldsp')][T]
        # the recurrence only constrains the choice if some impl would stretch II beyond the
        # target; if II_target >= both latencies the off-recurrence default applies
        f = 1 if reassociates(cfg) else ku(cfg)      # II_rec = L (reassociated) or u*L
        if (cfg.get('pipeline_ii') or 1) < f * max(a, b):
            return 'fabric' if a <= b else 'fulldsp'
    return 'fulldsp' if T >= 5.0 else 'fabric'


def op_impl_lat(cfg, op, recurrence):
    impl = cfg[op + '_impl']
    if impl in (None, 'auto'):
        impl = default_impl(cfg, op, recurrence)
    lat = cfg[op + '_latency']
    if lat in (None, 'auto'):
        lat = DEFAULT_LAT[(op, impl)][clock(cfg)]
    return impl, int(lat)


def reassociates(cfg):
    """unsafe_math reassociation of the k-sum happens only when the fadd is not bound
    (HS_dataset finding 4: a bound fadd disables it)."""
    return bool(cfg['unsafe_math']) and cfg['fadd_impl'] in (None, 'auto')


# ============================================================== A: access model
IDX = {'A': ('i', 'k'), 'B': ('k', 'j'), 'C': ('i', 'j')}
# module kinds whose inner loop_k is fully unrolled by the pipeline (j kinds); loop_i: pipeline on
# loop_i also fully unrolls loop_j (R = 64)
JK = ('loop_j', 'loop_i_loop_j', 'loop_i')


def layout(cfg, X):
    """(mode, type, dim, factor) normalized; factor 64 for complete."""
    mode = cfg[X + '_mode']
    if mode == 'none':
        return None
    t = cfg[X + '_type']
    f = N if t == 'complete' else int(cfg[X + '_factor'])
    return mode, t, int(cfg[X + '_dim'] or 1), f


def locate(cfg, X, coord):
    """(bank, word address) of element coord=(d1, d2) under partition / reshape (Vitis
    semantics: cyclic -> bank x%f addr x//f ; block -> bank x//(64/f) addr x%(64/f);
    reshape = partition whose banks are concatenated into one word)."""
    L = layout(cfg, X)
    if L is None:
        return (0,), coord
    mode, t, dim, f = L
    x = coord[dim - 1]
    if t == 'cyclic' or t == 'complete':
        b, a = x % f, x // f
    else:
        bs = N // f
        b, a = x // bs, x % bs
    rest = tuple(c for d, c in enumerate(coord) if d != dim - 1)
    if mode == 'partition':
        return (b,), (a,) + rest
    return (0,), (a,) + rest                          # reshape: one bank, lane b of word a


def banks(cfg, X):
    L = layout(cfg, X)
    return L[3] if (L and L[0] == 'partition') else 1


def module_kind(m):
    return m.replace('gemm_Pipeline_', '').rstrip('0123456789')


def varying(kind):
    """loop indices that change from iteration to iteration inside a pipelined module."""
    return {'loop_i_loop_j_loop_k': {'i', 'j', 'k'}, 'loop_k': {'k'}, 'loop_j': {'j'}, 'loop_i': {'i'},
            'loop_i_loop_j': {'i', 'j'}}[kind]


def accesses(cfg, X, kind):
    """element coordinates of X touched in one pipeline iteration (aligned representative
    iteration: i0 = j0 = k0 = 0)."""
    uk, uj = ku(cfg), ju(cfg)
    u = N if kind in JK else uk
    R = uj if kind in JK + ('none',) else 1
    if X == 'A':
        return [(0, t) for t in range(u)]
    if X == 'B':
        return [(t, r) for t in range(u) for r in range(R)]
    return [(0, r) for r in range(R)]


def hoisted(cfg, X, kind):
    """Rule H: is array X passed to the module as scalar values read by the caller FSM?
    H1 the array's indices do not vary inside the module (A in a loop_j module: i is fixed per
       call and k is fully unrolled) -> the whole row is hoisted;
    H2 the array is completely partitioned / reshaped along an index that varies inside the
       module (dynamic bank/word select) -> all 64 values are hoisted."""
    var = varying(kind)
    own = [ix for ix in IDX[X] if not ((ix == 'k' and kind in JK) or (ix == 'j' and kind == 'loop_i'))]
    if X != 'C' and not (set(own) & var):
        return 'H1'
    L = layout(cfg, X)
    if X != 'C' and L and L[1] == 'complete' and IDX[X][L[2] - 1] in var and IDX[X][L[2] - 1] != 'k' \
            and not (L[0] == 'reshape' and 'k' in var):     # reshaped word still addressed by k
        return 'H2'
    if X != 'C' and kind in ('loop_k', 'loop_i_loop_j_loop_k') and invariant_words(cfg, X, kind):
        return 'H3'
    return None


def invariant_words(cfg, X, kind):
    """Rule H3: in a k-pipelined module, if every bank is read at the same set of word
    addresses in every iteration in which it is read, the words are loop invariant: the caller
    reads them once and passes them in. Returns the number of words per bank, else None.
    Iterations: k0 = 0, uk, 2uk, ... (plus i, j in {0, 1} when they vary inside the module)."""
    uk = ku(cfg)
    ijs = [(0, 0)] if kind == 'loop_k' else [(0, 0), (0, 1), (1, 0), (1, 1)]
    seen = {}
    for i, j in ijs:
        for k0 in range(0, N, uk):
            cs = [(i, (k0 + t) % N) for t in range(uk)] if X == 'A' else [((k0 + t) % N, j) for t in range(uk)]
            per = {}
            for c in cs:
                b, a = locate(cfg, X, c)
                per.setdefault(b, set()).add(a)
            for b, ws in per.items():
                if seen.setdefault(b, ws) != ws:
                    return None
    return max(len(w) for w in seen.values())


def reachable_banks(cfg, X, kind):
    """Banks wired to the module: every bank the access can reach over the module's whole
    iteration space, including the indices fixed per call (i, j0) - a bank selected by any
    run-time index needs its own port connection. Module copy c of an unrolled-j group sees
    j = j0 + c with j0 a multiple of unroll_j (by symmetry the count does not depend on c)."""
    if banks(cfg, X) == 1:
        return 1
    step = ju(cfg) if kind in ('loop_k', 'loop_i_loop_j_loop_k') else 1
    js = range(0, N, step)
    if X == 'A':
        cs = [(i, k) for i in range(N) for k in range(N)]
    else:
        cs = [(k, j) for k in range(N) for j in js]
    return len({locate(cfg, X, c)[0] for c in cs})


def reads_per_bank(cfg, X, kind):
    acc = accesses(cfg, X, kind)
    per = {}
    for c in acc:
        b, a = locate(cfg, X, c)
        per.setdefault(b, set()).add(a)
    nb_touched = reachable_banks(cfg, X, kind)
    return max(len(s) for s in per.values()), nb_touched


def storage_ports(cfg, X):
    s = cfg.get(X + '_storage', 'default') if X != 'C' else 'default'
    return {'ram_1p': 1, 'ram_2p': 2, 'ram_t2p': 2}.get(s)


# ============================================================== P: initiation interval
def fadd_rec(cfg):
    """impl/latency of the fadd on the accumulation recurrence."""
    return op_impl_lat(cfg, 'fadd', recurrence=True)


def port_plan(cfg, X, kind, II_base):
    """M/A: reads per bank, ports per bank, banks touched, RAM_1WNR inferred.
    Explicit storage fixes the ports (1P: 1, 2P/T2P: 2). Default storage: the tool provides
    as many read ports as the schedule needs, ceil(reads / II_base), at least 2 (a BRAM has two
    ports), at most 16; more than 2 reads per bank -> the bank is replicated (RAM_1WNR)."""
    rpb, nb = reads_per_bank(cfg, X, kind)
    sp = storage_ports(cfg, X)
    if sp is None:
        p = min(rpb, max(2, math.ceil(rpb / II_base)), 16)
        wnr = rpb > 2
    else:
        p, wnr = min(sp, rpb), False
    return dict(reads=rpb, ports=p, banks_touched=nb, wnr=wnr, ii=math.ceil(rpb / max(1, p)))


def ii_plan(cfg, m):
    """P1-P3: II_achieved = max(II_target, II_recurrence, II_memory)."""
    kind = module_kind(m)
    tgt = cfg['pipeline_ii'] or 1
    if kind in ('loop_i_loop_j_loop_k', 'loop_k'):
        L_acc = fadd_rec(cfg)[1]
        ii_rec = L_acc if reassociates(cfg) else ku(cfg) * L_acc
    else:
        ii_rec = 1
    base = max(tgt, ii_rec)
    plans = {}
    for X in ('A', 'B'):
        if hoisted(cfg, X, kind):
            continue
        plans[X] = port_plan(cfg, X, kind, base)
    ii_mem = max([p['ii'] for p in plans.values()] + [1])
    ii = max(base, ii_mem)
    if ii == tgt:
        lim = 'target' + (' = recurrence' if ii_rec == tgt else '') + (' = memory' if ii_mem == tgt and tgt > 1 else '')
    else:
        lim = 'recurrence' if ii == ii_rec and ii_rec >= ii_mem else 'memory_port'
    return dict(ii=ii, ii_rec=ii_rec, ii_mem=ii_mem, plans=plans, limiter=lim)


# ============================================================== D: pipeline depth
def depth(cfg, m, ii, plans):
    """D (pipeline depth; offsets measured on the tool, see STAGE1_MODEL.md section D):
      k pipelines, no reassociation: u chained adds + multiply + read/write registers:
        D = u*La + Lm + 2 (+ first-iteration select stages when u = 1, see below);
      k pipelines with reassociation: D = Lm + 2 + (ceil(log2 u)+1)(Lt+1) + fetch,
        fetch = max over arrays of ceil(reads_per_bank / ports);
      j / i-j pipelines (k fully unrolled): D = levels*(La+1) + Lm + 3 (+1 below 10 ns),
        levels = 64 (chain) or 6 (tree, + fetch - 1 in front)."""
    kind = module_kind(m)
    re = reassociates(cfg)
    Lm = op_impl_lat(cfg, 'fmul', recurrence=False)[1]
    if kind in ('loop_i_loop_j_loop_k', 'loop_k'):
        u = ku(cfg)
        La = fadd_rec(cfg)[1]
        if re and u > 1:
            Lt = op_impl_lat(cfg, 'fadd', recurrence=False)[1]
            # operand fetch: the slowest bank needs ceil(reads_per_bank / ports) cycles
            fetch = max([math.ceil(p['reads'] / max(1, p['ports'])) for p in plans.values()] + [1])
            return Lm + 2 + (math.ceil(math.log2(u)) + 1) * (Lt + 1) + fetch
        extra = 0
        if u == 1:
            # single-product body: the accumulator's first-iteration select (acc = 0 vs acc)
            # sits in front of the fadd. It costs a stage when the fadd input is on the critical
            # path (La >= Lm): +2 at T < 10 ns, +1 at 10 ns only with a shallow (<3) fmul;
            # at 2.5 ns a deep (>=7 stage) fmul registers it for free (-1).
            if kind == 'loop_i_loop_j_loop_k':
                extra = 2
                if La >= Lm:
                    if clock(cfg) < 10.0:
                        extra += 2 - (1 if (clock(cfg) == 2.5 and Lm >= 7) else 0)
                    elif Lm < 3:
                        extra += 1
            else:
                extra = 1 + (1 if La > Lm else 0)
        return u * La + Lm + 2 + extra
    La = op_impl_lat(cfg, 'fadd', recurrence=False)[1]
    levels = 6 if re else N
    d = levels * (La + 1) + Lm + 3 + (0 if clock(cfg) == 10.0 else 1)
    if re:
        # a balanced tree needs all its leaves: the operand fetch (ceil(reads/ports) cycles of
        # the slowest bank) is added in front; a serial chain consumes operands in order and
        # overlaps the fetch
        fetch = max([math.ceil(p['reads'] / max(1, p['ports'])) for p in plans.values()] + [1])
        d += fetch - 1
    return d


def copy_depth(cfg, loop):
    """D-copy: depth of the II=1 copy pipelines. Base: read, (register,) write = 2 stages at
    10/5 ns, 4 at 3.33/2.5 ns (address and data registered on both sides).
    copy_in: writing a lane into a reshaped word adds a merge stage at T < 10 ns; at 10 ns only
      the 2048-bit whole-row word (complete reshape on dim 2) needs it.
    copy_out (reads C, bank/lane chosen by j):
      5 ns: +1 for a lane select out of a reshaped word or a 64:1 bank mux (complete partition);
      3.33/2.5 ns: -1 when C is partitioned along j or completely (the bank read is retimed into
        the mux stage), +1 for a lane select on dim 1;
      2.5 ns with a >= 512-bit word: the lane is sliced by the 4-stage shifter: 3 + 4."""
    T = clock(cfg)
    base = 2 if T >= 5.0 else 4
    if loop.startswith('copy_in'):
        res = [layout(cfg, X) for X in ('A', 'B') if layout(cfg, X) and layout(cfg, X)[0] == 'reshape']
        if not res:
            return base
        if T < 10.0:
            return base + 1
        return base + (1 if any(L[1] == 'complete' and L[2] == 2 for L in res) else 0)
    L = layout(cfg, 'C')
    if not L or T == 10.0:
        return base
    mode, t, dim, f = L
    if T == 5.0:
        return base + (1 if (mode == 'reshape' or t == 'complete') else 0)
    if T == 2.5 and mode == 'reshape' and 32 * f >= 512:
        return (base - 1) + SHIFT_LAT                 # fast-clock read path + 4-stage shifter
    if mode == 'partition' and (dim == 2 or t == 'complete'):
        return base - 1
    if mode == 'reshape' and dim == 1:
        return base + 1
    return base


# ============================================================== M: memories
def memories(cfg, wnr):
    out = {}
    for X, a in (('A', 'lA'), ('B', 'lB'), ('C', 'lC')):
        L = layout(cfg, X)
        nb, bits, words = 1, 32, N * N
        if L:
            mode, t, dim, f = L
            if mode == 'partition':
                nb, words = f, N * N // f
            else:
                bits, words = 32 * f, N * N // f
        st = cfg.get(X + '_storage', 'default') if X != 'C' else 'default'
        impl = (cfg.get(X + '_storage_impl') or 'auto') if X != 'C' else 'auto'
        ports = {'default': 'default', 'ram_1p': '1P', 'ram_2p': '2P', 'ram_t2p': 'T2P'}[st]
        impl_s = {'auto': 'AUTO', 'bram': 'BRAM', 'lutram': 'LUTRAM'}[impl]
        if st == 'ram_t2p':
            impl_s = 'BRAM'                        # true dual port exists only in BRAM
        inferred = None
        if st == 'default' and wnr.get(X):
            ports, inferred = '1WNR', ['RAM_1WNR']
        out[a] = dict(owner='gemm', storage_ports=ports, storage_impl=impl_s, banks=nb,
                      words_per_bank=[words] * 1, bits_per_word=[bits], uniform_banks=True,
                      in_registers=False, storage_inferred_by_tool=inferred, total_bits=N * N * 32)
    return out


# ============================================================== X: aux cores (bank select)
def sparsemux_slots(cfg, X, kind, ports=1):
    """X: bank-select multiplexers (sparsemux) for a partitioned array read inside a module.
    A slot is one element read per iteration (A[i][k0+t], B[k0+t][j0+r]). Its bank is a
    function of the loop indices; over every index assignment the module can see (call-fixed
    i/j included, since all banks are wired to the module) it reaches a set of banks. Slots
    whose bank is always the same are read through that bank's ports, so they need
    min(#slots, ports) muxes; each mux has |reach| inputs. A 2-way select is a plain select
    operator, not a sparsemux (needs >= 3 inputs). Returns a list of mux input counts."""
    L = layout(cfg, X)
    if not L or L[0] != 'partition' or L[3] < 3:
        return []
    uk, uj = ku(cfg), ju(cfg)
    jkind = kind in JK
    k0s = [0] if jkind else list(range(0, N, uk))
    js = list(range(0, N, uj if (jkind or kind in ('loop_k', 'none')) else 1))
    is_ = list(range(N))
    if X == 'A':
        assign = [(i, 0, k0) for i in is_ for k0 in k0s]
    else:
        assign = [(0, j, k0) for j in js for k0 in k0s]
    groups = {}
    for c in accesses(cfg, X, kind):
        seq = []
        for (i, j, k0) in assign:
            cc = (i + c[0], k0 + c[1]) if X == 'A' else (k0 + c[0], j + c[1])
            seq.append(locate(cfg, X, (cc[0] % N, cc[1] % N))[0])
        groups.setdefault(tuple(seq), []).append(c)
    out = []
    for seq, members in groups.items():
        reach = len(set(seq))
        if reach >= 3:
            out += [reach] * min(len(members), ports)
    return out


def lane(cfg, X, coord):
    """lane of element coord inside its reshaped word (same index arithmetic as locate)."""
    mode, t, dim, f = layout(cfg, X)
    x = coord[dim - 1]
    return x % f if t in ('cyclic', 'complete') else x // (N // f)


def shift_slots(cfg, X, kind):
    """X-SH: at 2.5 ns a lane select out of a reshaped word of >= 512 bits (>= 16 lanes) does not
    fit one cycle as a mux; the tool slices the word with a pipelined barrel shifter (lshr,
    latency 4). One shifter per read slot whose lane index changes at run time."""
    L = layout(cfg, X)
    if clock(cfg) != 2.5 or not L or L[0] != 'reshape' or 32 * L[3] < 512:
        return 0
    uk, uj = ku(cfg), ju(cfg)
    jkind = kind in JK
    k0s = [0] if jkind else list(range(0, N, uk))
    js = list(range(0, N, uj if (jkind or kind in ('loop_k', 'none')) else 1))
    assign = [(i, 0, k0) for i in range(N) for k0 in k0s] if X == 'A' else \
             [(0, j, k0) for j in js for k0 in k0s]
    n = 0
    for c in accesses(cfg, X, kind):
        lanes = set()
        for (i, j, k0) in assign:
            cc = (i + c[0], k0 + c[1]) if X == 'A' else (k0 + c[0], j + c[1])
            lanes.add(lane(cfg, X, (cc[0] % N, cc[1] % N)))
        n += len(lanes) > 1
    return n


SHIFT_LAT = 4      # lshr core stages at 2.5 ns (measured: every shift core in the data)


def shift_core(owner, X, cfg, n):
    W = 32 * layout(cfg, X)[3]
    return dict(owner=owner, **{'class': 'shift'},
                params={'class': 'shift', 'latency': SHIFT_LAT, 'op': 'lshr',
                        'shamt_bits': int(math.log2(W)), 'width': W},
                count_in_module=n, count_effective=n)


def smx(n):
    """sparsemux core parameters: n 32-bit data inputs, ceil(log2 n) select bits."""
    return {'class': 'sparsemux', 'inputs': n, 'op': 'sparsemux',
            'sel_bits': math.ceil(math.log2(n)), 'width': 32}


def shl_cores(cfg, st, comp):
    """X-SHL: at 2.5 ns, writing one 32-bit lane into a reshaped word of >= 512 bits uses a
    pipelined left shifter (data and byte-enable placed at the lane offset); the report lists it
    as an Expression of 2661 FF (the same register cost as the lshr core). Count per writer:
      copy_in, per reshaped A/B: 2 for a whole-row word (complete reshape on dim 2, 2048 bit:
        two shifter halves), else 1;
      C writer (the compute module that stores C, or gemm when C is stored by the top FSM),
        per lane write of one iteration (R = unroll_j copies; 1 in the flattened ijk nest):
        2 for a whole-row word, else 1.
    [support: 14 modules, 12 exact; all at 2.5 ns, train only]"""
    out = []
    if clock(cfg) != 2.5:
        return out
    def n_of(L):
        return 2 if (L[1] == 'complete' and L[2] == 2) else 1
    def core(owner, n, L):
        return dict(owner=owner, **{'class': 'shift'},
                    params={'class': 'shift', 'latency': SHIFT_LAT, 'op': 'shl',
                            'shamt_bits': int(math.log2(32 * L[3])), 'width': 32 * L[3],
                            'total_bits': n * 32 * L[3]},
                    count_in_module=n, count_effective=n)
    cin = [m for m in st['modules'] if m.startswith('gemm_Pipeline_copy_in')][0]
    n = 0
    for X in ('A', 'B'):
        L = layout(cfg, X)
        if L and L[0] == 'reshape' and 32 * L[3] >= 512:
            out.append(core(cin, n_of(L), L))
    L = layout(cfg, 'C')
    if L and L[0] == 'reshape' and 32 * L[3] >= 512:
        kinds = {module_kind(m) for m in comp}
        if cfg['pipeline_loop'] == 'none' or kinds == {'loop_k'}:
            owner, R = 'gemm', ju(cfg)                  # C stored by the top FSM
        else:
            owner = comp[0]
            R = 1 if kinds == {'loop_i_loop_j_loop_k'} else ju(cfg)
        out.append(core(owner, R * n_of(L), L))
    # one aux entry per owner (count summed, width = widest word), as the extractor lists them
    by = {}
    for a in out:
        if a['owner'] in by:
            b = by[a['owner']]
            n = b['count_in_module'] + a['count_in_module']
            W = max(b['params']['width'], a['params']['width'])
            b.update(count_in_module=n, count_effective=n)
            b['params'].update(width=W, shamt_bits=int(math.log2(W)),
                               total_bits=b['params']['total_bits'] + a['params']['total_bits'])
        else:
            by[a['owner']] = a
    return list(by.values())


def top_sparsemux(cfg):
    """Non-pipelined body (owner gemm): no II pressure, so the k-unrolled reads of one j copy are
    sequenced through one mux; each unrolled j copy that indexes the partitioned dimension
    computes its own bank select. Inputs = banks a slot can reach (same reach rule as above)."""
    out = []
    for X in ('A', 'B'):
        r = sparsemux_slots(cfg, X, 'none', ports=1)
        if r:
            L = layout(cfg, X)
            out += [max(r)] * (ju(cfg) if (X == 'B' and L[2] == 2) else 1)
    return out


def independent_copies(cfg, comp):
    """O2: the unroll_j copies of a loop_k module (loop_k, loop_k1, ...) are scheduled in
    parallel - and then each must own its fp cores - only if they share no memory port:
    A reaches them as hoisted scalars (H1/H3) and each copy reads a disjoint set of B banks
    (B partitioned along j). Otherwise they contend for ports, run one after another and share
    one pool of cores owned by gemm."""
    if 'O2par' in cfg.get('_disable', ()) or not all(module_kind(m) == 'loop_k' for m in comp):
        return False
    if not hoisted(cfg, 'A', 'loop_k'):
        return False
    L = layout(cfg, 'B')
    if not L or L[0] != 'partition':
        return False
    sets = []
    for c in range(ju(cfg)):
        sets.append({locate(cfg, 'B', (k, j0 + c))[0] for k in range(N) for j0 in range(0, N, ju(cfg))})
    return all(not (sets[a] & sets[b]) for a in range(len(sets)) for b in range(a + 1, len(sets)))


# ============================================================== assembly
def normalize(cfg):
    """N0: reduce a pragma set to the canonical form the rules are written for. Each rewrite is
    a documented HLS semantic, recorded in cfg['_norm_notes'] for the trace (parser notes stay
    in cfg['_notes']).
      - an unroll factor u that does not divide 64 keeps u body copies; the loop runs
        ceil(64/u) iterations and the copies of the last one are guarded by exit checks
        [extrapolated: no such design in the data; the guard logic is not costed];
      - PIPELINE on a loop that is itself fully unrolled has no loop left to pipeline and is
        dropped (loop_k with unroll_k = 64; loop_j with unroll_j = 64) [extrapolated];
      - PIPELINE on loop_i / loop_j fully unrolls the loops inside it."""
    c = dict(cfg)
    notes = []
    for key in ('unroll_k', 'unroll_j'):
        u = c.get(key)
        if u not in (None, 'full') and int(u) >= N:
            c[key] = 'full'
        elif u not in (None, 'full') and N % int(u):
            notes.append(f'{key}={u} does not divide 64: {u} body copies, trip count ceil(64/{u}) = '
                         f'{-(-N // int(u))} with a guarded last iteration')
    if c['pipeline_loop'] == 'k' and ku(c) == N:
        notes.append('PIPELINE on loop_k with loop_k fully unrolled: no loop to pipeline, dropped')
        c['pipeline_loop'] = 'none'
    if c['pipeline_loop'] in ('j', 'i'):
        c['unroll_k'] = 'full'
    if c['pipeline_loop'] == 'i':
        c['unroll_j'] = 'full'
    if c['pipeline_loop'] == 'j' and ju(c) == N:
        notes.append('PIPELINE on loop_j with loop_j fully unrolled: no loop to pipeline, dropped')
        c['pipeline_loop'] = 'none'
    c['_norm_notes'] = notes
    return c


def not_pipelined(cfg, loop):
    """S5 [low confidence, 1 supporting design, 0 contradicting in 510]: the tool drops a
    requested flushable (flp) pipeline whose iterations would barely overlap: the planned II
    is so close to the iteration latency that pipelining saves < 2 % of a 2-iteration loop
    while adding the flush control. The loop is then a sequential loop in the same module,
    with one fmul and one fadd reused across the unrolled body."""
    if 'S5' in cfg.get('_disable', ()) or cfg['pipeline_style'] != 'flp' or not loop['pipelined'] \
            or 'copy' in loop['loop']:
        return False
    d, ii = loop['pipeline_depth'], loop['ii_achieved']
    return loop['trip_count'] <= 2 and (d - ii) < 0.02 * d


def predict_hs(cfg):
    cfg = normalize(cfg)
    st = structure(cfg)
    comp = [m for m in st['modules'] if m != 'gemm' and 'copy' not in m]
    re = reassociates(cfg)
    loops, ops, aux, ap_all, wnr = [], [], [], {}, {}
    plans_by = {}
    for l in st['loops']:
        l = dict(l)
        l.update(module_instantiations=1, pipeline_style=None, ii_target=None, ii_achieved=None,
                 pipeline_depth=None)
        if l['pipelined'] and 'copy' in l['loop']:
            l.update(ii_target=1, ii_achieved=1, pipeline_depth=copy_depth(cfg, l['loop']))
        elif l['pipelined']:
            p = ii_plan(cfg, l['module'])
            plans_by[l['module']] = p
            l.update(pipeline_style='flp' if cfg['pipeline_style'] == 'flp' else None,  # tool reports only flp
                     ii_target=cfg['pipeline_ii'] or 1,
                     ii_achieved=p['ii'], pipeline_depth=depth(cfg, l['module'], p['ii'], p['plans']))
            for X, pl in p['plans'].items():
                wnr[X] = wnr.get(X, False) or pl['wnr']
        loops.append(l)

    # ---- O: operators
    shared_top = (len(comp) > 1 and not independent_copies(cfg, comp)) or cfg['pipeline_loop'] == 'none'
    def add(owner, op, impl, lat, n):
        if n > 0:
            ops.append(dict(owner=owner, op=op, impl=impl, latency=lat, count_in_module=n,
                            owner_instantiations=1, count_effective=n))
    if cfg['pipeline_loop'] == 'none':
        u, R = ku(cfg), ju(cfg)
        im, lm = op_impl_lat(cfg, 'fmul', False)
        ia, la = op_impl_lat(cfg, 'fadd', False)
        # B-none (sequential body, operators shared across cycles by the binder). A second
        # multiplier is kept when two products are needed in the same cycle:
        #  - u = 64: the full row of products feeds the chain back to back;
        #  - unsafe-math reassociation: the adder tree consumes products in pairs;
        #  - A split along k (partition/reshape on dim 2) with u >= 4: two A operands arrive per
        #    cycle from separate banks/lanes, alongside the two B ports.
        La_split = layout(cfg, 'A') is not None and layout(cfg, 'A')[2] == 2
        two = u == N or (re and u >= 2) or (La_split and u >= 4)
        add('gemm', 'fmul', im, lm, 2 if two else 1)
        # one adder per serial accumulation chain; with the whole row unrolled and >= 2 j copies,
        # two chains are interleaved
        add('gemm', 'fadd', ia, la, 2 if (u == N and R >= 2) else 1)
    else:
        m0 = comp[0]
        kind = module_kind(m0)
        p = plans_by[m0]
        II = p['ii']
        u = ku(cfg) if kind in ('loop_k', 'loop_i_loop_j_loop_k') else N
        R = ju(cfg) if kind in JK else 1
        P_ = u * R
        flp = cfg['pipeline_style'] == 'flp'
        share = (lambda n: n) if flp else (lambda n: math.ceil(n / II))
        owner = 'gemm' if shared_top else m0
        im, lm = op_impl_lat(cfg, 'fmul', False)
        add(owner, 'fmul', im, lm, share(P_))
        if kind in ('loop_k', 'loop_i_loop_j_loop_k'):
            ir, lr = fadd_rec(cfg)
            if re and u > 1:
                it, lt = op_impl_lat(cfg, 'fadd', False)
                ntree = share(u - 1)
                if it == ir:                       # tree and accumulator share one pool
                    add(owner, 'fadd', ir, max(lr, lt), share(u))
                else:
                    add(owner, 'fadd', it, lt, ntree)
                    add(owner, 'fadd', ir, lr, 1)
            else:
                add(owner, 'fadd', ir, lr, share(P_))
        else:
            ia, la = op_impl_lat(cfg, 'fadd', False)
            add(owner, 'fadd', ia, la, share((N - 1) * R) if re else share(P_))

    if not shared_top and len(comp) > 1:          # O2: parallel copies own identical cores
        base = [o for o in ops if o['owner'] == comp[0]]
        for m in comp[1:]:
            ops += [dict(o, owner=m) for o in base]

    # ---- A: access pattern + X: aux
    for m in comp:
        kind = module_kind(m)
        ap = {}
        p = plans_by.get(m)
        for X, a in (('A', 'lA'), ('B', 'lB')):
            h = hoisted(cfg, X, kind)
            L = layout(cfg, X)
            if h == 'H1':
                if L and (L[0] == 'reshape' or L[2] == 1):
                    continue                              # passed as whole words: no port
                nbk = L[3] if L else 1
                w = N // nbk
                ap[a] = dict(banks_accessed=0, read_ports_per_bank_max=0, read_ports_total=0,
                             write_ports_per_bank_max=0, write_ports_total=0, scalar_inputs=w,
                             scalar_input_bits=32 * w)
            elif h == 'H3':
                nw = invariant_words(cfg, X, kind)
                wb = 32 * (L[3] if L[0] == 'reshape' else 1)
                ap[a] = dict(banks_accessed=0, read_ports_per_bank_max=0, read_ports_total=0,
                             write_ports_per_bank_max=0, write_ports_total=0, scalar_inputs=nw,
                             scalar_input_bits=nw * wb)
            elif h == 'H2':
                bits = 32 * (L[3] if L[0] == 'reshape' else 1)
                ap[a] = dict(banks_accessed=0, read_ports_per_bank_max=0, read_ports_total=0,
                             write_ports_per_bank_max=0, write_ports_total=0, scalar_inputs=N,
                             scalar_input_bits=N * bits)
            else:
                pl = p['plans'][X]
                ap[a] = dict(banks_accessed=pl['banks_touched'], read_ports_per_bank_max=pl['ports'],
                             read_ports_total=pl['ports'] * pl['banks_touched'],
                             write_ports_per_bank_max=0, write_ports_total=0, scalar_inputs=0,
                             scalar_input_bits=0)
            # hoisted values (H2/H3) are scalar inputs: each slot gets its own select mux;
            # memory reads share the bank's ports
            if h != 'H2' and shift_slots(cfg, X, kind):     # H1 returns earlier
                aux.append(shift_core(m, X, cfg, shift_slots(cfg, X, kind)))
            prt = 10 ** 6 if h else p['plans'][X]['ports']
            for n_in in sparsemux_slots(cfg, X, kind, prt):
                aux.append(dict(owner=m, **{'class': 'sparsemux'},
                                params=smx(n_in),
                                count_in_module=1, count_effective=1))
        if kind != 'loop_k':
            L = layout(cfg, 'C')
            R = ju(cfg) if kind in JK else 1
            if L and L[1] == 'complete' and L[2] == 2 and L[0] == 'reshape' and kind == 'loop_j':
                # whole row i in one word, i fixed per call: the caller holds the word (H1 for C)
                ap['lC'] = dict(banks_accessed=0, read_ports_per_bank_max=0, read_ports_total=0,
                                write_ports_per_bank_max=0, write_ports_total=0, scalar_inputs=1,
                                scalar_input_bits=32 * N)
            else:
                per = {}
                for c in accesses(cfg, 'C', kind):
                    b, w = locate(cfg, 'C', c)
                    per.setdefault(b, []).append(w)
                nb = banks(cfg, 'C') if (L and L[0] == 'partition') else 1
                wpb = max(len(set(s)) for s in per.values())
                # a BRAM bank has two ports, so up to 2 stores per bank issue together. They are
                # serialised through one port when (i) C is reshaped (one byte-enabled wide-word
                # write port), (ii) the pipeline is flushable (flp: stores leave from the single
                # flush-controlled output stage), or (iii) an input array is single-port (ram_1p:
                # one operand per cycle, so results emerge one per cycle anyway).
                wp = min(wpb, 2)
                if ((L and L[0] == 'reshape') or cfg.get('pipeline_style') == 'flp'
                        or 'ram_1p' in (cfg.get('A_storage'), cfg.get('B_storage'))):
                    wp = 1
                # read-modify-write of a reshaped word: several lanes of one word written in one
                # iteration, or lanes of a word that stays live across the flattened i/j iterations
                merged = any(len(s) > len(set(s)) for s in per.values())
                rmw = L is not None and L[0] == 'reshape' and (
                    merged or kind == 'loop_i_loop_j_loop_k'
                    or (kind == 'loop_i_loop_j' and L[1] == 'complete'))
                rp = 1 if rmw else 0
                ap['lC'] = dict(banks_accessed=nb, read_ports_per_bank_max=rp, read_ports_total=rp * nb,
                                write_ports_per_bank_max=wp, write_ports_total=wp * nb,
                                scalar_inputs=0, scalar_input_bits=0)
        ap_all[m] = ap
    # copy_out reads lC across all banks: one bank-select mux
    if cfg['pipeline_loop'] == 'none':
        for n_in in top_sparsemux(cfg):
            aux.append(dict(owner='gemm', **{'class': 'sparsemux'},
                            params=smx(n_in),
                            count_in_module=1, count_effective=1))
    if cfg['pipeline_loop'] == 'none':
        for X in ('A', 'B'):
            if shift_slots(cfg, X, 'none'):           # one per unrolled j copy (sequenced k)
                aux.append(shift_core('gemm', X, cfg, ju(cfg)))
    LC = layout(cfg, 'C')                             # copy_out slices lane j of the C word
    if clock(cfg) == 2.5 and LC and LC[0] == 'reshape' and 32 * LC[3] >= 512:
        aux.append(shift_core('gemm_Pipeline_copy_out_copy_out_j', 'C', cfg, 1))
    aux += shl_cores(cfg, st, comp)
    if banks(cfg, 'C') >= 3:
        aux.append(dict(owner='gemm_Pipeline_copy_out_copy_out_j', **{'class': 'sparsemux'},
                        params=smx(banks(cfg, 'C')),
                        count_in_module=1, count_effective=1))
    # merge identical aux cores per owner
    merged = {}
    for a in aux:
        k = (a['owner'], a['class'], a['params'].get('inputs'), a['params'].get('width'))
        if k in merged:
            merged[k]['count_in_module'] += a['count_in_module']; merged[k]['count_effective'] += a['count_effective']
        else:
            merged[k] = a
    aux = list(merged.values())

    # ---- transforms
    tr = {'unroll': [], 'array': [], 'flatten': [], 'balanced_expressions': 0}
    if cfg['pipeline_loop'] == 'j':           # pipelining loop_j fully unrolls the inner loop_k
        tr['unroll'].append(dict(loop='loop_k', kind='implied_by_pipeline', factor=None))
    flat = {x for l in st['loops'] for x in (l['flattened_from'] or [])}
    if 'copy_in' in flat:
        tr['flatten'].append(dict(loop='copy_in', flattened=True))
    tr['flatten'].append(dict(loop='copy_out', flattened=True))
    if cfg['pipeline_loop'] != 'none' and cfg.get('flatten') != 'off':
        if cfg['pipeline_loop'] == 'k' or 'loop_i' in flat:
            tr['flatten'].append(dict(loop='loop_i', flattened='loop_i' in flat))
        if 'loop_k' in flat:
            tr['flatten'].append(dict(loop='loop_j', flattened=True))
    if cfg['unroll_k'] != 1:
        tr['unroll'].append(dict(loop='loop_k', kind='complete' if ku(cfg) == N else 'partial',
                                 factor=ku(cfg)))
    if ju(cfg) > 1:
        tr['unroll'].append(dict(loop='loop_j', kind='partial', factor=ju(cfg)))
    for X, a in (('A', 'lA'), ('B', 'lB'), ('C', 'lC')):
        L = layout(cfg, X)
        if not L:
            continue
        mode, t, dim, f = L
        word = 'partitioning' if mode == 'partition' else 'reshaping'
        det = (f'Complete {word} on dimension {dim}' if t == 'complete'
               else f'{t.capitalize()} {word} with factor {f} on dimension {dim}')
        tr['array'].append(dict(directive='array_' + mode, array=a, detail=det))
    if re:
        # one rebalanced add chain element per product of the unrolled body (u * R); a 2-term
        # chain has nothing to rebalance
        # = adds of the rebalanced chain: j pipeline: 63 per j copy (64 products);
        # k pipeline / no pipeline: u per j copy (acc + u products); u <= 2: nothing to rebalance
        if cfg['pipeline_loop'] == 'j':
            tr['balanced_expressions'] = (N - 1) * ju(cfg)
        else:
            tr['balanced_expressions'] = ku(cfg) * ju(cfg) if ku(cfg) > 2 else 0
    hier = [dict(module=m, instantiations=1, children=[]) for m in st['modules'] if m != 'gemm']
    hier.append(dict(module='gemm', instantiations=1, children=[m for m in st['modules'] if m != 'gemm']))
    if cfg['pipeline_loop'] == 'none':
        ap_all['gemm'] = {}                       # sequential body: no pipelined port map
    hs = dict(module_hierarchy=hier, loops=loops, operators=ops, aux_operators=aux,
                memories=memories(cfg, wnr), access_pattern=ap_all, transforms_applied=tr,
                timing={'target_ns': cfg['clock_ns']},
                latency_breakdown={'stages': st['stages'], 'top_loop_bodies': st['bodies']})
    dropped = [l for l in loops if not_pipelined(cfg, l)]
    if dropped:
        kinds = {}
        for o in hs['operators']:
            kinds.setdefault((o['op'], o['impl']), o)
        ops2 = []
        for l in dropped:
            m = l['module']
            l.update(pipelined=False, pipeline_style=None, ii_target=None, ii_achieved=None,
                     pipeline_depth=None)
            ops2 += [dict(o, owner=m, count_in_module=1, count_effective=1) for o in kinds.values()]
            hs['aux_operators'] = [a for a in hs['aux_operators'] if a['owner'] != m]
            for a in hs['access_pattern'].get(m, {}).values():   # one access per cycle
                if a['read_ports_per_bank_max']:
                    a['read_ports_per_bank_max'] = 1
                    a['read_ports_total'] = a['banks_accessed']
        hs['operators'] = ops2
    return hs
