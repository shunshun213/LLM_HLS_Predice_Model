"""Stage 2 resource model: sanitized HS -> LUT, FF, DSP, BRAM_18K, fits.

Total = sum over modules m of  inst(m) x [ cores(m) + Memory(m) + Expression(m)
                                           + Multiplexer(m) + Register(m) ]
(this bottom-up identity holds exactly in the reports: HS_dataset.md consistency checks).
"""
import math, re
from latency import ports, array_dim_reshape, is_fast

DEVICE = {'LUT': 70560, 'FF': 141120, 'DSP': 360, 'BRAM_18K': 432}

# ------------------------------------------------------------------ R1: core tables
# Characterization table of the floating-point cores: (op, impl) -> clock -> (DSP, FF, LUT)
# per instance (HS_dataset.md finding 1; no conflict across 114 combinations).
FP_COST = {
    ('fadd', 'fabric'):  {10.0: (0, 168, 439), 5.0: (0, 278, 413), 3.33: (0, 611, 421), 2.5: (0, 636, 427)},
    ('fadd', 'fulldsp'): {10.0: (2, 205, 206), 5.0: (2, 365, 244), 3.33: (2, 422, 244), 2.5: (2, 422, 244)},
    ('fmul', 'fabric'):  {10.0: (0, 236, 677), 5.0: (0, 523, 610), 3.33: (0, 763, 654), 2.5: (0, 763, 654)},
    ('fmul', 'meddsp'):  {10.0: (1, 144, 268), 5.0: (1, 326, 302), 3.33: (1, 425, 300), 2.5: (1, 425, 300)},
    ('fmul', 'fulldsp'): {10.0: (2, 144, 119), 5.0: (2, 166, 141), 3.33: (2, 197, 143), 2.5: (2, 254, 152)},
    ('fmul', 'maxdsp'):  {10.0: (3, 143, 140), 5.0: (3, 165, 145), 3.33: (3, 199, 149), 2.5: (3, 199, 149)},
}
# Bank-select multiplexer core (sparsemux): LUT per instance by number of data inputs
# (32-bit data, no FF). Library characterization: a 32-bit N:1 mux.
SPARSEMUX_LUT = {4: 20, 8: 43, 16: 65, 32: 148, 64: 273}
# Pipelined barrel shifter at 2.5 ns: lshr core slicing wide reshaped words (Instance), and the
# shl expression placing a lane / byte-enable into a wide word (reported under Expression with the
# identical 2661 FF / 3250 LUT); both counted here from the HS aux list
LSHR_COST = {'FF': 2661, 'LUT': 3250}


def clock_key(T):
    return min(FP_COST[('fadd', 'fabric')], key=lambda c: abs(c - T))


def sparsemux_lut(n):
    ks = sorted(SPARSEMUX_LUT)          # round the input count up to the next characterized size
    for x in ks:
        if x >= n:
            return SPARSEMUX_LUT[x]
    return SPARSEMUX_LUT[ks[-1]] * n / ks[-1]


def core_cost(hs, module):
    """R1: leaf cores owned by `module` (per instance of the module)."""
    T = clock_key(hs['timing']['target_ns'])
    c = {'LUT': 0, 'FF': 0, 'DSP': 0, 'BRAM_18K': 0}
    for o in hs['operators']:
        if o['owner'] != module:
            continue
        d, f, l = FP_COST[(o['op'], o['impl'])][T]
        n = o['count_in_module']
        c['DSP'] += d * n; c['FF'] += f * n; c['LUT'] += l * n
    for a in hs['aux_operators']:
        if a['owner'] != module:
            continue
        n = a['count_in_module']
        if a['class'] == 'sparsemux':
            c['LUT'] += sparsemux_lut(a['params']['inputs']) * n
        elif a['class'] == 'shift':
            c['LUT'] += LSHR_COST['LUT'] * n; c['FF'] += LSHR_COST['FF'] * n
    return c


# ------------------------------------------------------------------ R2: memory
def bram_width(depth, tdp):
    """Max data width of one BRAM18K for a given depth (7-series/UltraScale aspect ratios).
    True dual port (both ports active) limits the 512-deep shape to 18 bits."""
    for d, w in ((512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2), (16384, 1)):
        if depth <= d:
            return 18 if (tdp and w == 36) else w
    return 1


def array_ports_used(hs, arr):
    """Max ports of one bank in use in the same module (reads + writes)."""
    used = 0
    for mod, ap in hs['access_pattern'].items():
        v = ap.get(arr)
        if v:
            used = max(used, v['read_ports_per_bank_max'] + v['write_ports_per_bank_max'])
    return used


def memory_cost(hs, arr):
    m = hs['memories'][arr]
    c = {'LUT': 0, 'FF': 0, 'BRAM_18K': 0}
    words, bits, banks = m['words_per_bank'][0], m['bits_per_word'][0], m['banks']
    if m['in_registers']:
        c['FF'] += words * bits * banks
        return c
    p = ports(m)
    if m['storage_impl'] == 'LUTRAM':
        c['LUT'] += math.ceil(words / 64) * bits * banks
        c['FF'] += bits * p * banks
        return c
    tdp = _tdp_active(hs, arr)
    per_bank = math.ceil(bits / bram_width(words, tdp))
    copies = 1
    if m.get('storage_inferred_by_tool') and 'RAM_1WNR' in m['storage_inferred_by_tool']:
        # replication: one copy per read port beyond the first
        rp = max((ap[arr]['read_ports_per_bank_max'] for ap in hs['access_pattern'].values() if arr in ap),
                 default=1)
        copies = max(1, rp - 1)
    c['BRAM_18K'] += per_bank * banks * copies
    return c


def _axis_split(hs, arr, dim):
    """(elements per word, banks) along array dimension `dim` (1 = row, 2 = column)."""
    t = array_dim_reshape(hs, arr)
    m = hs['memories'][arr]
    if t and t[3] == dim:
        if t[0] == 'array_reshape':
            return max(1, m['bits_per_word'][0] // 32), 1
        return 1, m['banks']
    return 1, 1


def _top_body_accesses(hs, arr):
    """Accesses per bank in one iteration of the innermost non-pipelined top loop.
    u = k elements per iteration, R = j copies (unroll of loop_j)."""
    parents = {b['loop'] for b in hs['latency_breakdown']['top_loop_bodies'] if b['children']}
    inner = [l for l in hs['loops'] if l['module'] == 'gemm' and l['loop'] in ('loop_k', 'loop_j')
             and l['loop'] not in parents]          # inlined datapath only (no call inside)
    if not inner:
        return 1
    l = min(inner, key=lambda x: x['trip_count'] if x['loop'] == 'loop_k' else 10**9)
    from features import unroll_factor             # u from the transforms (non-divisor safe)
    u = (unroll_factor(hs, 'loop_k') or 64 // l['trip_count']) if l['loop'] == 'loop_k' else 64
    uj = [x.get('factor') or 1 for x in hs['transforms_applied']['unroll'] if x['loop'] == 'loop_j']
    R = max(uj) if uj else 1
    if arr == 'lA':                       # A[i][k]: k is dim 2
        e, b = _axis_split(hs, 'lA', 2)
        return math.ceil(math.ceil(u / e) / b)
    if arr == 'lB':                       # B[k][j]: k is dim 1, j is dim 2
        ek, bk = _axis_split(hs, 'lB', 1)
        ej, bj = _axis_split(hs, 'lB', 2)
        return math.ceil(math.ceil(u / ek) / bk) * math.ceil(math.ceil(R / ej) / bj)
    ej, bj = _axis_split(hs, 'lC', 2)     # C[i][j]: R writes per iteration
    return math.ceil(math.ceil(R / ej) / bj)


def _top_c_writes(hs):
    """Writes per lC bank per top iteration when the top FSM writes the R = unroll(loop_j)
    results back (calls or inlined body alike)."""
    uj = [x.get('factor') or 1 for x in hs['transforms_applied']['unroll'] if x['loop'] == 'loop_j']
    R = max(uj) if uj else 1
    ej, bj = _axis_split(hs, 'lC', 2)
    return math.ceil(math.ceil(R / ej) / bj)


def _tdp_active(hs, arr):
    """A bank needs true dual port when two reads or two writes hit it in the same cycle
    (1 read + 1 write fits the simple-dual-port 36-bit shape):
      - >= 2 reads or >= 2 writes per bank in one pipelined module (access pattern), or
      - the array is replicated as RAM_1WNR (each copy serves reads on both ports), or
      - values hoisted into the caller FSM (scalar inputs): read 2 per cycle (rule L5), or
      - the innermost top-level (non-pipelined) loop body accesses >= 2 words of a bank."""
    m = hs['memories'][arr]
    if m['storage_ports'] == '1P':
        return False
    if m.get('storage_inferred_by_tool') and 'RAM_1WNR' in m['storage_inferred_by_tool']:
        return True
    in_ap = False
    for ap in hs['access_pattern'].values():
        v = ap.get(arr)
        if not v:
            continue
        if v['read_ports_per_bank_max'] >= 2 or v['write_ports_per_bank_max'] >= 2:
            return True
        if v['read_ports_per_bank_max'] + v['write_ports_per_bank_max'] > 0:
            in_ap = True
        elif v.get('scalar_inputs', 0) >= 2 and arr in ('lA', 'lB'):
            return True          # >= 2 hoisted words per bank: FSM reads 2 words per cycle
    if in_ap:
        return False
    if arr in ('lA', 'lB'):
        # operand array absent from the ports of a called compute module: hoisted by the FSM
        comp = [mod for mod in hs['access_pattern'] if mod != 'gemm' and 'copy' not in mod]
        if comp and any(arr not in hs['access_pattern'][mod] for mod in comp):
            return True
        return _top_body_accesses(hs, arr) >= 2
    return _top_c_writes(hs) >= 2


# ------------------------------------------------------------------ assembly
CATS = ('Expression', 'Multiplexer', 'Register')


def predict_resources(hs, cat_model, oracle=None):
    """cat_model(hs, module, cat) -> {'LUT':..,'FF':..}; oracle(module, cat) for ablations."""
    tot = {'LUT': 0, 'FF': 0, 'DSP': 0, 'BRAM_18K': 0}
    for mm in hs['module_hierarchy']:
        mod, inst = mm['module'], mm['instantiations']
        c = core_cost(hs, mod)
        for k in c:
            tot[k] += c[k] * inst
        for cat in CATS:
            v = oracle(mod, cat) if oracle and oracle(mod, cat) is not None else cat_model(hs, mod, cat)
            tot['LUT'] += v.get('LUT', 0) * inst
            tot['FF'] += v.get('FF', 0) * inst
    for arr, m in hs['memories'].items():
        c = memory_cost(hs, arr)
        for k in c:
            tot[k] += c[k]
    tot['fits'] = all(tot[k] <= DEVICE[k] for k in DEVICE)
    return tot
