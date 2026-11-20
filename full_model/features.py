"""Structural quantities per module, computed only from the sanitized HS.
Each quantity has a hardware meaning; the cost model multiplies them by per-unit
constants (LUT or FF per unit) listed in fit.py BASIS / constants.json."""
import math
from latency import (ports, is_fast, op_lat, array_dim_reshape, hoisted_reads, read_cycles,
                     innermost_body, pipelined_module_latency)


def clog2(x):
    return max(1, math.ceil(math.log2(max(2, x))))


def module_class(m):
    if m == 'gemm':
        return 'top'
    if 'copy_in' in m:
        return 'copy_in'
    if 'copy_out' in m:
        return 'copy_out'
    return 'compute'


def unroll_factor(hs, loop, tc_div=1):
    """unroll factor of `loop` from the transforms (complete or implied by a pipeline -> 64);
    without an explicit entry it follows from the trip count, 64 * tc_div / TC (exact when the
    factor divides 64; a non-divisor factor u keeps u copies with TC = ceil(64 / u))."""
    fs = [x for x in hs['transforms_applied']['unroll'] if x['loop'] == loop]
    for x in fs:
        if x['kind'] in ('complete', 'implied_by_pipeline'):
            return 64
    for x in fs:
        if x.get('factor'):
            return int(x['factor'])
    return None


def unroll_inside(hs, m, loop):
    """(u, R): products per dot product per iteration, independent dot products per iteration."""
    name = m.replace('gemm_Pipeline_', '').rstrip('0123456789')
    uk = unroll_factor(hs, 'loop_k') or 1
    uj = unroll_factor(hs, 'loop_j') or 1
    if name in ('loop_i_loop_j_loop_k', 'loop_k'):
        return uk, 1
    if name == 'loop_j':
        return 64, uj
    if name == 'loop_i_loop_j':
        return 64, uj
    if name == 'loop_i':
        return 64, 64
    return 1, 1


def cores(hs, m):
    own = [o for o in hs['operators'] if o['owner'] == m]
    src = own if own else [o for o in hs['operators'] if o['owner'] == 'gemm']
    nm = sum(o['count_in_module'] for o in src if o['op'] == 'fmul')
    na = sum(o['count_in_module'] for o in src if o['op'] == 'fadd')
    return max(1, nm), max(1, na), bool(own)


def wide_bits(hs, arrays):
    return sum(hs['memories'][a]['bits_per_word'][0] for a in arrays if hs['memories'][a]['bits_per_word'][0] > 32)


def enclosing_top_loops(hs, m):
    """Names of the top-level (gemm) loops whose body (transitively) calls module m."""
    parent = {}
    for b in hs['latency_breakdown']['top_loop_bodies']:
        for c in b['children']:
            parent[c] = b['loop']
    out, cur = [], m
    while cur in parent:
        cur = parent[cur]
        out.append(cur)
    return out


def hoisted_array_bits(hs, a):
    """Bits of array a the top FSM keeps in registers for the compute calls (0 if not hoisted)."""
    fixed_by = {'lA': ('loop_i', 'loop_i_loop_j'), 'lB': ('loop_j', 'loop_i_loop_j')}[a]
    kdim = 2 if a == 'lA' else 1
    best = 0
    for mod, ap in hs['access_pattern'].items():
        if mod == 'gemm' or 'copy' in mod:
            continue
        v = ap.get(a)
        hoisted = (v is None) or bool(v.get('scalar_inputs'))
        if not hoisted:
            continue
        enc = enclosing_top_loops(hs, mod)
        mem = hs['memories'][a]
        if any(e in fixed_by for e in enc):
            # one row of A (column of B) = 64 elements. Reshaped along k: the words hold exactly
            # that row -> 64 x 32 bits. Reshaped along the other index: the row is spread over 64
            # words (one lane each). Complete reshape: the FSM keeps those 64 whole words, i.e. the
            # whole array (64 x W). Block / cyclic reshape (W = 64..512): the reports show a
            # 2-row slice of every word, 64 x 64 bits [low confidence: W = 64 (3 designs), 128
            # (mixed_65), 256 (S4d_wrong_dim_001); W = 512 (mixed_10) is not separable].
            t = array_dim_reshape(hs, a)
            other = t and t[0] == 'array_reshape' and t[3] != kdim and mem['bits_per_word'][0] > 32
            if not other:
                bits = 64 * 32
            elif t[1] == 'complete':
                bits = 64 * mem['bits_per_word'][0]   # 64 whole words = the whole array
            else:
                bits = 64 * 64                        # 2-row slice of each word (empirical)

        else:
            bits = mem['total_bits'] if mem.get('total_bits') else 64 * 64 * 32
        best = max(best, bits)
    return best


# index that each array dimension carries in C = A*B
DIM_INDEX = {'lA': {1: 'i', 2: 'k'}, 'lB': {1: 'k', 2: 'j'}, 'lC': {1: 'i', 2: 'j'}}


def packed_index(hs, a):
    """(index packed into a word, elements per word) for a reshaped array, else None."""
    t = array_dim_reshape(hs, a)
    m = hs['memories'][a]
    if not t or t[0] != 'array_reshape' or m['bits_per_word'][0] <= 32:
        return None
    return DIM_INDEX[a][t[3]], m['bits_per_word'][0] // 32


def dynamic_select(hs, a, unrolled):
    """True when the position of the element inside the wide word changes at run time
    (needs a part-select / merge datapath). `unrolled` = {index: copies per iteration}.
    Cyclic reshape packs consecutive indices (word w = elements f*w .. f*w+f-1): static
    when the packed index is unrolled at least f-fold. Block reshape packs indices with
    stride 64/f (word w = elements w, w+64/f, ...): the lane is the high part of the index,
    static only when the whole dimension (64) is unrolled."""
    p = packed_index(hs, a)
    if p is None:
        return False
    idx, epw = p
    kind = array_dim_reshape(hs, a)[1]
    need = epw if kind == 'cyclic' else 64
    return unrolled.get(idx, 1) < need


def shl_written(hs, m):
    """arrays whose lane writes in module m are done by pipelined shl cores (2.5 ns, words >=
    512 bit): their combinational lane-merge logic is replaced by the shl, which is costed
    explicitly (resources.core_cost)."""
    if not any(a['owner'] == m and a['params'].get('op') == 'shl' for a in hs['aux_operators']):
        return set()
    return {a for a in ('lA', 'lB', 'lC') if hs['memories'][a]['bits_per_word'][0] >= 512}


def module_features(hs, m):
    """Returns dict of named quantities for module m."""
    T = hs['timing']['target_ns']
    f = {'one': 1.0}
    # output register of each pipelined shl (one W-bit register per shifter)
    f['shl_reg_bits'] = float(sum(a['params'].get('total_bits', a['count_in_module'] * a['params']['width'])
                                  for a in hs['aux_operators']
                                  if a['owner'] == m and a['params'].get('op') == 'shl'))
    # W-bit AND/OR merge of the shifted lane with the old word, for words written through a
    # pipelined shl (the shl only places lane and byte enable; the merge stays combinational)
    f['shl_merge_bits'] = 0.0
    cls = module_class(m)
    if cls in ('copy_in', 'copy_out'):
        l = [x for x in hs['loops'] if x['module'] == m][0]
        arrs = ['lC'] if cls == 'copy_out' else ['lA', 'lB']
        f['depth'] = l['pipeline_depth']
        f['cnt_bits'] = clog2(l['trip_count'] + 1) + (clog2(64) if l.get('flattened_from') else 0)
        # read-modify-write of reshaped words: wide data path (bits) per array
        f['wide_bits'] = wide_bits(hs, arrs)
        f['wide_bits_x_depth'] = f['wide_bits'] * max(0, l['pipeline_depth'] - 2)
        f['banks'] = sum(hs['memories'][a]['banks'] for a in arrs if hs['memories'][a]['banks'] > 1)
        f['fast'] = 1.0 if is_fast(hs) else 0.0
        f['depth_x_32'] = 32 * max(0, l['pipeline_depth'] - 2) * len(arrs)
        # reshaped words written/read by the copy loop (inner copy index = array dim 2):
        #  packed along dim 2: consecutive elements share one word -> assembled in a W-bit
        #  register with a lane merge; packed along dim 1: per-element read-modify-write with
        #  lane select (the select logic saturates beyond 512-bit words: byte-enable writes)
        f['wide_inner'] = 0.0; f['wide_outer'] = 0.0
        shl = shl_written(hs, m) if cls == 'copy_in' else set()
        for a in arrs:
            t = array_dim_reshape(hs, a)
            W = hs['memories'][a]['bits_per_word'][0]
            if W <= 32 or not t or t[0] != 'array_reshape':
                continue
            if a in shl:
                f['shl_merge_bits'] += W
                continue
            if t[3] == 2:
                f['wide_inner'] += W
            else:
                f['wide_outer'] += min(W, 512)
        return f
    if cls == 'compute':
        l = [x for x in hs['loops'] if x['module'] == m][0]
        u, R = unroll_inside(hs, m, l)
        nm, na, own = cores(hs, m)
        if l['pipelined']:
            II = l['ii_achieved'] or 1
            D = l['pipeline_depth'] or 1
        else:                                       # loop module the tool did not pipeline:
            D = innermost_body(hs, l)               # one iteration = rule-L3 body schedule
            II = D                                  # iterations do not overlap
        La, Lm = op_lat(hs, 'fadd'), op_lat(hs, 'fmul')
        P = u * R                                   # products per iteration
        s_mul = math.ceil(P / nm)                    # uses of one fmul per iteration
        s_add = math.ceil(P / na)
        bal = hs['transforms_applied'].get('balanced_expressions', 0) > 0
        ap = hs['access_pattern'].get(m, {})
        # loop-control bits: the combined trip counter + one index register per original loop
        # that was flattened into this pipeline (each index needs its own counter + compare)
        ff = l.get('flattened_from') or []
        f['cnt_bits'] = clog2(l['trip_count'] + 1) + sum(clog2(64 + 1) for _ in ff)
        f['hoist_bits'] = sum(v.get('scalar_input_bits', 0) for v in ap.values())
        # operator input multiplexing when one core serves s uses per iteration
        f['mux_share'] = 64 * (nm * (s_mul - 1) + na * (s_add - 1))
        # with II > 1 every operand/product register is written in one of II slots:
        # a multi-source input mux per 32-bit term
        f['mux_ii'] = 32 * P if (l['ii_achieved'] or 1) > 1 else 0.0
        # pipeline FSM: one state per cycle of the initiation interval; registers written in
        # several of those states get an enable/data mux
        f['II_states'] = II if II > 1 else 0.0
        # memory-port multiplexing: accesses per port per iteration beyond the first
        mp = 0
        for a, v in ap.items():
            rp = v['read_ports_total'] + v['write_ports_total']
            if rp <= 0:
                continue
            acc = P if a in ('lA', 'lB') else R
            per_port = math.ceil(acc / rp)
            mp += rp * max(0, per_port - 1) * (32 + clog2(hs['memories'][a]['words_per_bank'][0]))
        f['mux_mem'] = mp
        f['II'] = II
        f['depth'] = D
        # values waiting for the serial add chain (no reassociation, unshared adders):
        # sum over products of their wait La*k, registered every II cycles, 32 bits each
        # Regime switch observed in the reports: with one adder per product (unshared) and
        # II > 1 the scheduler issues all products early and they wait for the chain;
        # with shared cores (or II = 1) products are issued just in time.
        if not bal and na >= P and P > 1 and II > 1:
            f['chain_wait'] = 32 * La * u * (u - 1) / 2 * R / II
            f['early_ops'] = 32 * P * P / II          # operands issued early, held until use
        else:
            f['chain_wait'] = 0.0
            f['early_ops'] = 0.0
        f['P32'] = 32 * P                               # A, B, product register per term
        # registers a value needs in a pipeline with initiation interval II: ceil(lifetime/II)
        f['sum_live'] = 32 * R * math.ceil(D / II)        # running sums live for the whole depth
        f['prod_live'] = 32 * P * math.ceil((Lm + 1) / II)  # operands/products live ~ L_mul + 1
        f['stages'] = math.ceil(D / II)                    # overlapped iterations (valid/control bits)
        # stages that hold a *different* iteration than the one being issued: with II >= D the
        # pipeline never overlaps iterations and needs no per-stage copies of in-flight values
        # in-flight values of the overlapped stages: held in this module's registers only when
        # the module owns its fp cores; with the cores shared in the top (several loop_k
        # copies) the pipeline registers of the cores carry the values outside the module
        f['depth_ov'] = max(0, D - II) if own else 0.0
        f['depth_ov_ext'] = 0.0 if own else max(0, D - II)
        f['core_regs'] = 32 * (nm + na)                  # operand/result registers per core
        f['core_regs_shared'] = 32 * (nm + na) if (s_mul > 1 or s_add > 1) else 0.0  # muxed operands registered
        # k fully unrolled inside the pipeline (loop_j / loop_i_loop_j modules): the 64 A values
        # of the row and the 64 B values of each column are held as operand registers
        f['kfull_operands'] = 32 * 64 * (1 + R) if u == 64 else 0.0
        f['wide_bits'] = wide_bits(hs, [a for a in ('lA', 'lB', 'lC') if a in ap])
        unr = {'k': u, 'j': R if R > 1 else 1}
        # dynamic lane select of reshaped words, W bits per element access. Two library
        # structures: a 64:1 lane mux tree for whole-row words (W >= 1024) and a barrel-shift
        # style select for narrower words; their per-bit costs differ.
        f['dyn_read_bits'] = 0.0
        f['dyn_read_wide'] = 0.0
        for a in ('lA', 'lB'):
            v = ap.get(a)
            if v and (v['read_ports_total'] > 0) and dynamic_select(hs, a, unr):
                W = hs['memories'][a]['bits_per_word'][0]
                acc = (u if a == 'lA' else u * R) * W
                if W >= 1024:
                    f['dyn_read_wide'] += acc
                else:
                    f['dyn_read_bits'] += acc
        # one part-select unit serves the II slots of an iteration: select hardware per cycle
        f['dyn_read_pc'] = f['dyn_read_bits'] / II
        v = ap.get('lC')
        wbits = R * hs['memories']['lC']['bits_per_word'][0] if v and dynamic_select(hs, 'lC', unr) else 0.0
        shlC = 'lC' in shl_written(hs, m)
        f['dyn_write_bits'] = 0.0 if shlC else wbits
        f['shl_merge_bits'] += wbits if shlC else 0.0
        # hoisted wide words passed in as scalars: one dynamic select per value used
        f['dyn_hoist_bits'] = 0.0
        for a in ('lA', 'lB'):
            v = ap.get(a)
            if v and v.get('scalar_inputs') and dynamic_select(hs, a, unr):
                W = hs['memories'][a]['bits_per_word'][0]
                # one select per element used per iteration (A: u, B: u*R)
                acc = (u if a == 'lA' else u * R) * W
                if W >= 1024:
                    f['dyn_read_wide'] += acc
                else:
                    f['dyn_hoist_bits'] += acc
        # address offsets/compares of the u unrolled accesses (k+0..k+u-1) into one bank;
        # free when A or B is partitioned/reshaped along k (each access has its own bank/word)
        ksplit = False
        for a, kd in (('lA', 2), ('lB', 1)):
            t = array_dim_reshape(hs, a)
            if t and t[3] == kd:
                ksplit = True
        aops = (u * math.log2(u) if u > 1 else 0.0) * (0.0 if ksplit else 1.0)
        f['addr_ops'] = aops
        # k fully unrolled (u = 64): all offsets are compile-time constants folded into the
        # address -> no run-time offset adders; partial unroll needs k0 + t adders
        f['addr_partial'] = aops if u < 64 else 0.0
        # partial unroll: the u operand pairs + products of an iteration are registered while
        # the serial chain consumes them (one 32-bit register per term; fitted 1.36 FF per bit)
        f['P32_partial'] = 32.0 * P if u < 64 else 0.0
        f['ksplit'] = 1.0 if ksplit else 0.0
        f['wide_sel'] = sum(hs['memories'][a]['bits_per_word'][0] * (P if a != 'lC' else R)
                            for a in ap if hs['memories'][a]['bits_per_word'][0] > 32)
        f['banks_sel'] = sum(v['banks_accessed'] for v in ap.values() if v['banks_accessed'] > 1)
        f['fast'] = 1.0 if is_fast(hs) else 0.0
        return f
    # ---------------- top
    gl = [x for x in hs['loops'] if x['module'] == 'gemm']
    f['n_calls'] = len([s for s in hs['latency_breakdown']['stages'] if s['kind'] == 'call']) + \
        sum(len([c for c in b['children'] if c.startswith('gemm_')]) for b in hs['latency_breakdown']['top_loop_bodies'])
    f['n_loops'] = len(gl)
    f['cnt_bits'] = sum(clog2(l['trip_count'] + 1) for l in gl)
    # hoisted values held in registers by the top FSM (rule R4):
    # a value read before a call is kept for the whole call. If a top loop around the call
    # fixes the array's outer index (i for A, j for B) only one row/column is kept:
    # (words holding it) x (word bits); otherwise every row/column is needed: whole array.
    # length of the sequential read the FSM performs per bank to hoist operands: one state,
    # one address constant and one capture-enable per word read (scalar_inputs is per bank)
    f['hoist_seq'] = 0.0
    for mod, ap in hs['access_pattern'].items():
        if mod == 'gemm' or 'copy' in mod:
            continue
        for a, v in ap.items():
            if v.get('scalar_inputs') and a in ('lA', 'lB'):
                f['hoist_seq'] = max(f['hoist_seq'], v['scalar_inputs'])
    f['hoist_reg_bits'] = 0.0
    f['hoist_words'] = 0.0
    for a in ('lA', 'lB'):
        b = hoisted_array_bits(hs, a)
        f['hoist_reg_bits'] += b
        # words the FSM reads for the hoisting (address generation + capture enables)
        f['hoist_words'] += b / hs['memories'][a]['bits_per_word'][0] if b else 0.0
    # non-pipelined innermost body inside the top: its unrolled datapath lives in the top
    inner = [l for l in gl if l['loop'] in ('loop_k', 'loop_j') and not any(
        b['loop'] == l['loop'] and b['children'] for b in hs['latency_breakdown']['top_loop_bodies'])]
    body_states, body_P, body_wide = 0, 0, 0
    for l in inner:
        body_states += innermost_body(hs, l)
        u = (unroll_factor(hs, 'loop_k') or 1) if l['loop'] == 'loop_k' else 64
        R = 1 if l['loop'] == 'loop_k' else (unroll_factor(hs, 'loop_j') or 1)
        body_P += u * R
    if inner:
        body_wide = wide_bits(hs, ['lA', 'lB', 'lC'])
    f['body_states'] = body_states
    f['body_P'] = body_P
    f['body_wide_sel'] = 0.0
    f['body_dyn_write'] = 0.0
    for l in inner:
        u = (unroll_factor(hs, 'loop_k') or 1) if l['loop'] == 'loop_k' else 64
        R = 1 if l['loop'] == 'loop_k' else (unroll_factor(hs, 'loop_j') or 1)
        uj = [x.get('factor') or 1 for x in hs['transforms_applied']['unroll'] if x['loop'] == 'loop_j']
        Rj = max(uj) if uj else 1
        unr = {'k': u, 'j': Rj}
        for a in ('lA', 'lB'):
            if dynamic_select(hs, a, unr):
                W = hs['memories'][a]['bits_per_word'][0]
                acc = (u if a == 'lA' else u * Rj) * W
                if W >= 1024:
                    f['body_wide_sel_wide'] = f.get('body_wide_sel_wide', 0.0) + acc
                else:
                    f['body_wide_sel'] += acc
        if dynamic_select(hs, 'lC', unr):
            if 'lC' in shl_written(hs, 'gemm'):
                f['shl_merge_bits'] += Rj * hs['memories']['lC']['bits_per_word'][0]
            else:
                f['body_dyn_write'] += Rj * hs['memories']['lC']['bits_per_word'][0]
    f['states'] = body_states + 2 * f['n_calls'] + 2 * len(gl) + 4
    # FSM kind of the top: only calls / loops around calls / loops with an inlined datapath
    f['tk_calls'] = 1.0 if not gl else 0.0
    f['tk_loopcalls'] = 1.0 if gl and not inner else 0.0
    f['tk_body'] = 1.0 if inner else 0.0
    nm = sum(o['count_in_module'] for o in hs['operators'] if o['owner'] == 'gemm' and o['op'] == 'fmul')
    na = sum(o['count_in_module'] for o in hs['operators'] if o['owner'] == 'gemm' and o['op'] == 'fadd')
    f['top_cores'] = nm + na
    # shared cores at top used by several sequential call sites: input muxes
    calls_with_ops = [c for b in hs['latency_breakdown']['top_loop_bodies'] for c in b['children']
                      if c.startswith('gemm_')]
    f['shared_call_mux'] = 64 * (nm + na) * max(0, len(calls_with_ops) - 1) if (nm + na) else 0
    f['mux_body'] = 64 * (nm + na) * max(0, math.ceil(body_P / max(1, nm)) - 1) if body_P else 0
    # memory ports driven by the top (copy modules + compute modules + top body): address/data muxing
    f['mem_arrays_banks'] = sum(m['banks'] for m in hs['memories'].values())
    f['hoist_mux'] = 0.0
    for mod, ap in hs['access_pattern'].items():
        for a, v in ap.items():
            if v.get('scalar_inputs'):
                f['hoist_mux'] += hs['memories'][a]['banks'] * 32 if hs['memories'][a]['banks'] > 1 else 0
    f['wide_all'] = wide_bits(hs, ['lA', 'lB', 'lC'])
    # wide words the top FSM itself reads/writes (inlined body, or arrays the compute modules
    # do not access through ports, e.g. lC accumulated as a scalar and written back by the top):
    # word registers + read-modify-write merge
    comp_aps = [ap for mod, ap in hs['access_pattern'].items() if mod != 'gemm' and 'copy' not in mod]
    tw = 0.0
    for a in ('lA', 'lB', 'lC'):
        W = hs['memories'][a]['bits_per_word'][0]
        if W <= 32:
            continue
        via_ports = any(a in ap and (ap[a]['read_ports_total'] + ap[a]['write_ports_total']) > 0 for ap in comp_aps)
        if (inner or not via_ports) and not (a != "lC" and hoisted_array_bits(hs, a)):  # hoisted words counted once
            tw += W
    f['top_wide_bits'] = tw
    # C accumulated as scalars and written back by the top into words packed along j:
    # R lane writes (merge) per word and one W-bit word register
    f['top_c_lane'] = 0.0; f['top_c_word'] = 0.0
    pC = packed_index(hs, 'lC')
    viaC = any('lC' in ap and (ap['lC']['read_ports_total'] + ap['lC']['write_ports_total']) > 0 for ap in comp_aps)
    if pC and pC[0] == 'j' and not viaC and not inner:
        uj = [x.get('factor') or 1 for x in hs['transforms_applied']['unroll'] if x['loop'] == 'loop_j']
        Wc = hs['memories']['lC']['bits_per_word'][0]
        lane = (max(uj) if uj else 1) * Wc
        if 'lC' in shl_written(hs, 'gemm'):
            f['shl_merge_bits'] += lane
        else:
            f['top_c_lane'] = lane
        f['top_c_word'] = Wc
    # below 10 ns the scheduler can no longer chain the word select / merge and the inlined
    # datapath combinationally inside one state: intermediate values get registered
    slow = hs['timing']['target_ns'] >= 9.99
    f['top_wide_bits_fast'] = 0.0 if slow else tw
    f['body_P_fast'] = 0.0 if slow else float(body_P)
    f['fast'] = 1.0 if is_fast(hs) else 0.0
    return f
