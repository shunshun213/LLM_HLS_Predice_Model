"""Stage 2 latency model: sanitized HS -> top latency (cycles).

All rules are white-box. Notation:
  TC  trip count, II achieved II, D pipeline depth (all from HS loops)
  p(a) read ports per bank of array a: 1 for storage '1P', else 2
"""
import math, re

FAST_NS = 3.34          # clock periods <= this are "fast" (3.33 ns, 2.5 ns)


def ports(mem):
    return 1 if mem.get('storage_ports') == '1P' else 2


def is_fast(hs):
    return hs['timing']['target_ns'] <= FAST_NS


# ---------------------------------------------------------------- rule L1
def pipelined_module_latency(loop):
    """L1: a module that wraps one pipelined loop.
    loop latency = (TC-1)*II + D - 1 ; module adds 2 cycles (start/done handshake)
    => module latency = (TC-1)*II + D + 1   (exact on all 1577 pipelined loops)."""
    return (loop['trip_count'] - 1) * loop['ii_achieved'] + loop['pipeline_depth'] + 1


# ---------------------------------------------------------------- helpers
def array_dim_reshape(hs, arr):
    """(directive, kind, factor, dim) of the partition/reshape applied to arr, or None."""
    for t in hs['transforms_applied']['array']:
        if t['array'] != arr:
            continue
        d = t['detail']
        m_dim = re.search(r'dimension (\d)', d)
        m_f = re.search(r'factor (\d+)', d)
        kind = 'complete' if d.startswith('Complete') else ('cyclic' if d.startswith('Cyclic') else 'block')
        return (t['directive'], kind, int(m_f.group(1)) if m_f else None, int(m_dim.group(1)) if m_dim else None)
    return None


def hoisted_reads(hs, module):
    """Scalar inputs a pipelined module receives from its caller: values read by the
    caller FSM before the call (one per cycle per port).  Returns list of (array, words)."""
    ap = hs['access_pattern'].get(module)
    out = []
    if ap is None:
        return out
    for a, v in ap.items():
        if v.get('scalar_inputs'):
            out.append((a, v['scalar_inputs']))
    # arrays the compute body needs but that do not appear as ports at all: the module
    # receives them as scalars too (whole reshaped words) -> count the words of one row/column
    for a in ('lA', 'lB'):
        if a not in ap:
            m = hs['memories'][a]
            elems_per_word = max(1, m['bits_per_word'][0] // 32)
            words = max(1, 64 // elems_per_word) if m['banks'] == 1 else max(1, 64 // m['banks'])
            out.append((a, words))
    return out


def read_cycles(hs, arr, words):
    return math.ceil(words / ports(hs['memories'][arr]))


# ---------------------------------------------------------------- operators
def op_lat(hs, op, owner=None):
    """Scheduled latency of the fp core used by the compute body."""
    cands = [o for o in hs['operators'] if o['op'] == op and (owner is None or o['owner'] == owner)]
    if not cands:
        cands = [o for o in hs['operators'] if o['op'] == op]
    return max(o['latency'] for o in cands)


def op_impls(hs, op):
    return {o['impl'] for o in hs['operators'] if o['op'] == op}


# ---------------------------------------------------------------- rule L3a
def read_path_delta(hs):
    """Extra registers the scheduler inserts on the operand path (BRAM read -> word/bank
    select -> fmul) as the clock period shrinks:
        10 ns: 0   5 ns: 1   3.33 ns: 1 (+1 if an operand word is wider than 32 bits)   2.5 ns: 2
    plus 1 for a meddsp fmul at 10 ns (its output is registered once more)."""
    T = hs['timing']['target_ns']
    wide = any(hs['memories'][a]['bits_per_word'][0] > 32 for a in ('lA', 'lB'))
    if T >= 9.99:
        d = 1 if 'meddsp' in op_impls(hs, 'fmul') else 0
    elif T >= 4.99:
        d = 1
    elif T >= 3.3:
        d = 2 if wide else 1
    else:
        d = 2
    return d


# ---------------------------------------------------------------- rule L3
def innermost_body(hs, loop):
    """L3: one iteration of a non-pipelined innermost loop_k (or loop_j with k fully
    unrolled) holding u copies of  sum += A*B.
    Without reassociation the u adds form a chain; each add costs L_fadd + 1
    (result registered before the next add).  The products overlap with the chain,
    so only the first fmul is exposed: L_fmul + 2 (address + BRAM read).
       body = u*(L_fadd+1) + L_fmul + 2 + delta
    delta: read-path registers (read_path_delta): 10 ns 0 (+1 for a meddsp fmul), 5 ns 1,
           3.33 ns 1 (+1 if an operand word is wider than 32 bits), 2.5 ns 2; an lshr slicer on
           the read path gives max(delta, L_lshr + 1).
    With unsafe_math reassociation (balanced_expressions > 0) the chain becomes a tree
    of depth ceil(log2 u) plus the accumulator add."""
    from features import unroll_factor
    u = (unroll_factor(hs, 'loop_k') or 1) if loop['loop'] == 'loop_k' else 64
    L_add = op_lat(hs, 'fadd')
    L_mul = op_lat(hs, 'fmul')
    delta = read_path_delta(hs)
    if hs['transforms_applied'].get('balanced_expressions', 0) > 0 and u > 1:
        # reassociated: resource-constrained list schedule (rule L3b)
        R = (unroll_factor(hs, 'loop_j') or 1) if loop['loop'] == 'loop_j' else 1
        k_per = 64 if loop['loop'] == 'loop_j' else u
        return tree_body(hs, k_per, R, L_add, L_mul, delta) + (1 if loop['loop'] == 'loop_j' else 0)
    else:
        chain = u * (L_add + 1)
    # wide reshaped words are sliced by an lshr core on the read path (its own latency)
    shifts = [a for a in hs['aux_operators'] if a.get('class') == 'shift'
              and a['params'].get('op', 'lshr') != 'shl'          # shl = write path, not read
              and a['owner'] in ('gemm', loop['module'])]
    if shifts:
        # the lshr slicer (latency from its core) + its output register replace the
        # clock-dependent read registers
        delta = max(delta, max(a['params'].get('latency', 0) for a in shifts) + 1)
    body = chain + L_mul + 2 + delta
    if loop['loop'] == 'loop_j':
        body += 1        # extra state for the fully-unrolled k body inside loop_j
    return body


# ---------------------------------------------------------------- rule L3b
def useful_elems_per_read(hs, arr):
    """Operand elements one read of `arr` delivers for one dot product.
    The dot product walks k: A[i][k] (k = dim 2 of lA), B[k][j] (k = dim 1 of lB).
    Reshape along the k dimension packs f useful elements per word; partition along the
    k dimension gives f banks read in parallel.  Along the other dimension: 1."""
    kdim = 2 if arr == 'lA' else 1
    m = hs['memories'][arr]
    t = array_dim_reshape(hs, arr)
    per_word, banks = 1, 1
    if t:
        directive, kind, f, dim = t
        f = f or (64 if kind == 'complete' else 1)
        if dim == kdim:
            if directive == 'array_reshape':
                per_word = max(1, m['bits_per_word'][0] // 32)
            else:
                banks = m['banks']
    return per_word * banks * ports(m)


def tree_body(hs, K, R, L_add, L_mul, delta):
    """L3b: one iteration holding R independent dot products of K terms each, with
    unsafe_math reassociation (adder tree + accumulator).  Greedy list schedule:
      - operand reads: per cycle, useful_elems_per_read(a) elements of A and B
        (A is shared by the R copies, B is not); data ready 1 + delta cycles later
      - n_fmul / n_fadd pipelined cores (count in the owner module), one issue per cycle
        each, result ready L + 1 cycles after issue
      - adds combine the two earliest-ready partial sums of the same copy; the last add
        is the accumulation into sum.
    Body latency = ready time of the last accumulation + 1 (write-back state)."""
    import heapq
    n_mul = max(1, sum(o['count_in_module'] for o in hs['operators'] if o['op'] == 'fmul' and o['owner'] == 'gemm'))
    n_add = max(1, sum(o['count_in_module'] for o in hs['operators'] if o['op'] == 'fadd' and o['owner'] == 'gemm'))
    ea, eb = useful_elems_per_read(hs, 'lA'), useful_elems_per_read(hs, 'lB')
    # operand ready times
    rdA = [k // ea + 1 + delta for k in range(K)]
    rdB = [[(c * K + k) // eb + 1 + delta for k in range(K)] for c in range(R)]
    # product issue: in order of operand availability, n_mul issues per cycle
    jobs = sorted((max(rdA[k], rdB[c][k]), c) for c in range(R) for k in range(K))
    used = {}
    ready = [[] for _ in range(R)]
    for t, c in jobs:
        while used.get(t, 0) >= n_mul:
            t += 1
        used[t] = used.get(t, 0) + 1
        ready[c].append(t + L_mul + 1)
    # reduction: per copy a heap of ready partial sums (+ the accumulator value, ready at 0)
    heaps = [sorted(r + [0]) for r in ready]
    for h in heaps:
        heapq.heapify(h)
    add_used = {}
    finish = 0
    while any(len(h) > 1 for h in heaps):
        # pick the copy whose two earliest operands are ready first
        best = min((max(h[0], sorted(h)[1]), c) for c, h in enumerate(heaps) if len(h) > 1)
        t, c = best
        a = heapq.heappop(heaps[c]); b = heapq.heappop(heaps[c])
        t = max(a, b)
        while add_used.get(t, 0) >= n_add:
            t += 1
        add_used[t] = add_used.get(t, 0) + 1
        heapq.heappush(heaps[c], t + L_add + 1)
        finish = max(finish, t + L_add + 1)
    return finish + 1


# ---------------------------------------------------------------- rule L4
def top_loop_overhead(hs, body, child_kinds):
    """L4: per-iteration overhead of a non-pipelined loop in the top FSM, added to the
    sum of its children's latencies.
      - n calls to pipelined modules: 1 + 2n cycles (call/return states), except a
        single call of gemm_Pipeline_loop_j from loop_i: 2 cycles.
      - nested non-pipelined loops: 2 cycles (loop exit/increment) per child loop.
      - hoisted scalar reads (lA row values passed to the called module):
        ceil(words / ports) cycles.
      - +1 at a fast clock when there are calls (extra handshake register)."""
    calls = [c for c in body['children'] if c.startswith('gemm_')]
    loops = [c for c in body['children'] if not c.startswith('gemm_')]
    ovh = 0
    if calls:
        if len(calls) == 1 and calls[0] == 'gemm_Pipeline_loop_j':
            ovh += 2
        else:
            ovh += 1 + 2 * len(calls)
        if is_fast(hs) and calls[0] != 'gemm_Pipeline_loop_j':
            ovh += 1
    if loops:
        ovh += 2 + max(0, len(loops) - 1) * 1
    return ovh


# ---------------------------------------------------------------- top
def predict_latency(hs, detail=False):
    loops = hs['loops']
    mod_loop = {l['module']: l for l in loops if l['module'] != 'gemm'}
    gemm_loops = {l['loop']: l for l in loops if l['module'] == 'gemm'}
    bodies = {b['loop']: b for b in hs['latency_breakdown']['top_loop_bodies']}

    def mod_lat(m):
        l = mod_loop[m]
        if l['pipelined']:
            return pipelined_module_latency(l)
        # a loop module the tool did not pipeline: TC * body + 1 (rule L3 body)
        return l['trip_count'] * innermost_body(hs, l) + 1

    # hoisted reads: lA values used by the compute module, read once per loop_i iteration
    # if loop_i is a top loop, else once at FSM entry (rule L5)
    def hoisted_split():
        per_iter, entry = 0, 0
        top_has_i = any(name in ('loop_i',) for name in gemm_loops)
        for m in mod_loop:
            if 'copy' in m:
                continue
            for a, w in hoisted_reads(hs, m):
                c = read_cycles(hs, a, w)
                if a == 'lA' and top_has_i:
                    per_iter = max(per_iter, c)
                else:
                    entry = max(entry, c)
        return per_iter, entry

    hoist_iter, hoist_entry = hoisted_split()

    def loop_lat(name):
        l = gemm_loops[name]
        b = bodies.get(name, {'children': []})
        if not b['children']:
            it = innermost_body(hs, l)
        else:
            calls = [c for c in b['children'] if c.startswith('gemm_')]
            owners = {o['owner'] for o in hs['operators']}
            if len(calls) > 1 and all(c in owners for c in calls):
                # L4b: the called loop modules each own their fp cores -> no shared
                # resource -> the FSM starts them together; body = slowest call + n + 1
                it = max(mod_lat(c) for c in calls) + len(calls) + 1
            else:
                it = 0
                for c in b['children']:
                    it += mod_lat(c) if c.startswith('gemm_') else loop_lat(c)
                it += top_loop_overhead(hs, b, None)
            if name == 'loop_i':
                it += hoist_iter
        return l['trip_count'] * it

    total = 0
    parts = {}
    for s in hs['latency_breakdown']['stages']:
        lat = mod_lat(s['stage']) if s['kind'] == 'call' else loop_lat(s['stage'])
        parts[s['phase']] = parts.get(s['phase'], 0) + lat
        total += lat
    compute_is_call = all(s['kind'] == 'call' for s in hs['latency_breakdown']['stages']
                          if s['phase'] == 'compute')
    fsm = (5 if compute_is_call else 3) + hoist_entry
    total += fsm
    if detail:
        return total, parts, fsm
    return total
