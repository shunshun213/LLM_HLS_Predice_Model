"""Stage 2 input restriction (stage2_impl.md section 2).

`sanitize(input_hs)` returns a copy of the true HS containing ONLY the fields that
Stage 1 can plausibly predict from the HLS code and the clock. Everything that is a
report output (latencies, utilization, per-array BRAM/LUT/FF, estimated clock,
unit costs, FSM state bits, top FSM overhead) is removed. The Stage 2 model only
ever sees the output of this function.

Every kept field is listed with its justification in ALLOWED below; anything not
listed is dropped (whitelist, not blacklist).
"""
import copy

# field -> justification (why Stage 1 can predict it from code + clock)
ALLOWED = {
    'module_hierarchy': {
        'module': 'which loops become pipelined sub-modules: pipeline pragma + loop nest',
        'instantiations': 'instantiation count of each module: unroll of the enclosing loop',
        'children': 'call structure: loop nest',
        # dropped: latency (output), fsm_state_bits (output-derived, Stage 2 computes)
    },
    'loops': {
        'module': 'owner module', 'module_instantiations': 'see above',
        'loop': 'loop name', 'flattened_from': 'loop_flatten rule',
        'level': 'nest level', 'trip_count': 'N / unroll factor',
        'pipelined': 'pipeline pragma', 'pipeline_style': 'pragma',
        'ii_target': 'pragma', 'ii_achieved': 'Stage 1 II rules (recurrence / memory port / clock)',
        'pipeline_depth': 'per-loop depth (listed as allowed in stage2_impl.md section 2; equals the '
                          'report iteration latency of a pipelined loop, so Stage 1 must predict it)',
        # dropped: latency, iteration_latency (outputs), log_depth (duplicate of depth),
        #          last_failed_ii, recurrence_vars, memport_arrays (tool diagnostics)
    },
    'operators': {
        'owner': 'owner module (shared at top or inside loop module)',
        'op': 'fadd / fmul', 'impl': 'bind_op or default-impl rule',
        'latency': 'bind_op or default-latency table (op, impl, clock)',
        'count_in_module': 'sharing rule n = ceil(u / II) ...',
        'owner_instantiations': 'see module instantiations',
        'count_effective': 'count_in_module x owner_instantiations',
        # dropped: core (name string, redundant), unit_cost (use characterization table)
    },
    'aux_operators': {
        'owner': 'owner module', 'class': 'sparsemux / shift / int: follows from banking & reshape',
        'params': 'mux width / shift width: follows from banks and word width',
        'count_in_module': 'one per accessed bank group', 'count_effective': 'x instantiations',
        # dropped: core, unit_cost
    },
    'memories': {
        'owner': 'array owner', 'storage_ports': 'bind_storage / default',
        'storage_impl': 'bind_storage impl', 'banks': 'partition factor',
        'words_per_bank': 'array size / banks (reshape)', 'bits_per_word': '32 x reshape factor',
        'uniform_banks': 'derived from banks', 'in_registers': 'complete partition into registers',
        'storage_inferred_by_tool': 'RAM_1WNR replication rule (Stage 1 finding 5)',
        'total_bits': 'array size x 32',
        # dropped: core, bram18k_per_bank, bank_shapes (contain bram/lut/ff), BRAM_18K, LUT, FF, URAM
    },
    'access_pattern': {
        'banks_accessed': 'banks of the array touched per iteration: partition + unroll',
        'read_ports_per_bank_max': 'ports per bank used: storage type + accesses per iteration',
        'read_ports_total': 'idem', 'write_ports_per_bank_max': 'idem', 'write_ports_total': 'idem',
        'scalar_inputs': 'values passed as scalars (loop-invariant reads hoisted into the caller): '
                         'Stage 1 rule H, see STAGE2_MODEL.md',
        'scalar_input_bits': 'scalar_inputs x word bits',
    },
    'transforms_applied': {
        'unroll': ['loop', 'kind', 'factor'],
        'array': ['directive', 'array', 'detail'],
        'flatten': ['loop', 'flattened'],
        'balanced_expressions': 'unsafe_math reassociation applied (count of rebalanced ops)',
    },
    'timing': {'target_ns': 'clock target (input)'},
    # latency_breakdown: only the structure (stage order, kind, phase, trip counts, children)
    'latency_breakdown': {
        'stages': ['stage', 'kind', 'phase', 'trip_count'],
        'top_loop_bodies': ['loop', 'trip_count', 'children'],
    },
}


def _pick(d, keys):
    return {k: copy.deepcopy(d[k]) for k in keys if k in d}


def sanitize(hs):
    out = {}
    out['module_hierarchy'] = [_pick(m, ALLOWED['module_hierarchy']) for m in hs['module_hierarchy']]
    out['loops'] = [_pick(l, ALLOWED['loops']) for l in hs['loops']]
    out['operators'] = [_pick(o, ALLOWED['operators']) for o in hs['operators']]
    out['aux_operators'] = [_pick(o, ALLOWED['aux_operators']) for o in hs['aux_operators']]
    out['memories'] = {a: _pick(m, ALLOWED['memories']) for a, m in hs['memories'].items()}
    out['access_pattern'] = {mod: {a: _pick(v, ALLOWED['access_pattern']) for a, v in ap.items()}
                             for mod, ap in hs['access_pattern'].items()}
    ta = hs['transforms_applied']; wl = ALLOWED['transforms_applied']
    out['transforms_applied'] = {
        'unroll': [_pick(u, wl['unroll']) for u in ta.get('unroll', [])],
        'array': [_pick(u, wl['array']) for u in ta.get('array', [])],
        'flatten': [_pick(u, wl['flatten']) for u in ta.get('flatten', [])],
        'balanced_expressions': ta.get('balanced_expressions', 0),
    }
    out['timing'] = _pick(hs['timing'], ALLOWED['timing'])
    lb = hs['latency_breakdown']
    out['latency_breakdown'] = {
        'stages': [_pick(s, ALLOWED['latency_breakdown']['stages']) for s in lb['stages']],
        'top_loop_bodies': [_pick(b, ALLOWED['latency_breakdown']['top_loop_bodies'])
                            for b in lb['top_loop_bodies']],
    }
    return out


FORBIDDEN_KEYS = {'latency', 'iteration_latency', 'fsm_state_bits', 'unit_cost', 'BRAM_18K',
                  'LUT', 'FF', 'URAM', 'bram18k_per_bank', 'bank_shapes', 'est_ns',
                  'fsm_overhead', 'per_iteration_overhead', 'body_children_latency',
                  'by_phase', 'top_latency', 'log_depth', 'core'}


def assert_clean(s, path=''):
    """Fail if any forbidden key survives anywhere (operators' 'latency' = op latency is allowed)."""
    if isinstance(s, dict):
        for k, v in s.items():
            p = f'{path}.{k}'
            if k in FORBIDDEN_KEYS and not (k == 'latency' and (path.startswith('.operators')
                                                                or path.startswith('.aux_operators'))):
                raise AssertionError(f'forbidden field leaked: {p}')
            assert_clean(v, p)
    elif isinstance(s, list):
        for i, v in enumerate(s):
            assert_clean(v, path + '[]')
