"""Robustness check: random pragma sets over the whole Target design space (including combinations
absent from the dataset) are rendered as a Tcl directive script AND as a C++ source with inline
pragmas, parsed back through the real front end (stage1_parse), and must (1) round-trip to the
same design, (2) give a finite, positive prediction with no exception."""
import random, collections, math, sys, os
import stage1, stage2, stage1_parse

BASE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemm_base.cpp')).read()


def render_tcl(c):
    L = [f"create_clock -period {c['clock_ns']}"]
    if c['unsafe_math']:
        L.append('config_compile -pipeline_loops 0 -unsafe_math_optimizations')
    if c['pipeline_loop'] != 'none':
        L.append(f"set_directive_pipeline -II {c['pipeline_ii'] or 1} -style {c['pipeline_style']} \"gemm/loop_{c['pipeline_loop']}\"")
    for x in ('k', 'j'):
        u = c['unroll_' + x]
        if u != 1:
            L.append(f'set_directive_unroll "gemm/loop_{x}"' + ('' if u == 'full' else f' -factor {u}'))
    if c['flatten'] == 'off':
        L.append('set_directive_loop_flatten -off "gemm/loop_j"')
    for X in 'ABC':
        if c[X + '_mode'] != 'none':
            f = f" -factor {c[X + '_factor']}" if c[X + '_type'] != 'complete' else ''
            L.append(f"set_directive_array_{c[X + '_mode']} -type {c[X + '_type']}{f} -dim {c[X + '_dim']} \"gemm\" l{X}")
        if X != 'C' and c[X + '_storage'] != 'default':
            L.append(f"set_directive_bind_storage -type {c[X + '_storage']} -impl {c[X + '_storage_impl'] or 'auto'} \"gemm\" l{X}")
    for op, var in (('fmul', 'prod'), ('fadd', 'acc')):
        if c[op + '_impl'] not in (None, 'auto'):
            lat = f" -latency {c[op + '_latency']}" if c[op + '_latency'] not in (None, 'auto') else ''
            L.append(f'set_directive_bind_op -op {op} -impl {c[op + "_impl"]}{lat} "gemm/loop_k" {var}')
    return '\n'.join(L) + '\n'


def render_source(c):
    s = BASE
    def at(anchor, line):
        nonlocal s
        s = s.replace(anchor, anchor + '\n' + line, 1)
    K = "loop_k:\n            for (int k = 0; k < N; k++) {"
    J = "loop_j:\n        for (int j = 0; j < N; j++) {"
    I = "loop_i:\n    for (int i = 0; i < N; i++) {"
    A = "    float lA[N][N], lB[N][N], lC[N][N];"
    if c['pipeline_loop'] != 'none':
        at({'k': K, 'j': J, 'i': I}[c['pipeline_loop']],
           f"#pragma HLS PIPELINE II={c['pipeline_ii'] or 1} style={c['pipeline_style']}")
    if c['unroll_k'] != 1:
        at(K, '#pragma HLS UNROLL' + ('' if c['unroll_k'] == 'full' else f" factor={c['unroll_k']}"))
    if c['unroll_j'] != 1:
        at(J, '#pragma HLS UNROLL' + ('' if c['unroll_j'] == 'full' else f" factor={c['unroll_j']}"))
    if c['flatten'] == 'off':
        at(J, '#pragma HLS LOOP_FLATTEN off')
    for X in 'ABC':
        if c[X + '_mode'] != 'none':
            f = f" factor={c[X + '_factor']}" if c[X + '_type'] != 'complete' else ''
            at(A, f"#pragma HLS ARRAY_{c[X + '_mode'].upper()} variable=l{X} type={c[X + '_type']}{f} dim={c[X + '_dim']}")
        if X != 'C' and c[X + '_storage'] != 'default':
            at(A, f"#pragma HLS BIND_STORAGE variable=l{X} type={c[X + '_storage']} impl={c[X + '_storage_impl'] or 'auto'}")
    for op, var in (('fmul', 'prod'), ('fadd', 'acc')):
        if c[op + '_impl'] not in (None, 'auto'):
            lat = f" latency={c[op + '_latency']}" if c[op + '_latency'] not in (None, 'auto') else ''
            at(K, f"#pragma HLS BIND_OP variable={var} op={op} impl={c[op + '_impl']}{lat}")
    return s

def rand_cfg(rng):
    cfg = {'pipeline_loop': rng.choice(['none', 'k', 'j', 'i']), 'pipeline_ii': rng.choice([None, 1, 2, 3, 4, 8, 16]),
           'pipeline_style': rng.choice(['stp', 'flp']),
           'unroll_k': rng.choice([1, 2, 3, 4, 5, 8, 16, 32, 48, 64, 'full']),
           'unroll_j': rng.choice([1, 2, 3, 4, 8, 16, 'full']), 'flatten': rng.choice(['auto', 'off']),
           'clock_ns': rng.choice([10.0, 5.0, 3.33, 2.5]), 'unsafe_math': rng.random() < .3}
    for X in 'ABC':
        m = rng.choice(['none', 'partition', 'reshape']); cfg[X + '_mode'] = m
        t = rng.choice(['block', 'cyclic', 'complete']) if m != 'none' else None
        cfg[X + '_type'] = t
        cfg[X + '_factor'] = rng.choice([2, 4, 8, 16, 32, 64]) if m != 'none' and t != 'complete' else None
        cfg[X + '_dim'] = rng.choice([1, 2]) if m != 'none' else None
        if X != 'C':
            cfg[X + '_storage'] = rng.choice(['default', 'ram_1p', 'ram_2p', 'ram_t2p'])
            cfg[X + '_storage_impl'] = rng.choice(['auto', 'bram', 'lutram']) if cfg[X + '_storage'] != 'default' else None
    for op, impls in (('fmul', ['fabric', 'meddsp', 'fulldsp', 'maxdsp']), ('fadd', ['fabric', 'fulldsp'])):
        cfg[op + '_impl'] = rng.choice([None, 'auto'] + impls)
        cfg[op + '_latency'] = rng.choice([None, 'auto', 1, 2, 3, 4, 6])
        if cfg[op + '_impl'] in (None, 'auto'):
            cfg[op + '_latency'] = None
    return cfg

def intended(cfg):
    """the design the random config means, in the parser's canonical form (independent of the
    parser code, so a bug shared by the Tcl and source paths is caught)."""
    c = dict(stage1_parse.DEFAULT, **cfg)
    for x in ('k', 'j'):
        if c['unroll_' + x] not in ('full', 1) and int(c['unroll_' + x]) >= 64:
            c['unroll_' + x] = 'full'
    for X in 'ABC':
        if c[X + '_mode'] != 'none' and c[X + '_factor'] == 64:
            c[X + '_type'] = 'complete'
        if c[X + '_type'] == 'complete':
            c[X + '_factor'] = None
    for op in ('fmul', 'fadd'):
        if c[op + '_impl'] in (None, 'auto'):
            c[op + '_impl'], c[op + '_latency'] = 'auto', None
        elif c[op + '_latency'] in (None, 'auto'):
            c[op + '_latency'] = 'auto'
    if c['pipeline_loop'] == 'none':
        c['pipeline_ii'], c['pipeline_style'] = None, None
    else:
        c['pipeline_ii'] = c['pipeline_ii'] or 1
    if c['pipeline_loop'] in ('j', 'i'):
        c['unroll_k'] = 'full'
    if c['pipeline_loop'] == 'i':
        c['unroll_j'] = 'full'
    return c


def main(n=5000, seed=1):
    rng = random.Random(seed); bad = collections.Counter(); ex = {}
    for _ in range(n):
        cfg = rand_cfg(rng)
        try:
            a = stage1_parse.parse_tcl(render_tcl(cfg))
            b = stage1_parse.parse_source(render_source(cfg), clock_ns=cfg['clock_ns'], unsafe_math=cfg['unsafe_math'])
            ka = {k: v for k, v in a.items() if k != '_notes'}
            kb = {k: v for k, v in b.items() if k != '_notes'}
            if ka != kb:
                raise ValueError('Tcl and source renderings parse differently: '
                                 + str({k: (ka[k], kb[k]) for k in ka if ka[k] != kb[k]}))
            ic = intended(cfg)
            if any(ka[k] != ic[k] for k in ka):
                raise ValueError('parsed design differs from the intended one: '
                                 + str({k: (ka[k], ic[k]) for k in ka if ka[k] != ic[k]}))
            p = stage2.predict(stage1.predict_hs(a))
            vals = [p[m] for m in ('latency_cycles', 'LUT', 'FF', 'DSP', 'BRAM_18K')]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in vals) or p['latency_cycles'] <= 0:
                raise ValueError(f'non-finite or negative output {vals}')
            bad['ok'] += 1
        except Exception as e:
            k = (cfg['pipeline_loop'], type(e).__name__, str(e)[:70]); bad[k] += 1; ex.setdefault(k, cfg)
    for k, v in bad.most_common(): print(v, k)
    return bad, ex

if __name__ == '__main__':
    bad, ex = main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    sys.exit(0 if bad['ok'] == n else 1)
