"""End-to-end model:  (HLS code, clock) -> latency, LUT, FF, DSP, BRAM_18K, fits  + explanation.

    python3 predict.py design.cpp [--clock NS] [--json] [--trace]
    python3 predict.py run.tcl    [--json] [--trace]

Input: the GEMM source with inline `#pragma HLS` (the clock comes from --clock, or from a
`create_clock -period` line in the file), or the Tcl directive script. The tool settings assumed
are those of the dataset: `config_compile -pipeline_loops 0` (no automatic pipelining) and
`config_array_partition -throughput_driven off`; unsafe_math via --unsafe-math or a
`config_compile -unsafe_math_optimizations` line. Pipeline:

    stage1_parse  (pragmas -> canonical design cfg)
    stage1        (cfg -> hardware structure HS, white-box rules S/P/D/O/M/A/X/T)
    stage2        (HS -> latency and resources, white-box rules L1-L5, R1-R4)

The trace (--trace) lists, for every pragma, the HS terms it set, then every Stage 1 decision
with the rule that made it, then the Stage 2 latency and resource terms per module, so a
prediction can be followed step by step back to the code (Target_refined rules 1-2, section 6).
`confidence` lists every extrapolated or low-confidence rule that fired for this design.
"""
import argparse, json, math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)          # the package's own modules always come first
import stage1_parse, stage1, stage2, sanitize, latency, resources, features   # noqa: E402

METRICS = ('latency_cycles', 'LUT', 'FF', 'DSP', 'BRAM_18K')

# Target_refined section 6: which HS terms each pragma drives (static map, shown in the trace)
PRAGMA_TERMS = {
    'pipeline': 'module hierarchy (S1-S3, S5-S6), II_target, II_achieved = max(target, recurrence, memory), '
                'depth, operator sharing ceil(P/II) (stp) or P (flp), implied full unroll of inner loops',
    'unroll': 'products per iteration P = u*R -> operator counts, reads per bank -> II_memory, '
              'recurrence II = u*L_fadd, number of loop_k modules (unroll_j)',
    'loop_flatten': 'structure: separate loop_i / loop_j top loops instead of a flattened nest',
    'array_partition': 'banks, words/bank -> BRAM; banks touched / reads per bank -> II_memory, ports; '
                       'bank-select sparsemux; hoisting rules H2/H3',
    'array_reshape': 'bits/word, words -> BRAM; lanes per read -> II_memory; lane select / lshr (2.5 ns); '
                     'C read-modify-write port',
    'bind_storage': 'ports per bank (1P: 1, 2P/T2P: 2) -> II_memory; BRAM vs LUTRAM; no RAM_1WNR replication',
    'bind_op': 'operator impl and latency -> DSP/LUT/FF per core, depth, recurrence II',
    'clock': 'default core latencies (DEFAULT_LAT), default impls, depth offsets, lshr use at 2.5 ns',
    'unsafe_math': 'reassociation (if fadd unbound): recurrence II = L_fadd, adder tree, balanced expressions',
}


def load_cfg(path, clock=None, unsafe_math=None):
    txt = open(path).read()
    if path.endswith('.tcl') or 'set_directive_' in txt and '#pragma' not in txt:
        cfg = stage1_parse.parse_tcl(txt, clock_ns=clock, unsafe_math=unsafe_math)
    else:
        cfg = stage1_parse.parse_source(txt, clock_ns=clock, unsafe_math=unsafe_math)
    if cfg.get('clock_ns') is None:
        raise stage1_parse.UnsupportedDesign('no clock: pass --clock NS or put "create_clock -period NS" in the file')
    return cfg


_COV = None
_SUP = None


def sparse_terms(hs):
    """Stage 2 fitted terms used by this design whose quantity is non-zero in fewer than 10 of the
    510 designs (their per-unit constant rests on few samples)."""
    global _SUP
    if _SUP is None:
        _SUP = json.load(open(os.path.join(_HERE, 'term_support.json')))
    sh = sanitize.sanitize(hs)
    C = stage2.consts()
    out = set()
    for mm in sh['module_hierarchy']:
        m = mm['module']
        f = features.module_features(sh, m)
        cls = features.module_class(m)
        for cat in resources.CATS:
            for rr in ('LUT', 'FF'):
                for b, v in C.get(f'{cls}|{cat}|{rr}', {}).items():
                    n = _SUP.get(f'{cls}|{cat}|{rr}|{b}')
                    if n is not None and n < 10 and v and f.get(b, 0.0):
                        out.add(f'low confidence: Stage 2 term {cls} {cat} {rr} {b} (constant from {n} designs)')
    return sorted(out)


def coverage_flags(cfg):
    """values of the design that the 510 dataset designs never (or rarely, < 3) used: the rules
    there are extrapolations even when no low-confidence rule fires."""
    global _COV
    if _COV is None:
        _COV = json.load(open(os.path.join(_HERE, 'coverage.json')))
    out = []
    cf = {k: v for k, v in cfg.items() if not k.startswith('_')}
    cf['clock_ns'] = float(stage1.clock(cfg))                 # 3.3 / 3.333 -> 3.33
    for k, v in cf.items():
        seen = _COV['single'].get(k, {})
        n = seen.get(str(v), 0)
        if n == 0:
            out.append(f'outside the data: {k}={v} (never used by the 510 designs)')
        elif n < 3:
            out.append(f'sparse data: {k}={v} (used by {n} design{"s" if n > 1 else ""})')
    for key, seen in _COV['pair'].items():
        a, b = key.split('|')
        n = seen.get(f'{cf.get(a)}|{cf.get(b)}', 0)
        if n == 0 and _COV['single'].get(a, {}).get(str(cf.get(a))) and _COV['single'].get(b, {}).get(str(cf.get(b))):
            out.append(f'outside the data: {b}={cf.get(b)} with {a}={cf.get(a)} (combination never used)')
    return out


def _c_store_serialised(c, comp):
    """rule A (C write ports) changed the port count: >= 2 stores hit one C bank per iteration and
    a serialising condition holds (C reshaped, flp, or a ram_1p input)."""
    if not comp or not ((stage1.layout(c, 'C') and stage1.layout(c, 'C')[0] == 'reshape')
                        or c.get('pipeline_style') == 'flp' or 'ram_1p' in (c.get('A_storage'), c.get('B_storage'))):
        return False
    kind = stage1.module_kind(comp[0]['module'])
    if kind == 'loop_k':
        return False                                     # C is stored by the top, not the pipeline
    per = {}
    for a in stage1.accesses(c, 'C', kind):
        b, w = stage1.locate(c, 'C', a)
        per.setdefault(b, set()).add(w)
    return max(len(s) for s in per.values()) >= 2


def confidence(cfg, hs):
    """Extrapolated / low-confidence rules that fired (see STAGE1_MODEL.md, VALIDATION_STAGE1 s5)."""
    c = stage1.normalize(cfg)
    out = [f'extrapolated: {n}' for n in c['_norm_notes']] + [f'parser: {n}' for n in cfg.get('_notes', [])]
    out += coverage_flags(c)
    out += sparse_terms(hs)
    if c['pipeline_loop'] == 'k' and c['flatten'] != 'off' and c['C_mode'] == 'reshape' \
            and c['C_type'] == 'complete' and c['C_dim'] == 2:
        out.append('low confidence: S3 C whole-row word separates loop_i / loop_j (2 train designs)')
    if any(c[X + '_mode'] == 'reshape' and c[X + '_type'] == 'complete' and c[X + '_dim'] == 2
           and c[X + '_storage'] == 'ram_1p' for X in 'AB'):
        out.append('low confidence: S4 copy_in kept as two loops for a ram_1p whole-row word (4 train designs)')
    if c['pipeline_loop'] == 'i':
        out.append('extrapolated: S6 pipeline on loop_i (no such design in the data)')
    if any((not l['pipelined']) and l['module'] != 'gemm' and 'copy' not in l['loop'] for l in hs['loops']):
        out.append('low confidence: S5 tool drops a barely-overlapping flp pipeline (1 supporting design)')
    if c['pipeline_loop'] == 'none':
        u, La = stage1.ku(c), stage1.layout(c, 'A')
        if u == stage1.N or (stage1.reassociates(c) and u >= 2) or (La and La[2] == 2 and u >= 4) \
                or (u == stage1.N and stage1.ju(c) >= 2):
            out.append('low confidence: B-none operator counts (second fmul: 1-7 designs per branch; fadd = 2: 2/4 exact)')
    owners = {o['owner'] for o in hs['operators']}
    comp = [l for l in hs['loops'] if l['pipelined'] and l['module'] != 'gemm' and 'copy' not in l['loop']]
    if stage1.reassociates(c) and comp and c['pipeline_style'] != 'flp' \
            and stage1.module_kind(comp[0]['module']) in ('loop_k', 'loop_i_loop_j_loop_k') \
            and stage1.ku(c) > (comp[0]['ii_achieved'] or 1) \
            and len({o['impl'] for o in hs['operators'] if o['op'] == 'fadd'}) == 1:
        out.append('low confidence: O shared tree/accumulator fadd pool with u > II (2 supporting designs, both u <= II)')
    if c['pipeline_loop'] == 'k' and stage1.ku(c) == 1 and not stage1.reassociates(c) and stage1.clock(c) < 10.0 \
            and any(stage1.module_kind(l['module']) == 'loop_k' for l in comp):
        out.append('low confidence: D u=1 loop_k depth offset below 10 ns (8 modules in 3 train designs, 0 extra)')
    if _c_store_serialised(c, comp):
        out.append('low confidence: A C-store serialised to 1 write port (17 train designs, 0 extra)')
    for X, kd in (('A', 2), ('B', 1)):
        L = stage1.layout(c, X)
        if L and L[0] == 'reshape' and L[2] != kd and L[1] != 'complete' and 32 * L[3] > 64 \
                and any(stage1.hoisted(c, X, stage1.module_kind(l['module'])) for l in comp):
            out.append(f'low confidence: R4 hoisted l{X} words reshaped along the non-k index, W = {32 * L[3]} '
                       f'(2-row slice; 3 designs with W > 64, one of them not separable)')
    if sum(1 for o in owners if 'loop_k' in o) > 1:
        out.append('low confidence: O2-par independent loop_k copies (2 supporting designs)')
    if any(a['class'] == 'shift' and a['params'].get('op', 'lshr') == 'lshr' for a in hs['aux_operators']):
        out.append('low confidence: X-SH lshr lane slicer at 2.5 ns (10 train designs, 0 extra; shift cores overall 7/11 exact)')
    if any(a['class'] == 'shift' and a['params'].get('op') == 'shl' for a in hs['aux_operators']):
        out.append('low confidence: X-SHL pipelined shl lane writer at 2.5 ns (11 supporting designs, train only)')
    base = 2 if stage1.clock(c) >= 5.0 else 4
    if any('copy' in l['loop'] and l['pipelined'] and l['pipeline_depth'] != base for l in hs['loops']):
        out.append('low confidence: D-copy non-base copy-loop depth branch (train-only support; +-1-3 cycles)')
    if c['unsafe_math'] and not stage1.reassociates(c):
        out.append('note: unsafe_math has no effect, the fadd is bound by BIND_OP')
    return out


def trace(cfg, hs, res):
    c = stage1.normalize(cfg)
    T = []
    w = T.append
    w('== 1. pragmas -> design description')
    keys = [k for k in c if not k.startswith('_') and c[k] not in (None, 'none', 'auto', False, 'default')]
    w('   ' + ', '.join(f'{k}={c[k]}' for k in keys))
    used = {'pipeline' if c['pipeline_loop'] != 'none' else None,
            'unroll' if (c['unroll_k'] != 1 or c['unroll_j'] != 1) else None,
            'loop_flatten' if c['flatten'] == 'off' else None,
            'array_partition' if any(c[X + '_mode'] == 'partition' for X in 'ABC') else None,
            'array_reshape' if any(c[X + '_mode'] == 'reshape' for X in 'ABC') else None,
            'bind_storage' if any(c[X + '_storage'] != 'default' for X in 'AB') else None,
            'bind_op' if any(c[o + '_impl'] not in (None, 'auto') for o in ('fmul', 'fadd')) else None,
            'clock', 'unsafe_math' if c['unsafe_math'] else None} - {None}
    for p in sorted(used):
        w(f'   {p:16s} -> {PRAGMA_TERMS[p]}')
    for n in c['_norm_notes'] + cfg.get('_notes', []):
        w(f'   note: {n}')
    w('== 2. Stage 1: hardware structure')
    w('   modules: ' + ', '.join(m['module'] for m in hs['module_hierarchy']))
    for l in hs['loops']:
        if l['module'] == 'gemm' or 'copy' in l['loop']:
            if l['pipelined']:
                w(f"   {l['loop']}: TC {l['trip_count']}, II 1, depth {l['pipeline_depth']} (D-copy)")
            continue
        if not l['pipelined']:
            w(f"   {l['module']}/{l['loop']}: TC {l['trip_count']}, not pipelined (S5)")
            continue
        p = stage1.ii_plan(c, l['module'])
        w(f"   {l['module']}/{l['loop']}: TC {l['trip_count']}, II_achieved {l['ii_achieved']} = max(target "
          f"{l['ii_target']}, recurrence {p['ii_rec']}, memory {p['ii_mem']}) [limiter: {p['limiter']}], "
          f"depth {l['pipeline_depth']}")
        for X, pl in p['plans'].items():
            w(f"      l{X}: {pl['reads']} reads/bank/iteration over {pl['ports']} ports, {pl['banks_touched']} banks wired"
              f"{', RAM_1WNR replication' if pl['wnr'] else ''}")
        kind = stage1.module_kind(l['module'])
        for X in 'AB':
            h = stage1.hoisted(c, X, kind)
            if h:
                w(f'      l{X}: hoisted to the caller as scalar inputs (rule {h})')
    comp = [l for l in hs['loops'] if l['pipelined'] and l['module'] != 'gemm' and 'copy' not in l['loop']]
    if comp:
        l0 = comp[0]
        kind = stage1.module_kind(l0['module'])
        u = stage1.ku(c) if kind in ('loop_k', 'loop_i_loop_j_loop_k') else stage1.N
        R = stage1.ju(c) if kind in stage1.JK else 1
        II = l0['ii_achieved']
        sh = (lambda n: f'{n}') if c['pipeline_style'] == 'flp' else (lambda n: f'ceil({n}/{II})')
        shared = any(o['owner'] == 'gemm' for o in hs['operators'])
        per = ((f' (the {len(comp)} loop_k copies share one pool of cores in gemm: the counts are for the whole pool)'
                if shared else f' per loop_k copy ({len(comp)} copies, each owns its cores)')
               if len(comp) > 1 else '')
        w(f'   operators{per}: P = u*R = {u}*{R} = {u * R} products per iteration, II = {II}, '
          + ('flp: no sharing' if c['pipeline_style'] == 'flp' else 'stp: cores shared over II slots'))
        w(f'      fmul n = {sh(u * R)}')
        if stage1.reassociates(c) and kind in ('loop_k', 'loop_i_loop_j_loop_k') and u > 1:
            ir, it = stage1.fadd_rec(c)[0], stage1.op_impl_lat(c, 'fadd', False)[0]
            if ir == it:
                w(f'      fadd n = {sh(u)} (unsafe_math: tree + accumulator share one pool, deviation (a))')
            else:
                w(f'      fadd n = {sh(u - 1)} tree ({it}) + 1 accumulator ({ir})')
        elif stage1.reassociates(c) and kind in stage1.JK:
            w(f'      fadd n = {sh(f"63*{R}")} (unsafe_math: 64-term tree per j copy, deviation (b))')
        else:
            w(f'      fadd n = {sh(u * R)}')
    elif c['pipeline_loop'] == 'none':
        w('   operators: non-pipelined body, cores reused across cycles (rule B-none)')
    for o in hs['operators']:
        w(f"   core {o['op']} {o['impl']} lat {o['latency']} x{o['count_effective']} in {o['owner']}")
    for a in hs['aux_operators']:
        pr = a['params']
        w(f"   aux {a['class']} {pr.get('op', '')} ({pr.get('inputs') or pr.get('width')}) x{a['count_effective']} in {a['owner']}")
    for a, m in hs['memories'].items():
        w(f"   {a}: {m['banks']} bank(s) x {m['words_per_bank'][0]} words x {m['bits_per_word'][0]} bit, "
          f"ports {m['storage_ports']}, impl {m['storage_impl']}")
    w('== 3. Stage 2: latency')
    tot, parts, fsm = latency.predict_latency(sanitize.sanitize(hs), detail=True)
    for l in sanitize.sanitize(hs)['loops']:
        if l['pipelined'] and l['module'] != 'gemm':
            w(f"   L1 {l['module']}: (TC-1)*II + D + 1 = ({l['trip_count']}-1)*{l['ii_achieved']} + "
              f"{l['pipeline_depth']} + 1 = {latency.pipelined_module_latency(l)}")
        elif l['module'] == 'gemm' and l['loop'].startswith('loop'):
            w(f"   L2 top loop {l['loop']}: TC {l['trip_count']} x (children + per-iteration overhead L4)")
    for k, v in parts.items():
        w(f'   {k}: {v} cycles')
    w(f'   top FSM + hoisted reads: {fsm} cycles  -> total {tot}')
    w('== 4. Stage 2: resources (per module: cores + Expression/Multiplexer/Register; memories)')
    sh = sanitize.sanitize(hs)
    for mm in sh['module_hierarchy']:
        m = mm['module']
        cc = resources.core_cost(sh, m)
        cats = {cat: stage2.category_model(sh, m, cat) for cat in resources.CATS}
        lut = cc['LUT'] + sum(v['LUT'] for v in cats.values())
        ff = cc['FF'] + sum(v['FF'] for v in cats.values())
        w(f"   {m} (x{mm['instantiations']}): cores DSP {cc['DSP']} LUT {cc['LUT']:.0f} FF {cc['FF']:.0f}; "
          + '; '.join(f"{k} LUT {v['LUT']:.0f} FF {v['FF']:.0f}" for k, v in cats.items())
          + f' -> LUT {lut:.0f} FF {ff:.0f}')
        # R3 terms: structural quantity x per-unit constant (CONSTANTS.md, meanings in STAGE2_MODEL.md)
        f = features.module_features(sh, m)
        C = stage2.consts()
        cls = features.module_class(m)
        for cat in resources.CATS:
            for rr in ('LUT', 'FF'):
                terms = [(b, f.get(b, 0.0), v) for b, v in C.get(f'{cls}|{cat}|{rr}', {}).items()
                         if f.get(b, 0.0) and v]
                if terms:
                    w(f'      {cat} {rr} = ' + ' + '.join(f'{b} {q:g} x {v:.3g}' for b, q, v in terms)
                      + f' = {sum(q * v for _, q, v in terms):.0f}')
    for a in sh['memories']:
        mc = resources.memory_cost(sh, a)
        w(f"   memory {a}: " + ', '.join(f'{k} {v:.0f}' for k, v in mc.items() if v))
    w('   total: ' + ', '.join(f'{k} {res[k]:.0f}' for k in METRICS[1:]) + f", fits {res['fits']}")
    return T


def predict(cfg, with_trace=False):
    hs = stage1.predict_hs(cfg)
    res = stage2.predict(hs)
    out = {k: (round(res[k]) if k != 'latency_cycles' else int(res[k])) for k in METRICS}
    out['fits'] = res['fits']
    out['confidence'] = confidence(cfg, hs) or ['all rules used are supported by the dataset']
    # rules resting on 1-2 designs that change the structure: also give the prediction without them
    fired = [r for r, hit in (('S5', any((not l['pipelined']) and l['module'] != 'gemm' and 'copy' not in l['loop']
                                         for l in hs['loops'])),
                              ('O2par', sum(1 for o in {o['owner'] for o in hs['operators']} if 'loop_k' in o) > 1))
             if hit]
    if fired:
        alt = stage2.predict(stage1.predict_hs(dict(cfg, _disable=tuple(fired))))
        out['without_low_support_rules'] = dict(rules=fired, **{k: (round(alt[k]) if k != 'latency_cycles'
                                                                  else int(alt[k])) for k in METRICS},
                                                fits=alt['fits'])
    if with_trace:
        out['trace'] = trace(cfg, hs, res)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('design', help='GEMM .cpp with #pragma HLS, or a .tcl directive script')
    ap.add_argument('--clock', type=float, help='clock period in ns (10 / 5 / 3.33 / 2.5)')
    ap.add_argument('--unsafe-math', dest='unsafe_math', action='store_true', default=None,
                    help='config_compile -unsafe_math_optimizations (else read from the file)')
    ap.add_argument('--no-unsafe-math', dest='unsafe_math', action='store_false',
                    help='no unsafe_math, whatever the file says')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--trace', action='store_true')
    a = ap.parse_args()
    try:
        out = predict(load_cfg(a.design, a.clock, a.unsafe_math), a.trace)
    except stage1_parse.UnsupportedDesign as e:
        print(f'error: design outside the modeled space: {e}', file=sys.stderr)
        sys.exit(2)
    except (OSError, UnicodeDecodeError) as e:
        print(f'error: cannot read the design file: {e}', file=sys.stderr)
        sys.exit(3)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    for k in METRICS:
        print(f'{k:15s} {out[k]}')
    print(f"{'fits':15s} {out['fits']}")
    for c in out['confidence']:
        print('confidence:', c)
    if 'without_low_support_rules' in out:
        w = out['without_low_support_rules']
        print(f"range: without rule(s) {', '.join(w['rules'])}: " + ', '.join(f'{k} {w[k]}' for k in METRICS))
    for line in out.get('trace', []):
        print(line)


if __name__ == '__main__':
    main()
