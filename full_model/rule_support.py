"""Support of the data-derived Stage-1 rules: for each rule, how many designs (train / extra)
exercise each branch and how many of those the rule gets exactly right."""
import json, collections, stage1, stage1_parse as P, sanitize, evaluate
S2 = {r['sample']: r for r in evaluate.load()}
rows = [json.loads(l) for l in open(evaluate.DATA.replace('stage2_dataset', 'stage1_dataset'))]
sup = collections.defaultdict(lambda: collections.Counter())
def rec(rule, branch, src, okk):
    sup[(rule, branch)][src] += 1; sup[(rule, branch)][src + '_ok'] += bool(okk)
for r in rows:
    cfg = P.parse_tcl(r['inputs']['tcl']); src = r['source']
    p = sanitize.sanitize(stage1.predict_hs(cfg)); t = sanitize.sanitize(S2[r['sample']]['input_hs'])
    u, R = stage1.ku(cfg), stage1.ju(cfg)
    # B-none
    if cfg['pipeline_loop'] == 'none':
        nf = lambda h: sum(o['count_effective'] for o in h['operators'] if o['op'] == 'fmul')
        na = lambda h: sum(o['count_effective'] for o in h['operators'] if o['op'] == 'fadd')
        LA = stage1.layout(cfg, 'A'); re = stage1.reassociates(cfg)
        br = 'u=64' if u == 64 else ('reassoc,u>=2' if re and u >= 2 else ('A split on k,u>=4' if LA and LA[2] == 2 and u >= 4 else 'default 1'))
        rec('B-none fmul', br, src, nf(p) == nf(t))
        rec('B-none fadd', '2 (u=64,R>=2)' if (u == 64 and R >= 2) else '1', src, na(p) == na(t))
    # X-SH
    sh = lambda h: sorted((a['owner'], a['count_effective']) for a in h['aux_operators'] if a['class'] == 'shift')
    if cfg['clock_ns'] == 2.5 and any(stage1.layout(cfg, X) and stage1.layout(cfg, X)[0] == 'reshape' for X in 'ABC'):
        rec('X-SH', 'fires' if sh(p) else 'no shift', src, sh(p) == sh(t))
    # D-copy
    for l in t['loops']:
        if 'copy' in l['loop'] and l['pipelined']:
            q = [x for x in p['loops'] if x['loop'] == l['loop']]
            d = stage1.copy_depth(cfg, l['loop'])
            base = 2 if stage1.clock(cfg) >= 5 else 4
            rec('D-copy', ('in' if 'in' in l['loop'] else 'out') + (' base' if d == base else ' modified'), src,
                bool(q) and q[0]['pipeline_depth'] == l['pipeline_depth'])
    # u=1 depth offsets
    for l in t['loops']:
        if l['pipelined'] and 'copy' not in l['loop'] and u == 1 and cfg['pipeline_loop'] == 'k' and not stage1.reassociates(cfg):
            q = [x for x in p['loops'] if x['module'] == l['module']]
            kind = stage1.module_kind(l['module'])
            La = stage1.fadd_rec(cfg)[1]; Lm = stage1.op_impl_lat(cfg, 'fmul', False)[1]
            br = kind[:6] + (' La>=Lm' if La >= Lm else ' La<Lm') + (' T<10' if stage1.clock(cfg) < 10 else ' T=10')
            rec('D u=1 offset', br, src, bool(q) and q[0]['pipeline_depth'] == l['pipeline_depth'])
    # C write ports
    for m, ap in t['access_pattern'].items():
        v = ap.get('lC'); q = p['access_pattern'].get(m, {}).get('lC')
        if v and v['write_ports_per_bank_max'] and R > 1:
            L = stage1.layout(cfg, 'C')
            ser = (L and L[0] == 'reshape') or cfg.get('pipeline_style') == 'flp' or 'ram_1p' in (cfg.get('A_storage'), cfg.get('B_storage'))
            rec('A C write ports (R>1)', 'serialised to 1' if ser else 'min(stores,2)', src,
                bool(q) and q['write_ports_per_bank_max'] == v['write_ports_per_bank_max'])
    # H3
    for m, ap in t['access_pattern'].items():
        kind = stage1.module_kind(m) if m != 'gemm' else None
        if kind in ('loop_k', 'loop_i_loop_j_loop_k'):
            for X in 'AB':
                if stage1.hoisted(cfg, X, kind) == 'H3':
                    rec('A H3 hoist', X, src, p['access_pattern'].get(m, {}).get('l' + X) == ap.get('l' + X))
    # O2-par
    comp = [m for m in p['latency_breakdown']['top_loop_bodies'][0]['children']] if p['latency_breakdown']['top_loop_bodies'] else []
    if cfg['pipeline_loop'] == 'k' and R > 1:
        own = lambda h: sorted({o['owner'] for o in h['operators']})
        par = stage1.independent_copies(cfg, [m for m in stage1.structure(cfg)['modules'] if 'loop_k' in m])
        rec('O2-par', 'parallel (per-module cores)' if par else 'shared (gemm)', src, own(p) == own(t))
    # fadd default under explicit II target (O1 recurrence binding)
    if cfg['pipeline_loop'] == 'k' and cfg.get('fadd_impl') in (None, 'auto'):
        ti = [o['impl'] for o in t['operators'] if o['op'] == 'fadd']
        pi = [o['impl'] for o in p['operators'] if o['op'] == 'fadd']
        a, b = stage1.DEFAULT_LAT[('fadd', 'fabric')][stage1.clock(cfg)], stage1.DEFAULT_LAT[('fadd', 'fulldsp')][stage1.clock(cfg)]
        f = 1 if stage1.reassociates(cfg) else u
        rec('O1 fadd default (k pipe)', 'recurrence binds' if (cfg.get('pipeline_ii') or 1) < f * max(a, b) else 'not binding', src, sorted(pi) == sorted(ti))
print('| rule | branch | train ok/n | extra ok/n |'); print('|---|---|---|---|')
for (rule, br), c in sorted(sup.items()):
    print(f"| {rule} | {br} | {c['train_ok']}/{c['train']} | {c['extra_ok']}/{c['extra']} |")
# S5 (dropped flp pipeline) and X-SHL (pipelined shl) support, printed after the table
s5 = collections.Counter(); shl = collections.Counter()
for r in rows:
    cfg = P.parse_tcl(r['inputs']['tcl']); src = r['source']
    hs = stage1.predict_hs(cfg); t = S2[r['sample']]['input_hs']
    dp = any(not l['pipelined'] and l['module'] != 'gemm' and 'copy' not in l['loop'] for l in hs['loops'])
    dt = any(not l['pipelined'] and l['module'] != 'gemm' and 'copy' not in l['loop'] for l in t['loops'])
    if dp or dt:
        s5[src] += 1; s5[src + '_ok'] += dp == dt
    cnt = lambda h: sorted((a['owner'], a['count_effective']) for a in h['aux_operators'] if a['params'].get('op') == 'shl')
    if cnt(hs) or cnt(t):
        shl[src] += 1; shl[src + '_ok'] += cnt(hs) == cnt(t)
print(f"| S5 dropped flp pipeline | fires or tool drops | {s5['train_ok']}/{s5['train']} | {s5['extra_ok']}/{s5['extra']} |")
print(f"| X-SHL pipelined shl (2.5 ns) | designs with shl | {shl['train_ok']}/{shl['train']} | {shl['extra_ok']}/{shl['extra']} |")
