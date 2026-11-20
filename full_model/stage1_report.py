"""Stage 1 validation report (Target_refined / stage2_impl Step 4-5).

Input path is the real one: HLS source with pragmas (or the Tcl directives when the design has
no source pragmas) + clock -> stage1_parse -> stage1.predict_hs -> sanitized HS -> stage2.predict.

Reports
  1. stage-wise exact-match rates of the predicted HS against the tool's HS (sanitized fields),
     train / extra, and per train group;
  2. end-to-end Judgement A (latency <= 10 %, resources <= 20 %), Judgement B (Spearman / Kendall)
     and fits accuracy, with the true-HS ceiling (Stage 2 on the tool's HS) next to it;
  3. LOGO: Stage 2 constants refit without group g, group g predicted from the Stage-1 HS.
Writes reports/VALIDATION_STAGE1.md (and prints it)."""
import json, collections, math, sys, os
from scipy.stats import spearmanr, kendalltau
import stage1, stage1_parse as P, stage2, sanitize, evaluate, fit

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = evaluate.DATA.replace('stage2_dataset.jsonl', 'stage1_dataset.jsonl')
rows = [json.loads(l) for l in open(S1)]
S2 = {r['sample']: r for r in evaluate.load()}
M = ['latency_cycles', 'LUT', 'FF', 'DSP', 'BRAM_18K']
TOL = evaluate.TOL


def cfg_of(r):
    src = r['inputs'].get('source_with_pragmas')
    cfg = P.parse_source(src) if src else None
    if cfg is None or cfg.get('clock_ns') is None:
        cfg = P.parse_tcl(r['inputs']['tcl'])
    return cfg


def ms(items):
    return sorted(json.dumps(x, sort_keys=True) for x in items)


def canon(x):
    """order-insensitive canonical form of a JSON value."""
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    if isinstance(x, list):
        return sorted((canon(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    return x


def sections(p, t):
    """per-section exact match of two sanitized HS -> dict name -> (hits, total)."""
    out = {}
    def put(k, ok, n=1):
        a, b = out.get(k, (0, 0)); out[k] = (a + ok, b + n)
    hier = lambda h: ms(dict(module=m['module'], children=sorted(m['children'])) for m in h['module_hierarchy'])
    key = ('module', 'loop', 'flattened_from', 'trip_count', 'pipelined', 'level')
    lps = lambda h: ms({k: l.get(k) for k in key} for l in h['loops'])
    put('S  structure (modules + loops)', hier(p) == hier(t) and lps(p) == lps(t))
    pl = {(l['module'], l['loop']): l for l in p['loops']}
    for l in t['loops']:
        q = pl.get((l['module'], l['loop']))
        if not l['pipelined']:
            continue
        if 'copy' in l['loop']:
            put('D  copy-loop depth exact', bool(q) and q['pipeline_depth'] == l['pipeline_depth'])
            continue
        put('P  II_achieved exact', bool(q) and q['ii_achieved'] == l['ii_achieved'])
        put('D  compute depth exact', bool(q) and q['pipeline_depth'] == l['pipeline_depth'])
        put('D  compute depth within 10 %', bool(q) and abs(q['pipeline_depth'] - l['pipeline_depth'])
            <= 0.1 * l['pipeline_depth'])
    put('O  fp operators exact', ms(p['operators']) == ms(t['operators']))
    put('M  memories exact', json.dumps(p['memories'], sort_keys=True) == json.dumps(t['memories'], sort_keys=True))
    put('A  access pattern exact', json.dumps(p['access_pattern'], sort_keys=True) == json.dumps(t['access_pattern'], sort_keys=True))
    put('X  aux cores exact', ms(p['aux_operators']) == ms(t['aux_operators']))
    tr = lambda h: (ms(h['transforms_applied']['unroll']), ms(h['transforms_applied']['array']),
                    ms(h['transforms_applied']['flatten']), h['transforms_applied']['balanced_expressions'])
    put('T  transforms exact', tr(p) == tr(t))
    lb = lambda h: (ms(h['latency_breakdown']['stages']),
                    ms(dict(b, children=sorted(b['children'])) for b in h['latency_breakdown']['top_loop_bodies']))
    put('L  latency_breakdown exact', lb(p) == lb(t))
    # per-field check of the loop attributes not covered above
    lk = ('pipeline_style', 'ii_target', 'module_instantiations')
    put('S  loop attributes (style, II target, instances)',
        ms({k: l.get(k) for k in key + lk} for l in p['loops']) == ms({k: l.get(k) for k in key + lk} for l in t['loops']))
    # whole HS = equality of the full sanitized JSON, order-insensitive in every list
    # every list is sorted on purpose: list order carries no meaning and Stage 2 does not depend
    # on it (identical Stage-2 outputs on all matched designs)
    put('*  whole HS exact (full sanitized JSON)', canon(p) == canon(t))
    return out


def ok(m, p, t):
    return evaluate.ape(p, t) <= TOL[m] + 1e-12


r_by = {r['sample']: r for r in rows}


def main():
    recs = []
    for r in rows:
        cfg = cfg_of(r)
        hs = sanitize.sanitize(stage1.predict_hs(cfg))
        th = sanitize.sanitize(S2[r['sample']]['input_hs'])
        recs.append(dict(sample=r['sample'], suite=r['suite'], source=r['source'], hs=hs,
                         sec=sections(hs, th), t=S2[r['sample']]['targets'],
                         p=stage2.predict(hs), q=stage2.predict(th)))
    L = []
    w = L.append
    w('# Stage 1 validation (HLS code + clock -> HS -> Stage 2)\n')
    w('Generated by `stage1_report.py`. Input = source pragmas (Tcl directives when the source has '
      'none) + clock; no tool output is read on the prediction path.\n')
    w('**Scope of these numbers.** The Stage-1 rules were derived by inspecting all 510 designs '
      '(train and extra). No Stage-1 number below is out-of-sample: "extra" is the development set '
      'on which most rules (core latencies, depth offsets) were characterized, and "train" was also '
      'inspected. LOGO (section 3) holds out a group only from the Stage-2 constant fit; the Stage-1 '
      'rules are in-sample for every group. Section 5 lists how many designs support each '
      'data-derived rule branch; branches with few supporting designs are low-confidence.\n')
    # ---- 1. stage-wise exact match
    w('## 1. Stage-wise exact match of the predicted HS\n')
    names = sorted({k for r in recs for k in r['sec']})
    w('| section | train | extra (development set) |'); w('|---|---|---|')
    for k in names:
        cells = []
        for src in ('train', 'extra'):
            a = sum(r['sec'].get(k, (0, 0))[0] for r in recs if r['source'] == src)
            b = sum(r['sec'].get(k, (0, 0))[1] for r in recs if r['source'] == src)
            cells.append(f'{a}/{b} = {100 * a / b:.1f} %' if b else '-')
        w(f'| {k} | {cells[0]} | {cells[1]} |')
    w('\nPer train group (exact-match %):\n')
    short = ['S', 'P', 'D  compute depth within', 'O', 'M', 'A', 'X', '*']
    hdr = ['S', 'P', 'D±10%', 'O', 'M', 'A', 'X', 'HS']
    w('| group | n | ' + ' | '.join(hdr) + ' |'); w('|---|---|' + '---|' * len(hdr))
    groups = sorted({r['suite'] for r in recs if r['source'] == 'train'})
    for g in groups:
        R = [r for r in recs if r['suite'] == g]
        cells = []
        for s in short:
            ks = [k for k in names if k.startswith(s)]
            a = sum(r['sec'].get(k, (0, 0))[0] for r in R for k in ks)
            b = sum(r['sec'].get(k, (0, 0))[1] for r in R for k in ks)
            cells.append(f'{100 * a / b:.0f}' if b else '-')
        w(f'| {g} | {len(R)} | ' + ' | '.join(cells) + ' |')
    # ---- 2. end to end
    w('\n## 2. End to end: Judgement A / B and fits\n')
    w('Stage-1 HS -> Stage 2 (constants fit on train, as frozen). "ceiling" = Stage 2 on the tool HS.\n')
    for src in ('train', 'extra'):
        R = [r for r in recs if r['source'] == src]
        w(f"**{src}{' (development set, rules derived here)' if src == 'extra' else ''}** (n = {len(R)})\n")
        w('| metric | pass A (Stage-1 HS) | pass A ceiling | Spearman | Kendall |'); w('|---|---|---|---|---|')
        for m in M:
            a = sum(ok(m, r['p'][m], r['t'][m]) for r in R)
            c = sum(ok(m, r['q'][m], r['t'][m]) for r in R)
            P_ = [r['p'][m] for r in R]; T_ = [r['t'][m] for r in R]
            w(f'| {m} | {100 * a / len(R):.1f} % | {100 * c / len(R):.1f} % | '
              f'{spearmanr(P_, T_).correlation:.4f} | {kendalltau(P_, T_).correlation:.4f} |')
        fa = sum(r['p']['fits'] == r['t']['fits'] for r in R)
        w(f'| fits | {100 * fa / len(R):.1f} % | | | |\n')
    w('Per train group, pass A % (Stage-1 HS / ceiling):\n')
    w('| group | n | ' + ' | '.join(M) + ' |'); w('|---|---|' + '---|' * len(M))
    for g in groups:
        R = [r for r in recs if r['suite'] == g]
        cells = [f"{100 * sum(ok(m, r['p'][m], r['t'][m]) for r in R) / len(R):.0f} / "
                 f"{100 * sum(ok(m, r['q'][m], r['t'][m]) for r in R) / len(R):.0f}" for m in M]
        w(f'| {g} | {len(R)} | ' + ' | '.join(cells) + ' |')
    # ---- 3. LOGO
    w('\n## 3. LOGO of Stage 2 only (Stage 2 constants refit without the group; the Stage 1 rules are in-sample for every group)\n')
    data = fit.collect(evaluate.load())
    logo = collections.defaultdict(collections.Counter)
    for g in groups:
        C = fit.fit(data, holdout=g)
        for r in recs:
            if r['suite'] != g:
                continue
            p = stage2.predict(r['hs'], C)
            for m in M:
                logo[g][m] += ok(m, p[m], r['t'][m])
            logo[g]['n'] += 1
    w('| group | n | ' + ' | '.join(M) + ' |'); w('|---|---|' + '---|' * len(M))
    tot = collections.Counter()
    for g in groups:
        c = logo[g]; tot.update(c)
        w(f"| {g} | {c['n']} | " + ' | '.join(f"{100 * c[m] / c['n']:.0f}" for m in M) + ' |')
    w(f"| **train, LOGO** | {tot['n']} | " + ' | '.join(f"**{100 * tot[m] / tot['n']:.1f}**" for m in M) + ' |')
    # ---- residual list
    w('\n## 4. Train samples that pass with the tool HS but fail with the Stage-1 HS\n')
    w('| sample | metric | Stage-1 HS | tool HS | target |'); w('|---|---|---|---|---|')
    for r in recs:
        if r['source'] != 'train':
            continue
        for m in M:
            if ok(m, r['q'][m], r['t'][m]) and not ok(m, r['p'][m], r['t'][m]):
                w(f"| {r['sample'][:24]} | {m} | {r['p'][m]:.0f} | {r['q'][m]:.0f} | {r['t'][m]} |")
    w('\nReverse direction (pass only with the Stage-1 HS: the true-HS "ceiling" is not a per-design '
      'upper bound, Stage 2 errors can cancel Stage 1 errors):\n')
    w('| sample | metric | Stage-1 HS | tool HS | target |'); w('|---|---|---|---|---|')
    for r in recs:
        if r['source'] != 'train':
            continue
        for m in M:
            if ok(m, r['p'][m], r['t'][m]) and not ok(m, r['q'][m], r['t'][m]):
                w(f"| {r['sample'][:24]} | {m} | {r['p'][m]:.0f} | {r['q'][m]:.0f} | {r['t'][m]} |")
    w('\n## 4b. Signed compute-depth error (Stage 1 - tool), modules by kind\n')
    w('| set | kind | error: count |'); w('|---|---|---|')
    for src in ('train', 'extra'):
        bys = collections.defaultdict(collections.Counter)
        for r in recs:
            if r['source'] != src:
                continue
            th = sanitize.sanitize(S2[r['sample']]['input_hs'])
            pl = {(l['module'], l['loop']): l for l in r['hs']['loops']}
            for l in th['loops']:
                q = pl.get((l['module'], l['loop']))
                if l['pipelined'] and 'copy' not in l['loop'] and q:
                    k = stage1.module_kind(l['module'])
                    if k in ('loop_k', 'loop_i_loop_j_loop_k') and stage1.reassociates(cfg_of(r_by[r['sample']])):
                        k += ' (reassoc)'
                    bys[k][q['pipeline_depth'] - l['pipeline_depth']] += 1
        for k, c in sorted(bys.items()):
            w(f"| {src} | {k} | " + ', '.join(f'{e:+d}: {n}' for e, n in sorted(c.items())) + ' |')
    w('\n## 5. Support of the data-derived rules (designs exercising each branch / exactly right)\n')
    import subprocess
    w(subprocess.run([sys.executable, os.path.join(HERE, 'rule_support.py')], cwd=HERE,
                     capture_output=True, text=True).stdout)
    txt = '\n'.join(L) + '\n'
    open(os.path.join(HERE, 'reports', 'VALIDATION_STAGE1.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
