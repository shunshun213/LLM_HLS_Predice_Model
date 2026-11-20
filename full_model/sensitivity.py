"""Step 3 of stage2_impl.md: sensitivity of the Stage 2 outputs to an error in each HS field
(i.e. how precisely Stage 1 must predict it). Each perturbation is applied to the sanitized TRUE HS
of the 216 train samples; reported: median and 90th-percentile |relative change| of each output and
the Judgement-A pass rate after the perturbation."""
import os
import copy, statistics
import evaluate, sanitize, stage2

rows = [r for r in evaluate.load() if r['source'] == 'train']
M = evaluate.METRICS


def compute_loops(hs):
    return [l for l in hs['loops'] if l['module'] != 'gemm' and 'copy' not in l['module'] and l['pipelined']]


def p_ii(hs):
    for l in compute_loops(hs):
        l['ii_achieved'] = (l['ii_achieved'] or 1) + 1


def p_depth(hs):
    for l in compute_loops(hs):
        l['pipeline_depth'] = (l['pipeline_depth'] or 1) + 1


def p_depth10(hs):
    for l in compute_loops(hs):
        l['pipeline_depth'] = round((l['pipeline_depth'] or 1) * 1.1)


def p_fadd_lat(hs):
    for o in hs['operators']:
        if o['op'] == 'fadd':
            o['latency'] += 1


def p_ncores(hs):
    for o in hs['operators']:
        o['count_in_module'] *= 2
        o['count_effective'] *= 2


def p_impl(hs):
    for o in hs['operators']:
        if o['op'] == 'fmul':
            o['impl'] = 'maxdsp' if o['impl'] != 'maxdsp' else 'fulldsp'


def p_banks(hs):
    for a, m in hs['memories'].items():
        if a == 'lB':
            m['banks'] *= 2
            m['words_per_bank'] = [max(1, w // 2) for w in m['words_per_bank']]


def p_tdp(hs):  # one more read port per bank in every module (shape / replication decisions)
    for ap in hs['access_pattern'].values():
        for v in ap.values():
            if v['read_ports_per_bank_max']:
                v['read_ports_per_bank_max'] += 1


def p_scalar(hs):
    for ap in hs['access_pattern'].values():
        for v in ap.values():
            if v.get('scalar_inputs'):
                v['scalar_inputs'] *= 2
                v['scalar_input_bits'] *= 2


def p_1wnr(hs):
    for m in hs['memories'].values():
        m['storage_inferred_by_tool'] = None


PERT = [('II_achieved +1 (compute loops)', p_ii), ('pipeline depth +1', p_depth),
        ('pipeline depth +10 %', p_depth10), ('fadd latency +1', p_fadd_lat),
        ('fp core count x2', p_ncores), ('fmul impl swapped', p_impl), ('lB banks x2', p_banks),
        ('read ports per bank +1', p_tdp), ('hoisted scalar count x2', p_scalar),
        ('RAM_1WNR replication missed', p_1wnr)]

base = {r['sample']: stage2.predict(sanitize.sanitize(r['input_hs'])) for r in rows}
lines = ['# Stage 2 input sensitivity (Step 3 of stage2_impl.md)\n',
         'Perturbation applied to every train sample (216) on the sanitized TRUE HS. Cells: median / 90th '
         'percentile of |relative change| of the prediction, then the Judgement-A pass rate after the perturbation '
         '(baseline in the first row).\n',
         '| perturbation | ' + ' | '.join(M) + ' |', '|---|' + '---|' * len(M)]


def passrate(preds):
    out = []
    for m in M:
        ok = sum(evaluate.ape(preds[r['sample']][m], r['targets'][m]) <= evaluate.TOL[m] + 1e-12 for r in rows)
        out.append(100 * ok / len(rows))
    return out


pr = passrate(base)
lines.append('| baseline (true HS) | ' + ' | '.join(f'pass {p:.0f} %' for p in pr) + ' |')
for name, fn in PERT:
    pred = {}
    for r in rows:
        s = sanitize.sanitize(r['input_hs'])
        fn(s)
        pred[r['sample']] = stage2.predict(s)
    pr = passrate(pred)
    cells = []
    for i, m in enumerate(M):
        d = sorted(abs(pred[k][m] - base[k][m]) / max(1, abs(base[k][m])) for k in base)
        cells.append(f'{100*statistics.median(d):.1f} / {100*d[int(0.9*len(d))]:.1f} % (pass {pr[i]:.0f} %)')
    lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'SENSITIVITY.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
