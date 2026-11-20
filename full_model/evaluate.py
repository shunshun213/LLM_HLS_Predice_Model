"""Evaluate Stage 2 on the TRUE HS (sanitized) against the true targets.
Judgement A: latency |err|<=10%, each resource |err|<=20%.  Judgement B: Spearman / Kendall.
Headline = source == train (216).  Also reported: extra, all, and per group."""
import json, sys, math, collections
from scipy.stats import spearmanr, kendalltau
import sanitize

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
# the package's own data/ folder (copied from full_dataset/hs)
DATA = os.path.join(_HERE, 'data', 'stage2_dataset.jsonl')
METRICS = ['latency_cycles', 'LUT', 'FF', 'DSP', 'BRAM_18K']
TOL = {'latency_cycles': 0.10, 'LUT': 0.20, 'FF': 0.20, 'DSP': 0.20, 'BRAM_18K': 0.20}


SHIFTS = os.path.join(os.path.dirname(DATA), 'expression_shifts.json')


def load():
    """Stage 2 dataset. True HS extension (hs_extract.py, pipelined shifts): the report's
    Expression table lists the pipelined shl expressions (2661 FF / 3250 LUT, 2.5 ns) that write a
    lane into a wide reshaped word; they are added to the true HS as aux 'shift' op 'shl' cores
    so that the true HS and the Stage-1 HS describe the same structure."""
    rows = [json.loads(l) for l in open(DATA)]
    if not os.path.exists(SHIFTS):
        raise FileNotFoundError(f'{SHIFTS} missing: run extract_expression_shifts.py (true-HS shl cores)')
    if True:
        sh = json.load(open(SHIFTS))
        for r in rows:
            for m, es in sh.get(r['sample'], {}).items():
                piped = [e for e in es if e['op'] == 'shl' and e['FF'] > 0]
                if piped:
                    W = max(e['w1'] for e in piped)
                    r['input_hs']['aux_operators'].append(dict(
                        owner=m, **{'class': 'shift'}, count_in_module=len(piped),
                        count_effective=len(piped), owner_instantiations=1,
                        params={'class': 'shift', 'latency': 4, 'op': 'shl',
                                'total_bits': sum(e['w1'] for e in piped),
                                'shamt_bits': (W - 1).bit_length(), 'width': W}))
    return rows


def ape(p, t):
    if t == 0:
        return 0.0 if p == 0 else float('inf')
    return abs(p - t) / abs(t)


def run(predict, rows=None, metrics=METRICS, verbose=True, groups=True):
    rows = rows or load()
    preds = []
    for r in rows:
        s = sanitize.sanitize(r['input_hs'])
        sanitize.assert_clean(s)
        preds.append(predict(s))
    res = {}
    for subset in ('train', 'extra', 'all'):
        idx = [i for i, r in enumerate(rows) if subset == 'all' or r['source'] == subset]
        out = {}
        for m in metrics:
            P = [preds[i][m] for i in idx]; T = [rows[i]['targets'][m] for i in idx]
            a = [ape(p, t) for p, t in zip(P, T)]
            fin = sorted(x for x in a if math.isfinite(x))
            out[m] = dict(passA=sum(x <= TOL[m] + 1e-12 for x in a) / len(a),
                          mape=sum(fin) / len(fin) if fin else 0,
                          medape=fin[len(fin) // 2] if fin else 0,
                          spearman=spearmanr(P, T).correlation if len(set(T)) > 1 else float('nan'),
                          kendall=kendalltau(P, T).correlation if len(set(T)) > 1 else float('nan'))
        if 'fits' in preds[0]:
            F = [(preds[i]['fits'], rows[i]['targets']['fits']) for i in idx]
            nf = [(p, t) for p, t in F if not t]
            out['fits'] = dict(acc=sum(p == t for p, t in F) / len(F),
                               notfit_recall=(sum(not p for p, t in nf) / len(nf)) if nf else float('nan'),
                               notfit_precision=(sum((not p) and (not t) for p, t in F) /
                                                 max(1, sum(not p for p, t in F))))
        res[subset] = out
        if verbose:
            print(f'--- {subset} (n={len(idx)})')
            for m in metrics:
                o = out[m]
                print(f"  {m:15s} passA {o['passA']*100:6.1f}%  MAPE {o['mape']*100:7.2f}%  "
                      f"medAPE {o['medape']*100:6.2f}%  rho {o['spearman']:.4f}  tau {o['kendall']:.4f}")
            if 'fits' in out:
                print('  fits', {k: round(v, 4) for k, v in out['fits'].items()})
    if groups and verbose:
        print('--- per group (passA %)')
        g = collections.defaultdict(list)
        for i, r in enumerate(rows):
            g[(r['source'], r['suite'])].append(i)
        for k in sorted(g):
            line = f'  {k[0]:5s} {k[1]:22s} n={len(g[k]):3d}'
            for m in metrics:
                a = [ape(preds[i][m], rows[i]['targets'][m]) for i in g[k]]
                line += f'  {m[:3]} {sum(x <= TOL[m] + 1e-12 for x in a) / len(a) * 100:5.1f}'
            print(line)
    return res, preds, rows
