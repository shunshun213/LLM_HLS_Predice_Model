"""Write reports/VALIDATION.md: Stage 2 on the TRUE HS. Judgement A/B, fits, per group, leave-one-group-out."""
import os
import io, contextlib, collections, math
import evaluate, stage2, sanitize, fit

rows = evaluate.load()
res, preds, _ = evaluate.run(stage2.predict, rows=rows, verbose=False, groups=False)
M = evaluate.METRICS
out = io.StringIO()
w = lambda s='': out.write(s + '\n')
w('# Stage 2 validation on the TRUE hardware structure\n')
w('Input: sanitized true HS (`sanitize.py`). Targets: csynth totals. Judgement A: latency |err| <= 10 %, '
  'each resource |err| <= 20 %. Judgement B: Spearman rho / Kendall tau. Constants from all 510 samples; '
  'headline = `source == train` (216).\n')
for subset in ('train', 'extra', 'all'):
    o = res[subset]
    n = sum(1 for r in rows if subset == 'all' or r['source'] == subset)
    w(f'## {subset} (n = {n})\n')
    w('| metric | Judgement A pass | MAPE | median APE | Spearman | Kendall |')
    w('|---|---|---|---|---|---|')
    for m in M:
        x = o[m]
        w(f"| {m} | {x['passA']*100:.1f} % | {x['mape']*100:.2f} % | {x['medape']*100:.2f} % | "
          f"{x['spearman']:.4f} | {x['kendall']:.4f} |")
    f = o['fits']
    w(f"\nfits flag: accuracy {f['acc']*100:.1f} %, not-fit recall {f['notfit_recall']*100:.1f} %, "
      f"not-fit precision {f['notfit_precision']*100:.1f} %\n")

# per group
w('## Judgement A pass rate by group\n')
w('| source | group | n | latency | LUT | FF | DSP | BRAM_18K |')
w('|---|---|---|---|---|---|---|---|')
g = collections.defaultdict(list)
for i, r in enumerate(rows):
    g[(r['source'], r['suite'])].append(i)
for k in sorted(g):
    cells = []
    for m in M:
        a = [evaluate.ape(preds[i][m], rows[i]['targets'][m]) <= evaluate.TOL[m] + 1e-12 for i in g[k]]
        cells.append(f'{100*sum(a)/len(a):.0f}')
    w(f'| {k[0]} | {k[1]} | {len(g[k])} | ' + ' | '.join(cells) + ' |')

# leave-one-group-out
w('\n## Generalization: leave-one-group-out (constants refit without the group)\n')
data = fit.collect(rows)
ok = collections.Counter(); n = collections.Counter()
for grp in sorted({r['suite'] for r in rows}):
    C = fit.fit(data, holdout=grp)
    for r in rows:
        if r['suite'] != grp:
            continue
        p = stage2.predict(sanitize.sanitize(r['input_hs']), C)
        for m in M:
            t = r['targets'][m]
            ok[(r['source'], m)] += evaluate.ape(p[m], t) <= evaluate.TOL[m] + 1e-12
            n[(r['source'], m)] += 1
w('| source | ' + ' | '.join(M) + ' |')
w('|---|' + '---|' * len(M))
for src in ('train', 'extra'):
    w(f'| {src} | ' + ' | '.join(f"{100*ok[(src,m)]/n[(src,m)]:.1f} %" for m in M) + ' |')
w('\nNote: only the per-unit constants of R3 are refit per fold; the written rules (L1-L5, R1, R2) were '
  'derived by inspecting all 510 samples.')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'VALIDATION.md'), 'w').write(out.getvalue())
print(out.getvalue())
