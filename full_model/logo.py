"""Leave-one-group-out: refit the per-unit constants without group g, predict group g."""
import evaluate, fit, stage2, sanitize, collections
rows = evaluate.load()
data = fit.collect(rows)
groups = sorted({r['suite'] for r in rows})
TOL = evaluate.TOL
ok = collections.Counter(); n = collections.Counter(); per = {}
for g in groups:
    C = fit.fit(data, holdout=g)
    for r in rows:
        if r['suite'] != g: continue
        p = stage2.predict(sanitize.sanitize(r['input_hs']), C)
        for m in ('LUT', 'FF', 'DSP', 'BRAM_18K', 'latency_cycles'):
            t = r['targets'][m]
            good = abs(p[m] - t) <= TOL[m] * t + 1e-9 if t else p[m] == 0
            ok[(r['source'], m)] += good; n[(r['source'], m)] += 1
            per.setdefault(g, collections.Counter())[m] += good
        per[g]['n'] += 1
for src in ('train', 'extra'):
    print(src, {m: round(100 * ok[(src, m)] / n[(src, m)], 1) for m in ('latency_cycles', 'LUT', 'FF', 'DSP', 'BRAM_18K')})
for g in groups:
    c = per[g]; print(f"  {g:24s} n={c['n']:3d} LUT {100*c['LUT']/c['n']:5.1f} FF {100*c['FF']/c['n']:5.1f}")
