"""Stage 2 model entry point: sanitized HS -> {latency_cycles, LUT, FF, DSP, BRAM_18K, fits}."""
import json, os
import latency, resources, features, sanitize

_CONSTS = None


def consts():
    global _CONSTS
    if _CONSTS is None:
        _CONSTS = json.load(open(os.path.join(os.path.dirname(__file__), 'constants.json')))
    return _CONSTS


def category_model(hs, m, cat, C=None):
    C = C or consts()
    cls = features.module_class(m)
    f = features.module_features(hs, m)
    out = {}
    for res in ('LUT', 'FF'):
        k = f'{cls}|{cat}|{res}'
        out[res] = sum(c * f.get(b, 0.0) for b, c in C.get(k, {}).items())
    return out


def predict(hs, C=None):
    """hs may be a raw or an already sanitized HS: it is (re)sanitized and checked here, so no
    forbidden field can reach the model whatever the caller passes."""
    hs = sanitize.sanitize(hs)
    sanitize.assert_clean(hs)
    r = resources.predict_resources(hs, lambda h, m, c: category_model(h, m, c, C))
    r['latency_cycles'] = latency.predict_latency(hs)
    return r
