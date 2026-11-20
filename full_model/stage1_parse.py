"""Stage 1 front end: HLS code (+ Tcl) -> canonical design description `cfg`.

Two equivalent inputs are accepted:
  * the Tcl script (set_directive_* + create_clock + config_compile), or
  * the C++ source with inline #pragma HLS. The clock and config_compile come from an explicit
    argument, or from Tcl-style lines echoed in the file (e.g. a header comment
    `// create_clock -period 2.5`, `// config_compile -unsafe_math_optimizations`).
The result has the same keys as the dataset's config. That config is used only to *check* the
parser (0 mismatches on 510 designs), never as model input.

Contract: a pragma the model covers is applied; a pragma or value outside the modeled space either
raises `UnsupportedDesign` (a clean error naming the construct) or is ignored with a note in
cfg['_notes'] that predict.py reports under `confidence`. Nothing is dropped silently.
"""
import re, shlex

DEFAULT = {
    'pipeline_loop': 'none', 'pipeline_ii': None, 'pipeline_style': None,
    'unroll_k': 1, 'unroll_j': 1,
    'A_mode': 'none', 'A_type': None, 'A_dim': None, 'A_factor': None,
    'B_mode': 'none', 'B_type': None, 'B_dim': None, 'B_factor': None,
    'C_mode': 'none', 'C_type': None, 'C_dim': None, 'C_factor': None,
    'A_storage': 'default', 'B_storage': 'default', 'A_storage_impl': None, 'B_storage_impl': None,
    'fmul_impl': 'auto', 'fmul_latency': None, 'fadd_impl': 'auto', 'fadd_latency': None,
    'flatten': 'auto', 'clock_ns': None, 'unsafe_math': False,
}
ARR = {'lA': 'A', 'lB': 'B', 'lC': 'C'}
LOOPS = ('loop_i', 'loop_j', 'loop_k')
IMPLS = {'fmul': ('fabric', 'meddsp', 'fulldsp', 'maxdsp'), 'fadd': ('fabric', 'fulldsp')}
STORAGE = ('ram_1p', 'ram_2p', 'ram_t2p')
STORAGE_IMPL = ('auto', 'bram', 'lutram')
SUPPORTED = ('pipeline', 'unroll', 'loop_flatten', 'array_partition', 'array_reshape',
             'bind_storage', 'bind_op')


from stage1 import UnsupportedDesign     # one exception type for parser and model domain errors


def _num(x, what):
    if x is None or x is True:
        return None
    try:
        return int(str(x))
    except ValueError:
        raise UnsupportedDesign(f'{what}={x}: not an integer')


def _settings_text(text):
    """text in which Tcl setting lines are looked up: block comments and `#if 0` blocks removed,
    `//` line comments kept (the echoed header lines); only line-anchored commands count."""
    s = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    out, skip = [], 0
    for line in s.splitlines():
        t = line.strip()
        if skip:
            skip += 1 if re.match(r'#\s*if', t) else (-1 if re.match(r'#\s*endif', t) else 0)
            continue
        if re.match(r'#\s*if\s+0\b', t):
            skip = 1
            continue
        out.append(line)
    return '\n'.join(out)


def parse_clock(text):
    """create_clock -period P (P in ns, or with an MHz suffix), on a line of its own (optionally
    behind //, as in the echoed header). An explicit --clock argument overrides it."""
    text = _settings_text(text)
    m = re.search(r'(?m)^\s*(?://\s*)?create_clock\s+(?:-name\s+\S+\s+)?-period\s+["{]?([0-9.]+)\s*(mhz|ns)?["}]?', text, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return round(1000.0 / v, 3) if (m.group(2) or '').lower() == 'mhz' else v


def unsafe_math_in(text):
    text = _settings_text(text)
    m = re.search(r'^\s*(?://\s*)?config_compile\b[^\n]*-unsafe_math_optimizations(?:[ =]+(\w+))?', text, re.M)
    return bool(m) and (m.group(1) or 'true').lower() not in ('false', '0', 'off', 'no')


def tool_setting_notes(text):
    """the model assumes the dataset's tool settings (no automatic pipelining)"""
    m = re.search(r'config_compile\b[^\n]*-pipeline_loops[ =]+(\d+)', text)
    if m is None or m.group(1) != '0':
        return ['tool setting: config_compile -pipeline_loops 0 not found; the model assumes no '
                'automatic pipelining (the dataset setting)']
    return []


class _Builder:
    """Collects directives, validates them, resolves conflicts (outermost PIPELINE wins)."""

    def __init__(self):
        self.cfg = dict(DEFAULT)
        self.notes = []
        self.pipes = []                                  # (loop, II, style)
        self.copy_pipes = set()

    def apply(self, kind, opts, loop, var):
        c, n = self.cfg, self.notes
        if kind == 'dataflow' or (kind == 'pipeline' and loop is None and not opts.get('off')):
            raise UnsupportedDesign(f'{kind.upper()} at function level restructures the whole design: '
                                    f'outside the modeled space')
        if kind not in SUPPORTED:
            n.append(f'{kind}: pragma not modeled, ignored')
            return
        if kind in ('pipeline', 'unroll', 'loop_flatten'):
            if loop in ('copy_in', 'copy_in_j', 'copy_out', 'copy_out_j'):
                ii = _num(opts.get('ii'), 'II')
                if kind == 'pipeline' and loop in ('copy_in_j', 'copy_out_j') and not opts.get('off') \
                        and (ii in (None, 1)) and str(opts.get('style') or 'stp').lower() == 'stp':
                    self.copy_pipes.add(loop)            # the fixed copy-loop pragma (II=1)
                    return
                raise UnsupportedDesign(f'{kind} on {loop}: the copy loops are fixed in gemm_base.cpp '
                                        f'(PIPELINE II=1 on copy_in_j / copy_out_j only)')
            if loop not in LOOPS:
                n.append(f'{kind} outside loop_i/j/k ({loop}): ignored')
                return
        if kind == 'pipeline':
            if opts.get('off'):
                return
            ii = _num(opts.get('ii'), 'II')
            if ii is not None and ii < 1:
                ii = None                                # II=-1 / 0: tool default (II = 1)
            style = (opts.get('style') or 'stp').lower()
            if style not in ('stp', 'flp', 'frp'):
                raise UnsupportedDesign(f'PIPELINE style={style}')
            if style == 'frp':
                n.append('PIPELINE style=frp (free-running) not in the data: modeled as flp')
                style = 'flp'
            self.pipes.append((loop, ii or 1, style))
        elif kind == 'unroll':
            if loop == 'loop_i':
                raise UnsupportedDesign('UNROLL on loop_i is outside the modeled space')
            f = opts.get('factor')
            if f is None or f is True:
                c['unroll_' + loop[-1]] = 'full'
            else:
                f = _num(f, 'UNROLL factor')
                if f < 1:
                    raise UnsupportedDesign(f'UNROLL factor={f}')
                c['unroll_' + loop[-1]] = 'full' if f >= 64 else f
        elif kind == 'loop_flatten':
            if opts.get('off'):
                c['flatten'] = 'off'
        elif kind in ('array_partition', 'array_reshape'):
            if var not in ARR:
                n.append(f'{kind} on {var}: not a local array lA/lB/lC, ignored')
                return
            a = ARR[var]
            t = str(opts.get('type') or 'complete').lower()
            if t not in ('block', 'cyclic', 'complete'):
                raise UnsupportedDesign(f'{kind} type={t}')
            d = _num(opts.get('dim'), 'dim')
            d = 1 if d is None else d
            if d not in (1, 2):
                raise UnsupportedDesign(f'{kind} dim={d} on {var}: only dim=1 or 2 is modeled '
                                        f'(dim=0, all dimensions, is outside the design space)')
            c[a + '_mode'] = 'partition' if kind == 'array_partition' else 'reshape'
            c[a + '_type'] = t
            c[a + '_dim'] = d
            if t == 'complete':
                c[a + '_factor'] = None
            else:
                f = _num(opts.get('factor'), 'factor')
                if f is None or f < 2 or 64 % f:
                    raise UnsupportedDesign(f'{kind} factor={f} on {var}: must divide 64 (2..64)')
                if f == 64:
                    c[a + '_type'], c[a + '_factor'] = 'complete', None
                else:
                    c[a + '_factor'] = f
        elif kind == 'bind_storage':
            if var not in ARR or var == 'lC':
                n.append(f'bind_storage on {var}: only lA / lB are modeled, ignored')
                return
            a = ARR[var]
            t = str(opts.get('type') or '').lower()
            impl = str(opts.get('impl') or 'auto').lower()
            if t not in STORAGE:
                raise UnsupportedDesign(f'BIND_STORAGE type={t or "?"}: modeled types are {", ".join(STORAGE)}')
            if impl not in STORAGE_IMPL:
                raise UnsupportedDesign(f'BIND_STORAGE impl={impl}: modeled impls are bram / lutram')
            c[a + '_storage'], c[a + '_storage_impl'] = t, impl
        elif kind == 'bind_op':
            op = str(opts.get('op') or {'prod': 'fmul', 'acc': 'fadd'}.get(var, '')).lower()
            if op not in IMPLS:
                n.append(f'bind_op on {var} op={op or "?"}: not the fmul/fadd of the kernel, ignored')
                return
            impl = str(opts.get('impl') or 'auto').lower()
            if impl not in IMPLS[op] + ('auto',):
                raise UnsupportedDesign(f'BIND_OP {op} impl={impl}: characterized impls are '
                                        f'{", ".join(IMPLS[op])}')
            lat = _num(opts.get('latency'), 'latency')
            c[op + '_impl'] = impl
            c[op + '_latency'] = None if lat is None or lat < 0 else lat

    def finish(self):
        c = self.cfg
        if self.pipes:
            order = {'loop_i': 0, 'loop_j': 1, 'loop_k': 2}
            loop, ii, style = min(self.pipes, key=lambda p: order[p[0]])
            if len({p[0] for p in self.pipes}) > 1:
                self.notes.append(f'PIPELINE on {", ".join(sorted({p[0] for p in self.pipes}))}: the outermost '
                                  f'({loop}) wins, the loops inside it are fully unrolled')
            c['pipeline_loop'], c['pipeline_ii'], c['pipeline_style'] = loop[-1], ii, style
        _defaults(c)
        c['_notes'] = self.notes
        return c


def check_size(text):
    for m in re.finditer(r'(?:#\s*define\s+N\s+|-DN=)(\d+)', text):
        if int(m.group(1)) != 64:
            raise UnsupportedDesign(f'N={m.group(1)}: the model is built for N = 64')


def parse_tcl(tcl, clock_ns=None, unsafe_math=None):
    """set_directive_* script. Lines are tokenized like Tcl words (shlex): the location
    ("gemm/loop_k", quoted or not) and the options may come in any order."""
    b = _Builder()
    text = re.sub(r'\\\n', ' ', tcl)
    check_size(text)
    b.cfg['clock_ns'] = clock_ns if clock_ns is not None else parse_clock(text)
    b.cfg['unsafe_math'] = unsafe_math_in(text) if unsafe_math is None else unsafe_math
    b.notes += tool_setting_notes(text)
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r'set_directive_(\w+)\s+(.*)', s)
        if not m:
            continue
        kind = m.group(1).lower()
        try:
            toks = [t.strip('{}') for t in shlex.split(m.group(2), comments=True)]
        except ValueError:
            raise UnsupportedDesign(f'cannot tokenize Tcl line: {s}')
        opts, pos, i = {}, [], 0
        while i < len(toks):
            t = toks[i]
            if t.startswith('-') and len(t) > 1 and not re.match(r'-\d', t):
                key = t[1:].lower()
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt is not None and not nxt.startswith('-') and not nxt.startswith('gemm') \
                        and key not in ('off', 'rewind', 'skip_exit_check', 'region', 'enable_flush'):
                    opts[key] = nxt; i += 2
                else:
                    opts[key] = True; i += 1
            else:
                pos.append(t); i += 1
        loc = next((p for p in pos if p.split('/')[0] == 'gemm'), None)
        if loc is None:
            b.notes.append(f'{s}: no gemm location, ignored')
            continue
        parts = loc.split('/')
        loop = parts[1] if len(parts) > 1 else None
        var = next((p for p in pos if p != loc), None) or opts.get('variable')
        if kind == 'bind_op' and 'op' not in opts and var:
            opts['op'] = {'prod': 'fmul', 'acc': 'fadd'}.get(var)
        b.apply(kind, opts, loop, var)
    return b.finish()


def _src_opts(rest):
    """options of one #pragma HLS line: key=value pairs (keys lower-case) plus bare words
    (off, and complete / cyclic / block for the old ARRAY_PARTITION syntax)."""
    opts = {}
    for k, v in re.findall(r'(\w+)\s*=\s*([\w.\-]+)', rest):
        opts[k.lower()] = v
    for w in re.sub(r'\w+\s*=\s*[\w.\-]+', ' ', rest).split():
        wl = w.lower()
        if wl == 'off':
            opts['off'] = True
        elif wl in ('complete', 'cyclic', 'block') and 'type' not in opts:
            opts['type'] = wl
    return opts


def _strip_code(src):
    """remove /* */ comments, `#if 0 ... #endif` blocks and line continuations."""
    s = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    s = re.sub(r'\\\n', ' ', s)
    out, skip = [], 0
    for line in s.splitlines():
        t = line.strip()
        if skip:
            if re.match(r'#\s*if', t):
                skip += 1
            elif re.match(r'#\s*endif', t):
                skip -= 1
            elif re.match(r'#\s*(else|elif)', t) and skip == 1:
                skip = 0
            continue
        if re.match(r'#\s*if\s+0\b', t):
            skip = 1
            continue
        out.append(re.sub(r'//.*', '', line))
    return '\n'.join(out)


def parse_source(src, clock_ns=None, unsafe_math=None):
    """C++ source with inline #pragma HLS. A pragma applies to the innermost enclosing labeled
    loop (brace-depth tracking), or to the function scope (array / storage / bind pragmas).
    Labels may stand alone or sit in front of the `for`; keywords and values are
    case-insensitive."""
    b = _Builder()
    b.cfg['clock_ns'] = clock_ns if clock_ns is not None else parse_clock(src)
    b.cfg['unsafe_math'] = unsafe_math_in(src) if unsafe_math is None else unsafe_math
    code = _strip_code(src)
    check_size(code)
    i0 = code.find('void gemm')
    body = code[i0 if i0 >= 0 else 0:]
    stack, depth, pending = [], 0, None
    for line in body.splitlines():
        s = line.strip()
        pm = re.match(r'#\s*pragma\s+HLS\s+(\w+)\s*(.*)', s, re.I)
        if pm:
            kind, opts = pm.group(1).lower(), _src_opts(pm.group(2))
            label = stack[-1][0] if stack else None
            b.apply(kind, opts, label, opts.get('variable'))
            continue
        lm = re.match(r'(\w+)\s*:\s*(for\b.*)?$', s)
        if lm and lm.group(1) not in ('default', 'public', 'private'):
            pending = lm.group(1)
        if re.search(r'\bfor\b', s) and pending is None and '{' in s:
            pending = '<unlabeled>'
        for ch in s:
            if ch == '{':
                depth += 1
                if pending is not None:
                    stack.append((pending, depth)); pending = None
            elif ch == '}':
                if stack and stack[-1][1] == depth:
                    stack.pop()
                depth -= 1
    missing = {'copy_in_j', 'copy_out_j'} - b.copy_pipes
    if missing:
        raise UnsupportedDesign(f'PIPELINE II=1 missing on {", ".join(sorted(missing))}: the copy loops '
                                f'are fixed in gemm_base.cpp')
    return b.finish()


def _defaults(cfg):
    # PIPELINE on an outer loop fully unrolls every loop inside it (HLS semantics)
    if cfg['pipeline_loop'] in ('j', 'i'):
        cfg['unroll_k'] = 'full'
    if cfg['pipeline_loop'] == 'i':
        cfg['unroll_j'] = 'full'
    if cfg['fmul_impl'] != 'auto' and cfg['fmul_latency'] is None:
        cfg['fmul_latency'] = 'auto'
    if cfg['fadd_impl'] != 'auto' and cfg['fadd_latency'] is None:
        cfg['fadd_latency'] = 'auto'
