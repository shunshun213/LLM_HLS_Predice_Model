"""Parser robustness tests (non-canonical but valid syntax, and out-of-space input).
Run: python3 test_parser.py   (exit code 0 = all pass)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_parse as P, predict

BASE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemm_base.cpp')).read()
K = "loop_k:\n            for (int k = 0; k < N; k++) {"
J = "loop_j:\n        for (int j = 0; j < N; j++) {"
ARRS = "    float lA[N][N], lB[N][N], lC[N][N];"


def src(at, *pr):
    return BASE.replace(at, at + '\n' + '\n'.join(pr))


def ok(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    return bool(cond)


def raises(fn):
    try:
        fn()
    except P.UnsupportedDesign:
        return True
    return False


res = []
c = P.parse_source(src(J, '#pragma HLS PIPELINE II=1 style=FLP'), clock_ns=10)
res.append(ok('style value case-insensitive', c['pipeline_style'] == 'flp'))
c = P.parse_source(src(J, '#pragma HLS pipeline style=frp'), clock_ns=10)
res.append(ok('frp -> flp with note', c['pipeline_style'] == 'flp' and c['_notes']))
c = P.parse_source(BASE.replace(K, 'loop_k: for (int k = 0; k < N; k++) {\n#pragma hls pipeline ii=2'), clock_ns=5)
res.append(ok('label on the same line + lower-case pragma', c['pipeline_loop'] == 'k' and c['pipeline_ii'] == 2))
c = P.parse_source(src(J, '#pragma HLS PIPELINE II=1') .replace(K, K + '\n#pragma HLS PIPELINE II=4'), clock_ns=10)
res.append(ok('outermost PIPELINE wins', c['pipeline_loop'] == 'j' and c['unroll_k'] == 'full' and c['_notes']))
c = P.parse_source(src(K, '#if 0', '#pragma HLS UNROLL factor=8', '#endif', '/* #pragma HLS UNROLL factor=4 */'), clock_ns=10)
res.append(ok('#if 0 and /* */ ignored', c['unroll_k'] == 1))
c = P.parse_source(src(K, '#pragma HLS PIPELINE \\', '    II=2'), clock_ns=10)
res.append(ok('line continuation', c['pipeline_ii'] == 2))
c = P.parse_source('// do NOT use -unsafe_math_optimizations here\n' + BASE, clock_ns=10)
res.append(ok('unsafe_math only from a config_compile line', c['unsafe_math'] is False))
c = P.parse_source('// config_compile -unsafe_math_optimizations\n' + BASE, clock_ns=10)
res.append(ok('unsafe_math from config_compile', c['unsafe_math'] is True))
c = P.parse_source(src(ARRS, '#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=4 dim=1'), clock_ns=10)
res.append(ok('old ARRAY_PARTITION syntax', c['B_mode'] == 'partition' and c['B_type'] == 'cyclic' and c['B_factor'] == 4))
c = P.parse_source(src(ARRS, '#pragma HLS BIND_STORAGE variable=lA type=RAM_2P impl=BRAM'), clock_ns=10)
res.append(ok('storage values case-insensitive', c['A_storage'] == 'ram_2p' and c['A_storage_impl'] == 'bram'))
c = P.parse_source(src(K, '#pragma HLS BIND_OP variable=prod impl=FullDSP latency=3'), clock_ns=10)
res.append(ok('BIND_OP impl case-insensitive, op from variable', c['fmul_impl'] == 'fulldsp' and c['fmul_latency'] == 3))
res.append(ok('BIND_OP unknown impl -> UnsupportedDesign',
              raises(lambda: P.parse_source(src(K, '#pragma HLS BIND_OP variable=prod op=fmul impl=primitivedsp'), clock_ns=10))))
res.append(ok('BIND_STORAGE ram_s2p -> UnsupportedDesign',
              raises(lambda: P.parse_source(src(ARRS, '#pragma HLS BIND_STORAGE variable=lA type=ram_s2p'), clock_ns=10))))
res.append(ok('UNROLL factor=0 -> UnsupportedDesign', raises(lambda: P.parse_source(src(K, '#pragma HLS UNROLL factor=0'), clock_ns=10))))
res.append(ok('dim=0 -> UnsupportedDesign', raises(lambda: P.parse_source(src(ARRS, '#pragma HLS ARRAY_PARTITION variable=lA type=complete dim=0'), clock_ns=10))))
res.append(ok('dim=3 -> UnsupportedDesign', raises(lambda: P.parse_source(src(ARRS, '#pragma HLS ARRAY_PARTITION variable=lA type=complete dim=3'), clock_ns=10))))
res.append(ok('UNROLL on loop_i -> UnsupportedDesign',
              raises(lambda: P.parse_source(BASE.replace("loop_i:\n    for (int i = 0; i < N; i++) {", "loop_i:\n    for (int i = 0; i < N; i++) {\n#pragma HLS UNROLL factor=2"), clock_ns=10))))
c = P.parse_source(src(K, '#pragma HLS UNROLL factor=48'), clock_ns=10)
res.append(ok('UNROLL factor 48 kept (48 copies)', c['unroll_k'] == 48))
tcl = 'create_clock -period 200MHz\nset_directive_pipeline -II 1 -style flp gemm/loop_j\nset_directive_unroll gemm/loop_k -factor 4\n'
c = P.parse_tcl(tcl)
res.append(ok('Tcl: unquoted location, options after location, MHz clock',
              c['pipeline_loop'] == 'j' and c['pipeline_style'] == 'flp' and c['clock_ns'] == 5.0))
tcl = 'create_clock -period 5\nset_directive_array_partition "gemm" lB -type cyclic -factor 8 -dim 1\nset_directive_bind_op -op fadd -impl fabric "gemm/loop_k" acc\n'
c = P.parse_tcl(tcl)
res.append(ok('Tcl: variable before options', c['B_factor'] == 8 and c['fadd_impl'] == 'fabric'))
try:
    predict.predict(P.parse_source(BASE, clock_ns=4.0))
    res.append(ok('clock 4 ns -> clean error', False))
except ValueError as e:
    res.append(ok('clock 4 ns -> clean error', 'characterized' in str(e)))
res.append(ok('copy-loop PIPELINE II=2 -> UnsupportedDesign',
              raises(lambda: P.parse_source(BASE.replace('#pragma HLS PIPELINE II=1', '#pragma HLS PIPELINE II=2', 1), clock_ns=10))))
res.append(ok('#define N 32 -> UnsupportedDesign', raises(lambda: P.parse_source(BASE.replace('#define N 64', '#define N 32'), clock_ns=10))))
res.append(ok('DATAFLOW -> UnsupportedDesign', raises(lambda: P.parse_source(src(ARRS, '#pragma HLS DATAFLOW'), clock_ns=10))))
res.append(ok('function-level PIPELINE -> UnsupportedDesign', raises(lambda: P.parse_source(src(ARRS, '#pragma HLS PIPELINE'), clock_ns=10))))
o = predict.predict(P.parse_source(src(J, '#pragma HLS PIPELINE II=16'), clock_ns=10))
res.append(ok('value outside the data is flagged', any('outside the data: pipeline_ii=16' in c for c in o['confidence'])))
print(f'{sum(res)}/{len(res)} passed')
sys.exit(0 if all(res) else 1)
