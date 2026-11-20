// GEMM kernel: C = A * B, all N x N float.
// The top-level ports A/B/C are copied into local arrays lA/lB/lC, the product is
// computed on the local arrays, and lC is written back to C. Array directives
// (partition / reshape / bind_storage) target lA/lB/lC only, so every configuration
// has the same external interface. The copy loops are fixed at PIPELINE II=1 here;
// every other pragma is injected by hls_runner.py via set_directive_* in the
// generated Tcl, so one source file covers the whole design space. Loop labels and
// the named temporaries (prod, acc) are the handles the directives attach to.
#ifndef N
#define N 64
#endif

void gemm(const float A[N][N], const float B[N][N], float C[N][N]) {
    float lA[N][N], lB[N][N], lC[N][N];

copy_in:
    for (int i = 0; i < N; i++) {
    copy_in_j:
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
            lB[i][j] = B[i][j];
        }
    }

loop_i:
    for (int i = 0; i < N; i++) {
    loop_j:
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
        loop_k:
            for (int k = 0; k < N; k++) {
                float prod = lA[i][k] * lB[k][j];  // BIND_OP target for fmul
                float acc = sum + prod;            // BIND_OP target for fadd
                sum = acc;
            }
            lC[i][j] = sum;
        }
    }

copy_out:
    for (int i = 0; i < N; i++) {
    copy_out_j:
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = lC[i][j];
        }
    }
}
