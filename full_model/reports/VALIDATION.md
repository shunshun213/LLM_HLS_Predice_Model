# Stage 2 validation on the TRUE hardware structure

Input: sanitized true HS (`sanitize.py`). Targets: csynth totals. Judgement A: latency |err| <= 10 %, each resource |err| <= 20 %. Judgement B: Spearman rho / Kendall tau. Constants from all 510 samples; headline = `source == train` (216).

## train (n = 216)

| metric | Judgement A pass | MAPE | median APE | Spearman | Kendall |
|---|---|---|---|---|---|
| latency_cycles | 99.5 % | 0.36 % | 0.00 % | 0.9995 | 0.9949 |
| LUT | 77.3 % | 12.51 % | 8.16 % | 0.9876 | 0.9215 |
| FF | 79.6 % | 12.66 % | 7.59 % | 0.9860 | 0.9145 |
| DSP | 100.0 % | 0.00 % | 0.00 % | 1.0000 | 1.0000 |
| BRAM_18K | 93.5 % | 3.18 % | 0.00 % | 0.9885 | 0.9642 |

fits flag: accuracy 99.5 %, not-fit recall 96.6 %, not-fit precision 100.0 %

## extra (n = 294)

| metric | Judgement A pass | MAPE | median APE | Spearman | Kendall |
|---|---|---|---|---|---|
| latency_cycles | 100.0 % | 0.03 % | 0.00 % | 0.9998 | 0.9987 |
| LUT | 92.2 % | 6.35 % | 3.64 % | 0.9795 | 0.9037 |
| FF | 93.9 % | 7.45 % | 5.50 % | 0.9823 | 0.8965 |
| DSP | 100.0 % | 0.00 % | 0.00 % | 1.0000 | 1.0000 |
| BRAM_18K | 100.0 % | 0.00 % | 0.00 % | 1.0000 | 1.0000 |

fits flag: accuracy 100.0 %, not-fit recall 100.0 %, not-fit precision 100.0 %

## all (n = 510)

| metric | Judgement A pass | MAPE | median APE | Spearman | Kendall |
|---|---|---|---|---|---|
| latency_cycles | 99.8 % | 0.17 % | 0.00 % | 0.9994 | 0.9968 |
| LUT | 85.9 % | 8.96 % | 4.47 % | 0.9903 | 0.9306 |
| FF | 87.8 % | 9.66 % | 5.91 % | 0.9848 | 0.9130 |
| DSP | 100.0 % | 0.00 % | 0.00 % | 1.0000 | 1.0000 |
| BRAM_18K | 97.3 % | 1.35 % | 0.00 % | 0.9916 | 0.9813 |

fits flag: accuracy 99.8 %, not-fit recall 97.4 %, not-fit precision 100.0 %

## Judgement A pass rate by group

| source | group | n | latency | LUT | FF | DSP | BRAM_18K |
|---|---|---|---|---|---|---|---|
| extra | S1a_fmul | 78 | 100 | 100 | 100 | 100 | 100 |
| extra | S1b_fadd | 40 | 100 | 100 | 100 | 100 | 100 |
| extra | S2a_pipe_k | 23 | 100 | 100 | 100 | 100 | 100 |
| extra | S2b_pipe_j | 22 | 100 | 95 | 68 | 100 | 100 |
| extra | S3a_unroll_k | 5 | 100 | 100 | 80 | 100 | 100 |
| extra | S3b_unroll_k_ports | 4 | 100 | 100 | 25 | 100 | 100 |
| extra | S3c_unroll_j | 3 | 100 | 67 | 100 | 100 | 100 |
| extra | S3d_unroll_nopipe | 5 | 100 | 60 | 80 | 100 | 100 |
| extra | S3e_unroll_unsafe | 5 | 100 | 100 | 100 | 100 | 100 |
| extra | S4a_lA | 16 | 100 | 38 | 100 | 100 | 100 |
| extra | S4b_lB | 18 | 100 | 100 | 100 | 100 | 100 |
| extra | S4c_lC | 16 | 100 | 69 | 94 | 100 | 100 |
| extra | S4d_wrong_dim | 4 | 100 | 25 | 100 | 100 | 100 |
| extra | S5a_storage_AB | 16 | 100 | 100 | 100 | 100 | 100 |
| extra | S5b_storage_A | 8 | 100 | 100 | 100 | 100 | 100 |
| extra | S6a_recurrence | 28 | 100 | 100 | 86 | 100 | 100 |
| extra | S6b_flatten | 3 | 100 | 67 | 67 | 100 | 100 |
| train | bind_op | 10 | 100 | 100 | 100 | 100 | 100 |
| train | compile_config | 10 | 100 | 90 | 100 | 100 | 100 |
| train | loop_flatten | 10 | 100 | 60 | 80 | 100 | 100 |
| train | mixed | 99 | 99 | 72 | 70 | 100 | 88 |
| train | partition | 9 | 100 | 67 | 100 | 100 | 100 |
| train | pipeline | 8 | 100 | 100 | 100 | 100 | 100 |
| train | reshape | 9 | 100 | 78 | 78 | 100 | 78 |
| train | storage | 9 | 100 | 78 | 100 | 100 | 100 |
| train | unroll | 8 | 100 | 88 | 62 | 100 | 100 |
| train | x_partition_storage | 9 | 100 | 67 | 89 | 100 | 100 |
| train | x_pipeline_bind_op | 8 | 100 | 100 | 100 | 100 | 100 |
| train | x_pipeline_clock | 8 | 100 | 100 | 88 | 100 | 100 |
| train | x_unroll_partition | 10 | 100 | 80 | 80 | 100 | 100 |
| train | x_unroll_unsafe_math | 9 | 100 | 67 | 67 | 100 | 100 |

## Generalization: leave-one-group-out (constants refit without the group)

| source | latency_cycles | LUT | FF | DSP | BRAM_18K |
|---|---|---|---|---|---|
| train | 99.5 % | 73.1 % | 74.5 % | 100.0 % | 93.5 % |
| extra | 100.0 % | 90.5 % | 91.8 % | 100.0 % | 100.0 % |

Note: only the per-unit constants of R3 are refit per fold; the written rules (L1-L5, R1, R2) were derived by inspecting all 510 samples.
