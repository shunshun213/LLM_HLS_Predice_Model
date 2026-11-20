# Stage 2 input sensitivity (Step 3 of stage2_impl.md)

Perturbation applied to every train sample (216) on the sanitized TRUE HS. Cells: median / 90th percentile of |relative change| of the prediction, then the Judgement-A pass rate after the perturbation (baseline in the first row).

| perturbation | latency_cycles | LUT | FF | DSP | BRAM_18K |
|---|---|---|---|---|---|
| baseline (true HS) | pass 100 % | pass 77 % | pass 80 % | pass 100 % | pass 94 % |
| II_achieved +1 (compute loops) | 4.2 / 24.8 % (pass 69 %) | 0.2 / 2.1 % (pass 76 %) | 0.2 / 2.9 % (pass 76 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 94 %) |
| pipeline depth +1 | 0.0 / 0.7 % (pass 100 %) | 0.0 / 0.0 % (pass 77 %) | 0.1 / 1.3 % (pass 80 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 94 %) |
| pipeline depth +10 % | 0.5 / 5.7 % (pass 100 %) | 0.0 / 0.9 % (pass 78 %) | 0.9 / 6.0 % (pass 81 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 94 %) |
| fadd latency +1 | 0.0 / 13.2 % (pass 88 %) | 0.0 / 0.5 % (pass 77 %) | 0.0 / 2.5 % (pass 80 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 94 %) |
| fp core count x2 | 0.0 / 0.0 % (pass 100 %) | 25.2 / 70.6 % (pass 38 %) | 44.6 / 74.3 % (pass 35 %) | 100.0 / 100.0 % (pass 3 %) | 0.0 / 0.0 % (pass 94 %) |
| fmul impl swapped | 0.0 / 0.0 % (pass 99 %) | 1.1 / 8.4 % (pass 69 %) | 0.1 / 7.2 % (pass 75 %) | 25.0 / 200.0 % (pass 44 %) | 0.0 / 0.0 % (pass 94 %) |
| lB banks x2 | 0.0 / 0.0 % (pass 100 %) | 0.9 / 8.0 % (pass 77 %) | 0.1 / 1.3 % (pass 80 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 65.2 % (pass 69 %) |
| read ports per bank +1 | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 77 %) | 0.0 / 0.0 % (pass 80 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 54.4 % (pass 81 %) |
| hoisted scalar count x2 | 0.0 / 3.2 % (pass 100 %) | 0.0 / 7.5 % (pass 79 %) | 0.0 / 0.0 % (pass 80 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 93 %) |
| RAM_1WNR replication missed | 0.0 / 0.0 % (pass 100 %) | 0.0 / 0.0 % (pass 77 %) | 0.0 / 0.0 % (pass 80 %) | 0.0 / 0.0 % (pass 100 %) | 0.0 / 77.8 % (pass 72 %) |
