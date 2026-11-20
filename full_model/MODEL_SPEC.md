# GEMM HLS model: specification of the closed model

**What it predicts.** Given a GEMM HLS source (with `#pragma HLS`) or its Tcl directive script, and a clock, the model predicts what Vitis HLS 2023.2 C-synthesis would report: latency in cycles, LUT, FF, DSP and BRAM_18K. It also predicts the `fits` flag for the xczu3eg ZU3EG device. It does not run the tool.

**How it is built.** The model is white-box and works in two stages, as required by `Target_refined.md`:

    HLS code + clock --[Stage 1: stage1_parse + stage1]--> hardware structure (HS)
                     --[Stage 2: stage2]--> latency, LUT, FF, DSP, BRAM_18K, fits

**Entry point.**

```
python3 predict.py design.cpp [--clock NS] [--trace] [--json]
python3 predict.py run.tcl [--trace] [--json]
```

Each run returns:
- the five numbers and `fits`;
- `confidence`, a list of the extrapolated or low-confidence rules that fired for this design, with values outside the data and Stage 2 terms that rest on fewer than 10 designs. It covers every rule marked low confidence in STAGE1_MODEL.md and STAGE2_MODEL.md;
- with `--trace`, a step-by-step derivation, from pragma to HS term to cost.

## 1. Scope

| item | value |
|---|---|
| Kernel | `gemm_base.cpp`: C = A·B, N = 64, float32, copy_in → loop_i/j/k → copy_out |
| Tool and device | Vitis HLS 2023.2 on xczu3eg-sbva484-1-i (70,560 LUT / 141,120 FF / 360 DSP / 432 BRAM_18K) |
| Clocks | 10 / 5 / 3.33 / 2.5 ns. Other clocks are rejected: the core latencies are measured at these four. |

The pragma space covers:
- `PIPELINE` (on loop_i, loop_j or loop_k; II; stp or flp style);
- `UNROLL` on loop_j or loop_k (any factor ≥ 1; a factor that does not divide 64 keeps u copies, flagged as extrapolated);
- `LOOP_FLATTEN off`;
- `ARRAY_PARTITION` and `ARRAY_RESHAPE` (block, cyclic or complete; factor; dim; on lA, lB or lC);
- `BIND_STORAGE` (ram_1p, ram_2p or ram_t2p; bram or lutram);
- `BIND_OP` (fmul or fadd; impl; latency);
- `config_compile -unsafe_math_optimizations`.

**Tool settings assumed.** These are the dataset's settings:
- `config_compile -pipeline_loops 0`, so there is no automatic pipelining;
- `config_array_partition -throughput_driven off`;
- unsafe_math comes from a `config_compile -unsafe_math_optimizations` line, or from `--unsafe-math`.

A design synthesized with the Vitis defaults (auto-pipelining) is outside the model.

**Robustness.**
- 5000 random pragma sets over this space are rendered as a Tcl script and as a C++ source, then parsed back by `fuzz_design_space.py`. Every one round-trips identically and gives a finite prediction.
- `test_parser.py` (26 cases) covers non-canonical but valid syntax:
  - labels on the same line as the `for`;
  - mixed-case values;
  - `#if 0` blocks and `/* */` comments;
  - line continuations;
  - Tcl options in any order;
  - an unquoted location;
  - MHz clocks;
  - several PIPELINE pragmas (the outermost wins).
- Values outside the modeled space raise a clean `UnsupportedDesign` error; `predict.py` exits with code 2 and names the construct. Examples:
  - an uncharacterized `BIND_OP` impl;
  - `ram_s2p`;
  - `dim=0`;
  - `UNROLL` on loop_i;
  - a clock other than 10 / 5 / 3.33 / 2.5 ns.
- Other constructs are ignored with a note under `confidence` and never dropped silently. Examples: unknown pragmas, and pragmas on other arrays or loops.
- The kernel itself must be unchanged. A PIPELINE other than II=1 on the copy loops, a missing copy pipeline, N ≠ 64, DATAFLOW, or a function-level PIPELINE all raise `UnsupportedDesign`.
- **Coverage flags.** `coverage.json` (built by `build_coverage.py` from the 510 dataset configs, inputs only) records which values each pragma field takes in the data. A value the data never used is reported as "outside the data"; a value used by fewer than 3 designs is reported as "sparse data".
- **Ranges for structure-changing rules.** When rule S5 or O2-par fires (each rests on 1–2 designs), `predict.py` also reports the prediction without that rule, so the user sees the range.

## 2. Hardware structure (HS): the interface between the two stages

The HS is the list of fields Stage 1 must predict and the only thing Stage 2 may read. It is enforced by `sanitize.py`, a whitelist plus `assert_clean`.

| group | fields |
|---|---|
| module hierarchy | module, instantiations, children |
| loops | trip count, pipelined, style, II_target, II_achieved, pipeline depth, flattened_from, level |
| fp operators | op, impl, latency, count per module, owner module, effective count |
| aux cores | sparsemux (inputs), shift (lshr read slicer / shl lane writer, width, latency), count, owner |
| memories | ports (1P / 2P / T2P / default / RAM_1WNR), impl, banks, words per bank, bits per word |
| access pattern | per module and array: banks accessed, read/write ports per bank, scalar inputs (hoisted values) and their bits |
| transforms | unroll (explicit or implied), partition / reshape directives, flatten, balanced expressions |
| timing | target clock |
| latency breakdown | structure only: stage order, call vs top loop, trip counts, children |

No latency, utilization, slack or tool-only number ever enters the HS.

## 3. How each pragma enters the model (Target section 6)

| pragma | Stage 1 rule(s) | HS terms | Stage 2 terms affected |
|---|---|---|---|
| `PIPELINE` loop, II, style | S1–S6, P1–P3, D, O counts, sharing (stp: ceil(P/II); flp: P) | modules, II_target, II_achieved = max(target, recurrence, memory), depth, operator counts | L1 latency (TC − 1)·II + D + 1, Register FF (depth, overlap), Multiplexer (II states, operand muxes), cores |
| `UNROLL` factor | P = u·R products per iteration, number of loop_k copies, N0 for non-divisors | operator counts, reads per bank, II_rec = u·L_fadd, R loop_k modules | cores (DSP/LUT/FF), address logic, operand muxes, L3 body latency |
| `LOOP_FLATTEN off` | S2/S3 | top loop_i + loop_j instead of a flattened nest | L4 per-iteration overhead, loop counters |
| `ARRAY_PARTITION` | bank mapping (`locate`), reachable banks, H2/H3, sparsemux | banks, banks wired, ports, aux muxes | BRAM per bank (R2), mux LUT, bank-select logic, II_memory |
| `ARRAY_RESHAPE` | lanes per word, H1–H3, X-SH / X-SHL at 2.5 ns, C read-modify-write | bits per word, scalar inputs, lshr / shl cores | BRAM width rule, lane-select logic, shifters, hoisted-register FF |
| `BIND_STORAGE` | ports 1P → 1, 2P / T2P → 2, no RAM_1WNR; T2P forces BRAM | storage ports and impl | II_memory, BRAM vs LUTRAM, TDP shape |
| `BIND_OP` | impl and latency of fmul / fadd; disables reassociation when the fadd is bound | operator impl and latency | core cost table (op, impl, clock), depth, recurrence II |
| clock | DEFAULT_LAT, default impls (O1), depth offsets, copy depths, X-SH / X-SHL | latencies, depths, shifters | core cost table, δ read registers, FF of fast-clock registers |
| `unsafe_math` | reassociation if the fadd is unbound: II_rec = L_fadd, tree depth, tree / accumulator cores | II, depth, fadd count, balanced expressions | L3b tree schedule, cores |

## 4. Stage 1 (`stage1_parse.py`, `stage1.py`)

The full rule list is in **docs/STAGE1_MODEL.md**:
- parser; N0 canonical form;
- S1–S6 structure;
- P1–P3 initiation interval;
- D depth, including D-copy;
- O1–O2 operators, including B-none and O2-par;
- M memories;
- A access pattern, including hoisting H1–H3 and C ports;
- X aux cores (sparsemux, X-SH, X-SHL);
- T transforms.

Each rule is pragma-traceable, and each constant is a tool characteristic with a hardware meaning. Examples: core latencies per clock, 2 ports per BRAM, 16 read ports per replicated bank, 4 shifter stages.

## 5. Stage 2 (`stage2.py`, `latency.py`, `resources.py`, `features.py`, `fit.py`, `constants.json`)

The full description is in **docs/STAGE2_MODEL.md**.

**Latency rules.**
- **L1:** pipelined module.
- **L2:** top loops.
- **L3 / L3b:** non-pipelined body; reassociated tree schedule.
- **L4:** loop overheads.
- **L4b:** concurrent calls.
- **L5:** hoisted reads.

**Resource rules.**
- **R1, characterized cores:**
  - fp cores: (op, impl, clock) table;
  - sparsemux: by number of inputs;
  - lshr / shl: 2661 FF / 3250 LUT.
- **R2, memory:** BRAM aspect-ratio rule, TDP shape, RAM_1WNR copies, LUTRAM.
- **R3, Expression / Multiplexer / Register:** a sum of structural quantities times per-unit constants:
  - per-unit constants are non-negative, from least squares weighted by relative error;
  - constants fixed by physics are not fitted (1 FF per bit: FSM state, loop-counter bit, hoisted bit, shl output bit).
- **R4:** hoisted registers.

## 6. Results (reports/FINAL_VALIDATION.md, generated by `final_report.py`)

End to end on **train (216 designs)**. Session target: MAPE below 10 % for latency and below 20 % for each resource.

| metric | MAPE | target | median APE | pass A (≤ 10 % / ≤ 20 %) | Spearman |
|---|---|---|---|---|---|
| latency | **1.3 %** | < 10 % | 0.0 % | 98.1 % | 0.999 |
| LUT | **12.7 %** | < 20 % | 8.4 % | 76.9 % | 0.988 |
| FF | **13.2 %** | < 20 % | 7.8 % | 78.2 % | 0.985 |
| DSP | **1.4 %** | < 20 % | 0.0 % | 97.7 % | 0.995 |
| BRAM_18K | **3.2 %** | < 20 % | 0.0 % | 93.5 % | 0.988 |

- **fits flag:** 99.5 % accuracy. On the not-fit class (29 designs), precision is 100 % and recall is 96.6 %.
- **LOGO of the Stage 2 constants:** MAPE latency 1.3 %, LUT 17.1 %, FF 15.7 %, DSP 1.4 %, BRAM 3.2 %.
- **Constants fitted on the extra sweeps only:** LUT 19.1 %, FF 17.3 % on train.
- **Main risk: the `mixed` group** (99 designs, the most test-like). Its LUT / FF MAPE is 16.0 / 18.2 % in-sample and 25.2 / 22.6 % under LOGO (FINAL_VALIDATION §4). Its bias is negative: LUT −7.4 %, FF −11.9 %.
- **Effect of the confidence flags.** Designs with a low-confidence flag (76) have LUT / FF MAPE of 17.9 / 18.9 %; designs without one (140) have 9.9 / 10.1 %.

**Scope of these numbers.** The Stage 1 rules and Stage 2 constants were derived from all 510 designs, so these numbers are in-sample. The hidden test set is the real check, and `confidence` flags designs that fall outside the supported rules.

## 7. Files

The folder layout is in README.md. Rule details are in docs/STAGE1_MODEL.md and docs/STAGE2_MODEL.md; generated reports are in reports/; the review history is in docs/REVIEW_LOG.md.
