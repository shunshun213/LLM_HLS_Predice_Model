# Review log (closed-model review rounds)

Earlier rounds:
- Stage 2: 3 rounds, recorded in STAGE2_MODEL.md §5.
- Stage 1: 2 rounds, recorded in STAGE1_MODEL.md.

The rounds below cover the closed model: `predict.py` and Stage 1 + Stage 2 end to end.

## Round 1a: rules and compliance, ACCEPT WITH FIXES

The reviewer confirmed:
- no leakage on the prediction path;
- the injection of shift expressions into the true HS is legitimate ground-truth extraction;
- the headline numbers reproduce independently.

| # | finding | resolution |
|---|---|---|
| B1 | The shl output-register width was miscounted: entries merged per owner took width = the widest shifter. | The HS now carries `params.total_bits` (sum of output widths), from both Stage 1 and the extractor injection. `shl_reg_bits` uses it. |
| B2 | Non-divisor unroll was rounded to the next divisor and called an HLS semantic. | It is now modeled as the tool does it: u copies, trip count ⌈64/u⌉, guarded last iteration. Stage 2 reads u from `transforms_applied` instead of 64/TC. It is flagged as extrapolated. |
| N1 | Zeroing the lane-merge terms under-predicted Expression LUT on 2048-bit words. | New fitted term `shl_merge_bits`: the W-bit merge stays next to the shifter. |
| N2 | The X-SHL physical meaning was wrong. | Restated as data shifter plus mask shifter; the two misses are explained (a 512-bit data shifter pipelined). |
| N3 | S5 is supported by one design. | The doc states the untested range: threshold between 1.2 % and 5 %; the flp restriction is untested. |
| N4 | Confidence flags conflated lshr and shl. | Flags are now separate for X-SH and X-SHL; the trace prints the op. |
| N5 | A missing `expression_shifts.json` was skipped silently. | It is now an error. |
| N6 | `dim=0` was read as dim 1. | It now raises `UnsupportedDesign` (all dimensions is outside the modeled space), in both parsers. |
| N7 | `predict.py` could import stale sibling folders. | Only the package folder is on the path. |
| N8 | `REVIEW_LOG.md` was missing. | This file. |

## Round 1b: evaluation and robustness, ACCEPT WITH FIXES

The reviewer recomputed the headline numbers independently and they agree:
- MAPE at that time: latency 1.27 %, LUT 12.76 %, FF 13.41 %, DSP 1.41 %, BRAM 3.20 % (the current values are in FINAL_VALIDATION.md; later fixes moved LUT/FF by less than 0.05 points);
- 95 % bootstrap range of the LUT MAPE: 11.0–14.5 %.

| # | finding | resolution |
|---|---|---|
| B1 | The parser misread valid syntax silently or crashed. | Rewrote `stage1_parse.py`:<br>• values are lower-cased;<br>• Tcl is tokenized with shlex (location quoted or unquoted, options in any order);<br>• the outermost PIPELINE wins, with a note;<br>• `/* */`, `#if 0` and line continuations are handled;<br>• unsafe_math is read only from a `config_compile` line or `--unsafe-math`;<br>• `-period` accepts MHz;<br>• unknown values raise `UnsupportedDesign`, and `predict.py` exits with code 2 and a message.<br>`test_parser.py` covers 21 cases, all passing. |
| B2 | The report hid the generalization risk of the `mixed` group. | FINAL_VALIDATION §4 now gives per-group LOGO and an extra-only fit, states the risk, and splits the error by confidence flag. |
| N1 | Non-divisor unroll. | Fixed in round 1a (u copies, ⌈64/u⌉ iterations). |
| N2 | The fuzz test bypassed the parser and produced impossible configs. | The fuzz now renders each design to Tcl and to source and parses both back (they must agree). Default storage no longer gets an impl. |
| N3 | Tool settings were undocumented. | MODEL_SPEC §1 states them; `--unsafe-math` flag added. |
| N4 | Docs did not match the code. | REVIEW_LOG added; numbers synced from FINAL_VALIDATION; the design-space wording is precise. |
| N5 | Low-confidence rules drive the worst errors. | Acknowledged. They are flagged under `confidence` (FINAL_VALIDATION §4 gives the split); more data would be needed to firm them up. |

## Round 2a: full audit, ACCEPT WITH FIXES

The reviewer checked:
- the parser against `inputs.config`: 510/510 exact for Tcl, 216/216 for source;
- the constants: reproduced exactly;
- the headline numbers: unchanged at one decimal;
- leakage and white-box rules: none found.

| # | finding | resolution |
|---|---|---|
| B1 | `resources._top_body_accesses` still derived u as 64 // TC, which is wrong for a non-divisor unroll. | It now uses `features.unroll_factor`. |
| N1 | Brace-quoted Tcl location; `-unsafe_math_optimizations=false`; `-period "5"`; no check of the tool settings. | Braces are stripped. The unsafe_math value is parsed. The clock may be quoted. A Tcl script without `config_compile -pipeline_loops 0` gets a note. |
| N2 | Parser notes were duplicated and mislabelled "extrapolated". | `normalize` keeps its own notes in `_norm_notes`. |
| N3 | `predict.py` caught every `ValueError`. | One exception type, `stage1.UnsupportedDesign`, is used for both the parser and the clock domain, and it is the only exception caught. |
| N4 | Stale numbers in the docs. | `fit.py` now writes CONSTANTS.md, which is authoritative. Stale values in STAGE2_MODEL.md are corrected, MODEL_SPEC §6 is synced, and the 1a-N6 entry is corrected. |
| N5 | The fuzz compared only Tcl against source. | The parsed config is now also compared with an independently written canonical form of the random design. |

## Round 2b: adversarial designs, ACCEPT WITH FIXES

The reviewer ran 80 hand-written designs, a re-render of the 510 configs in their own syntax, and 3000 single-pragma checks. Nothing crashed, and no physics or monotonicity violation was found inside the supported region.

| # | finding | resolution |
|---|---|---|
| B1 | "all rules supported" was reported for values the data never used. | Added `coverage.json` and `build_coverage.py`. `predict.py` flags each value as "outside the data" or "sparse data", and also flags pairs of pipeline loop with unroll / II / style. |
| B2 | Copy-loop pragmas and N were not checked. | Only PIPELINE II=1 is accepted on copy_in_j / copy_out_j, and both must be present (source path). N must be 64, in `#define` or `-DN`. |
| N1 | Rules resting on 1–2 designs cause large jumps between neighbouring designs. | When S5 or O2-par fires, the prediction without the rule is reported too (`without_low_support_rules`, printed as a range). |
| N2 | DATAFLOW and function-level PIPELINE were accepted with only a note. | Both raise `UnsupportedDesign`. The duplicated parser notes were fixed in 2a. |
| N3 | Negative bias on `mixed` (LUT −7.5 %, FF −12.3 %). | Bias per group added to FINAL_VALIDATION §4 and stated in the risk paragraph. The fit was not reweighted, which avoids tuning to the in-sample train group. |
| N4 | `--clock 3.3` was rejected. | Tolerance widened to 0.05 ns. |
| N5 | ns latency is not monotone in the clock for fadd-chain-limited loops. | Explained in STAGE1_MODEL.md: the measured fadd latency jumps from 5 to 12 cycles. |

`test_parser.py`: 26/26 pass.

## Round 3a: Target compliance, ACCEPT WITH FIXES

The requirements of Target_refined §1–8 were checked one by one, and all were met or met in part. The numbers reproduce exactly.

| # | finding | resolution |
|---|---|---|
| B1 | `--trace` stopped at category totals, so LUT/FF could not be traced to pragmas. | The trace now lists every R3 term as quantity × per-unit constant per module and category, and prints the operator arithmetic: P = u·R, II, the stp/flp sharing formula and reassociation. |
| B2 | Operator counts deviate from the Target formulas without documentation. | STAGE1_MODEL §O has a "deviations from the Target formulas" paragraph with evidence: (a) the shared tree/accumulator pool, low confidence, flagged in `confidence` when u > II; (b) loop_j reassociation, 63·R adds; (c) flp does not share. |
| N1 | The top FSM state count was undocumented. | Documented in STAGE2 §2, with the meaning of each term. |
| N2 | The measured tables were only in code. | The DEFAULT_LAT table is in STAGE1 §O; the FP_COST table is in STAGE2 R1. |
| N3 | sparsemux behaviour above 64 inputs. | Documented: it cannot occur (at most 64 banks); the code would scale linearly. |
| N4 | The test count and the δ docstring were stale. | Fixed. |
| N5 | The limiter label read "none" when II = target = recurrence. | It now reads "target = recurrence" or "target = memory". |

## Round 3b: clean-room reproduction, ACCEPT (no blockers)

The package was copied with `cp -rL`, and the 8 scripts were run from the clean copy. All outputs are byte-identical and there is no hidden dependency (checked with strace).

| # | finding | resolution |
|---|---|---|
| N1 | Stale numbers in the docs: test count, loop_j depth count, DSP-miss count, the D-section branch support, the B-none wording, the STAGE2 weak groups, SENSITIVITY.md, and the round-1b numbers. | All corrected. SENSITIVITY.md is regenerated, and `sensitivity.py` is added to the regeneration list in README. |
| N2 | The fuzz test exited 0 even when everything failed. | The exit code is now 0 only if every design is ok. |
| N3 | There was no README. | README.md added: install, run, inputs and outputs, JSON schema, exit codes 0/1/2/3, regeneration order. |
| N4 | A missing file produced a traceback. | Now exit code 3 with a message. |
| N5 | The clock and unsafe_math were read inside `/* */` and `#if 0`. | Only line-anchored commands outside block comments and disabled code are read; `--clock` overrides. |

## Round 4a: fresh full review, ACCEPT (no blockers)

The reviewer checked:
- the acceptance target, with their own script: MAPE latency 1.27 %, LUT 12.78 %, FF 13.36 %, DSP 1.41 %, BRAM 3.20 % (historical; the values before the 4b fix);
- the round-3 fixes;
- the exit codes: every case tested;
- the JSON schema: 510/510;
- leakage: the only files opened are `constants.json` and `coverage.json`.

| # | finding | resolution |
|---|---|---|
| N1 | A file that is not UTF-8 gave exit code 1 with a traceback. | Now exit code 3. |
| N2 | The reassociation arithmetic in the trace was only a label. | The trace prints the formula actually applied: shared pool ceil(u/II), tree ceil((u−1)/II) + 1 accumulator, or 63·R. It marks counts as per loop_k copy. |
| N3 | Latency trace. | Each pipelined module is printed with L1 = (TC−1)·II + D + 1, and top loops with L2. |
| N4 | Some low-confidence rules had no flag. | Added flags for the D u=1 loop_k offset and the C-store serialisation; the claim is reworded. |
| N5 | Stale constants in STAGE2. | Fixed; CONSTANTS.md stays authoritative. |
| N6 | README gaps. | Clock and config lines read from `//` comments are documented; `--no-unsafe-math` added. |

## Round 4b: physics of the rules, ACCEPT WITH FIXES

| # | finding | resolution |
|---|---|---|
| B1 | The R4 "2-row slice" branch rested on one sample (S4d_wrong_dim_001) and broke mixed_97 (−51 % FF). | R4 now uses the physical rule 64 × W: the row is spread over 64 words, and the FSM keeps those words. mixed_97 goes to about 0 % FF; S4d_001 is disclosed as the exception. Train FF MAPE is 13.4 → 13.2 %. |
| N1 | dyn_hoist_bits rests on 2 conflicting designs and was never flagged. | General mechanism added: `fit.py` writes `term_support.json` (designs per fitted term), and `predict.py` flags any term with fewer than 10 supporting designs that the design uses. |
| N2 | Unflagged low-support branches: S3 C row word, S4. | Both are labelled in STAGE1 and flagged at run time; the C-store flag was added in 4a. |
| N3 | Fitted magnitudes against their meaning. | `depth` (0.49 FF per stage) is replaced by `stages` = ⌈D/II⌉ at a fixed 1 FF each (no fit). depth_ov is restated as index / address / control bits, since the sums sit in the fadd core. early_ops' magnitude is explained and disclosed. dyn_read_pc is restated as a combinational barrel shifter (library point 6.3 LUT per bit). The sum_live weakness is disclosed. |
| N4 | Rules looked inconsistent with each other. | Documented: a chained add costs La in a k pipeline (in-core recurrence) and La + 1 elsewhere (registered at stage boundaries); dyn_read_pc is time-shared over II, while the per-element 64:1 and hoisted selects are not. The P32_partial comment is fixed. |
| N5 | Depth branches not flagged. | The u=1 loop_k flag was added in 4a. The ijk branches are documented as in effect keyed on default versus bound fmul latency. |
| N6 | Flag texts. | X-SH says 11 fire / 7 exact; B-none covers the fadd = 2 branch. |

## Round 5: verification of the round-4 fixes, ACCEPT WITH FIXES

The reviewer confirmed:
- the acceptance target, with their own script: train MAPE latency 1.27 %, LUT 12.75 %, FF 13.20 %, DSP 1.41 %, BRAM 3.20 %;
- term_support: recomputed independently, 0 mismatches;
- stages fixed at 1 FF;
- the trace: 1577 L1 lines correct, and the components sum to the latency;
- exit codes and `--no-unsafe-math`;
- byte-identical regeneration.

| # | finding | resolution |
|---|---|---|
| B1 | R4 64 × W over-predicts block/cyclic reshape with W > 64 (S4d_001 +62 % FF, mixed_65, mixed_10). The disclosure "one exception" was wrong, and nothing was flagged. | The rule is now: complete reshape keeps the whole array (64 × W); block/cyclic reshape keeps a 2-row slice (64 × 64). It is documented with its support (W = 64, 128, 256; W = 512 not separable). The W > 64 branch is flagged at run time (3 designs). Extra FF MAPE 7.7 → 7.5 %. |
| N1 | The C-store flag fired on 43 designs, but the rule changes ports on only 17. | The flag now requires ≥ 2 stores per C bank in the pipelined module: 17 designs. |
| N2 | The D u=1 flag had no clock test. | Below 10 ns only: 8 modules in 3 train designs. |
| N3 | The S3 flag fired on a flatten-off design. | It now requires flatten not off: 2 designs. |
| N4 | The trace wording "per loop_k copy" was misleading for a shared pool. | It now states that the copies share one pool in gemm and that the counts are for the whole pool. |
| N5 | Stale table values (cnt_bits, Mux states) and round-4a numbers not marked historical. | Fixed. |

## Round 6: final verification, ACCEPT (no blockers)

The reviewer confirmed:
- R4 is within 3 % of the true top Register FF on all 7 designs that reach the branch;
- every flag's firing count matches its text;
- the shared-pool trace wording is correct (46 designs, 0 mismatches);
- regeneration is byte-identical;
- the acceptance target is met: train MAPE latency 1.27 %, LUT 12.74 %, FF 13.20 %, DSP 1.41 %, BRAM 3.20 %;
- the tests pass.

Cosmetic items:
- The X-SH flag text now says 10 lshr designs.
- The round-5 LUT figure, 12.75 %, differs from this round's 12.74 % only by rounding; both are 12.7 % as published.

Review summary for the closed model: 6 rounds, with 10 independent reviewer sessions. Every blocker raised was fixed and verified in a later round.
