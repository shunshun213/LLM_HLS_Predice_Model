# GEMM HLS model: predict Vitis HLS 2023.2 C-synthesis results without running the tool

White-box, two-stage model:

    HLS code + clock --Stage 1--> hardware structure --Stage 2--> latency, LUT, FF, DSP, BRAM_18K, fits

The specification is in `MODEL_SPEC.md`, the results are in `reports/FINAL_VALIDATION.md`, and the review history is in `docs/REVIEW_LOG.md`.

## Folder layout

| path | contents |
|---|---|
| `predict.py` | entry point |
| `stage1_parse.py`, `stage1.py` | Stage 1: code + clock → hardware structure |
| `sanitize.py`, `stage2.py`, `latency.py`, `resources.py`, `features.py` | Stage 2: hardware structure → numbers |
| `constants.json`, `coverage.json`, `term_support.json` | fitted constants, data coverage and term support, all used by `predict.py` |
| `fit.py`, `evaluate.py`, `report.py`, `logo.py`, `sensitivity.py`, `stage1_report.py`, `rule_support.py`, `final_report.py`, `build_coverage.py` | rebuild and validation scripts |
| `test_parser.py`, `fuzz_design_space.py` | tests |
| `hs_extract.py`, `extract_expression_shifts.py` | HS extractor (tool reports → true HS) |
| `gemm_base.cpp` | the kernel |
| `MODEL_SPEC.md` | specification |
| `docs/` | STAGE1_MODEL.md, STAGE2_MODEL.md (rule details), REVIEW_LOG.md, REVIEW_BRIEF.md |
| `reports/` | FINAL_VALIDATION.md (headline), VALIDATION.md (Stage 2 on true HS), VALIDATION_STAGE1.md, SENSITIVITY.md, CONSTANTS.md (all generated) |
| `data/` | stage1_dataset.jsonl, stage2_dataset.jsonl, expression_shifts.json (copied from full_dataset/hs; needed only to rebuild and validate) |

## Install

Python 3.9 or later, with `numpy` and `scipy` (`pip install numpy scipy`). No other dependency.

## Predict one design

```
python3 predict.py design.cpp --clock 5            # GEMM source with #pragma HLS
python3 predict.py run.tcl                          # Tcl directive script (clock from create_clock)
python3 predict.py design.cpp --clock 2.5 --trace   # step-by-step derivation, pragma -> structure -> cost
python3 predict.py design.cpp --clock 5 --json      # machine-readable
python3 predict.py design.cpp --clock 5 --unsafe-math
```

**Inputs.**
- `design.cpp` is `gemm_base.cpp` with `#pragma HLS` directives added, or `run.tcl` is a `set_directive_*` script for it.
- The clock is 10, 5, 3.33 or 2.5 ns. Pass it with `--clock`, or put a `create_clock -period` line in the file.
- The tool settings assumed are `config_compile -pipeline_loops 0` and `config_array_partition -throughput_driven off`. For unsafe_math, use `--unsafe-math`, or a `config_compile -unsafe_math_optimizations` line; `--no-unsafe-math` overrides the file.
- In a .cpp, `create_clock -period ...` and `config_compile ...` are also read from `//` comment lines of their own, which is how the dataset sources carry them. Lines inside `/* */` or `#if 0` are not read, and `--clock` overrides the file.

**Outputs.**
- `latency_cycles`, `LUT`, `FF`, `DSP`, `BRAM_18K` and `fits` (all four resources within the ZU3EG).
- `confidence`: every extrapolated or low-confidence rule that fired, plus pragma values the 510 dataset designs never used or rarely used. A design with no flag is printed with "all rules used are supported by the dataset".
- `without_low_support_rules`, when present: the prediction without rule S5 or O2-par, shown as a range.
- With `--trace`, the derivation lines.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | prediction printed |
| 1 | internal error (a bug; please report it with the design) |
| 2 | the design is outside the modeled space; the message names the construct (e.g. an uncharacterized BIND_OP impl, `ram_s2p`, `dim=0`, a clock outside the four, N ≠ 64, DATAFLOW, no clock given) |
| 3 | the design file cannot be read |

**JSON schema** (`--json`):

```
{"latency_cycles": int, "LUT": int, "FF": int, "DSP": int, "BRAM_18K": int, "fits": bool,
 "confidence": [str, ...],
 "without_low_support_rules": {"rules": [str], "latency_cycles": int, "LUT": int, "FF": int,
                               "DSP": int, "BRAM_18K": int, "fits": bool},   # only when S5 / O2par fired
 "trace": [str, ...]}                                                          # only with --trace
```

`predict.py` needs only the Python standard library. `fit.py` and the reports also need numpy and scipy.

From Python:

```python
import stage1_parse, predict
cfg = stage1_parse.parse_source(open('design.cpp').read(), clock_ns=5)
print(predict.predict(cfg, with_trace=True))
```

## Rebuild and validate

The data is in `data/`: `stage1_dataset.jsonl`, `stage2_dataset.jsonl` and `expression_shifts.json`, copied from `full_dataset/hs`. `predict.py` does not need it. Generated reports go to `reports/`.

```
python3 build_coverage.py      # coverage.json      (design-space coverage of the data, inputs only)
python3 fit.py                 # constants.json, term_support.json, reports/CONSTANTS.md
python3 report.py              # reports/VALIDATION.md (Stage 2 on the true HS)
python3 logo.py                # leave-one-group-out of the Stage 2 constants
python3 stage1_report.py       # reports/VALIDATION_STAGE1.md (Stage 1 HS exact match, rule support)
python3 sensitivity.py         # reports/SENSITIVITY.md (effect of HS field errors on the metrics)
python3 final_report.py        # reports/FINAL_VALIDATION.md (end to end, the headline numbers)
python3 test_parser.py         # parser robustness tests
python3 fuzz_design_space.py 5000   # design-space robustness
```

True HS from tool reports (ground truth for Stage 1, input for the Stage 2 checks):

```
python3 hs_extract.py <dataset_root> <out_dir>
python3 extract_expression_shifts.py <dataset_root>/samples <out_dir>/expression_shifts.json
```
