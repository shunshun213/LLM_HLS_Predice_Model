# Brief for reviewers of the closed GEMM HLS model

**Goal (`Target_refined.md`).** Predict what Vitis HLS 2023.2 C-synthesis reports for GEMM designs: latency, LUT, FF, DSP, BRAM_18K and fits. The designs are N = 64 on the ZU3EG; the input is HLS code and a clock. The model must be a white-box, two-stage model:
- Stage 1: code + clock → hardware structure (HS);
- Stage 2: HS → numbers.

**Session acceptance target.** On the 216 train designs, end to end (code → Stage 1 → Stage 2), the MAPE must be below 10 % for latency and below 20 % for each of LUT, FF, DSP and BRAM_18K. The user chose the MAPE reading of the target.

## Hard rules

1. **White-box only.** Use explicit rules and formulas; every term and constant must have a hardware meaning. No black-box regressor, and no lookup by sample name or per-sample table. Non-negative per-unit constants fitted by least squares on quantities with a physical meaning are allowed. This is "dataset used to obtain constants".
2. **Pragma-driven.** You must be able to trace how each pragma changes the prediction (Target section 6). `predict.py --trace` is the mechanism for this.
3. **Stage 2 reads only the sanitized HS** (`sanitize.py` whitelist). It never reads latencies, utilization, slack or other tool outputs.
4. **No tool output on the prediction path.** Stage 1 reads only the code (pragmas or Tcl) and the clock.
5. **Deliverables (Target section 8):**
   - a spec document;
   - a reference implementation: HS extractor, Stage 1, Stage 2;
   - a validation report: stage-wise, end to end, Judgement A / B and fits.

## Package

Location: `/home/claude/model/`; the data is in `data/`.

**Read first:** `MODEL_SPEC.md`, then `STAGE1_MODEL.md`, `STAGE2_MODEL.md` and `FINAL_VALIDATION.md`.

**Run:**
- `python3 final_report.py` (about 90 s);
- `python3 predict.py <file.cpp|.tcl> [--clock NS] --trace`;
- `python3 fuzz_design_space.py 2000`.

To get an example .cpp or .tcl: in `data/stage1_dataset.jsonl`, each row's `inputs` holds `source_with_pragmas` and `tcl`.

## Changes in this session (scrutinize these)

- **Stage 1 additions:**
  - N0 canonical form;
  - S5: a dropped flp pipeline, supported by 1 design;
  - S6: pipeline on loop_i, extrapolated;
  - X-SHL: pipelined shl at 2.5 ns;
  - a parser hardened for real-world pragma syntax.
- **Stage 2 changes:**
  - the shl expression is costed with the characterized core cost (2661 FF / 3250 LUT), subtracted from the Expression fit target, and it replaces the fitted lane-merge terms;
  - `shl_reg_bits` fixed at 1 FF per bit;
  - `depth_ov` is split into `depth_ov` (module owns its fp cores) and `depth_ov_ext` (cores shared in the top).
- **True-HS extension:** the shifts in the Expression table (`extract_expression_shifts.py` → `data/expression_shifts.json`) are injected into the true HS by `evaluate.load()`.
- **New files:** `predict.py` (entry point with confidence flags and trace), `final_report.py` and `fuzz_design_space.py`.

## Report format

Give a verdict: ACCEPT, ACCEPT WITH FIXES, or REJECT. Then list numbered BLOCKERs and NON-BLOCKERs. Each must come with evidence (file:line and numbers) and a concrete fix.
