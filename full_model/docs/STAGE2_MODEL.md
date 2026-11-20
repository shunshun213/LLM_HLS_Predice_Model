# Stage 2 model: hardware structure (HS) -> latency, LUT, FF, DSP, BRAM_18K, fits

Implements `stage2_impl.md` Steps 1-2 under the rules of `Target_refined.md` (white-box, every
constant has a hardware meaning, constants may be derived from the 510 samples, headline on the
216 `source == train` samples). Code: `sanitize.py`, `latency.py`, `resources.py`, `features.py`,
`fit.py` (+ `constants.json`), `stage2.py`; evaluation `evaluate.py`, `logo.py`.

## 0. Input restriction (sanitize.py)

Stage 2 reads only the output of `sanitize()`: a whitelist of HS fields, each with a justification
string in the code. `assert_clean()` fails if any output-like key survives. Every prediction in the
evaluation goes through it.

Kept (Stage 1 must predict these from code + clock):

| Group | Fields |
|---|---|
| module hierarchy | module, instantiations, children |
| loops | trip count, pipelined, style, II_target, II_achieved, **pipeline depth**, flattened_from, level |
| fp operators | op, impl, latency, count per module, owner module, effective count |
| aux cores | class (sparsemux / shift), params (inputs, width, latency), count, owner |
| memories | storage ports, impl, banks, words/bank, bits/word, in_registers, RAM_1WNR inferred, total bits |
| access pattern | per module and array: banks accessed, read/write ports per bank and total, scalar inputs, scalar input bits |
| transforms | unroll (loop, kind, factor), array partition/reshape (array, kind, factor, dim), flatten, balanced expressions |
| timing | target clock only |
| latency_breakdown | structure only: stage order, kind (call / top loop), phase, trip counts, children |

Dropped: every latency (module, loop, iteration, breakdown, FSM overhead), FSM state bits, all
utilization numbers, per-array BRAM/LUT/FF, unit costs in the HS files, estimated clock, log depth,
II-limiter diagnostics, core-name strings.

Note on `pipeline_depth`: for a pipelined loop the depth equals the report's iteration latency.
It is in the allowed list of stage2_impl.md §2. So rule L1 is exact *given* the depth, and the
latency score on true HS measures how Stage 2 composes the hierarchy, not how it predicts the depth.
Predicting the depth is a Stage 1 obligation (§5).

## 1. Latency rules (latency.py)

- **L1, pipelined module.** Module latency = (TC − 1)·II + D + 1 (loop (TC−1)·II + D − 1, plus 2 handshake cycles).
  Exact on all 1577 pipelined loops.
- **L2, non-pipelined loop in the top FSM.** L = TC · (Σ children + per-iteration overhead).
- **L3, non-pipelined innermost body** (u copies of `sum += A*B`, no reassociation).
  - Body = u·(L_fadd + 1) + L_fmul + 2 + δ.
    - Each chained add takes L_fadd + 1 cycles.
    - The first product's latency is exposed.
    - Address and BRAM read take 2 cycles.
  - δ = read-path registers added as the clock shrinks:

    | Clock | δ |
    |---|---|
    | 10 ns | 0, +1 for a meddsp fmul |
    | 5 ns | 1 |
    | 3.33 ns | 1, +1 if an operand word is wider than 32 bits |
    | 2.5 ns | 2 |

  - If an `lshr` slicer sits on the read path, δ = max(δ, L_lshr + 1).
  - A loop_j body (k fully unrolled) takes 1 extra state.
- **L3b, reassociated body** (unsafe_math, balanced expressions). A greedy list schedule:
  - Operand reads are limited by ports × banks × useful elements per word along k.
  - n_fmul / n_fadd pipelined cores, each with 1 issue per cycle; a result is ready L + 1 cycles after issue.
  - The adder tree combines the two earliest partial sums.
- **L4, per-iteration overhead of a top loop.**
  - n calls: 1 + 2n cycles. A single loop_j call from loop_i takes 2.
  - +1 at clocks ≤ 3.33 ns when there are calls.
  - Nested non-pipelined loops: 2 for the first, +1 for each further sibling loop.
- **L4b, concurrent calls.** If every called module owns its fp cores (nothing shared), the FSM starts them together: body = max(children) + n + 1.
- **L5, hoisted reads.** Values a pipelined module takes as scalar inputs are read by the caller's FSM before the call: ⌈words / ports⌉ cycles, with ports = 1 for 1P storage and 2 otherwise.
  - An lA row inside loop_i is read every iteration.
  - Everything else is read once at FSM entry.
  - Top FSM overhead = 5 (all stages are calls) or 3 (a top loop), plus the entry hoisting.

## 2. Resource rules

Total = Σ over modules of inst × (leaf cores + Expression + Multiplexer + Register), plus Memory.
This bottom-up identity holds exactly in the reports.

- **R1, cores** (exact on 2088 of 2088 modules). The fp core table (`resources.FP_COST`) holds measured library values: DSP / FF / LUT per instance.

  | core | 10 ns | 5 ns | 3.33 ns | 2.5 ns |
  |---|---|---|---|---|
  | fadd fabric | 0 / 168 / 439 | 0 / 278 / 413 | 0 / 611 / 421 | 0 / 636 / 427 |
  | fadd fulldsp | 2 / 205 / 206 | 2 / 365 / 244 | 2 / 422 / 244 | 2 / 422 / 244 |
  | fmul fabric | 0 / 236 / 677 | 0 / 523 / 610 | 0 / 763 / 654 | 0 / 763 / 654 |
  | fmul meddsp | 1 / 144 / 268 | 1 / 326 / 302 | 1 / 425 / 300 | 1 / 425 / 300 |
  | fmul fulldsp | 2 / 144 / 119 | 2 / 166 / 141 | 2 / 197 / 143 | 2 / 254 / 152 |
  | fmul maxdsp | 3 / 143 / 140 | 3 / 165 / 145 | 3 / 199 / 149 | 3 / 199 / 149 |
  - fp cores: the characterization table (op, impl, clock) → DSP/FF/LUT (HS_dataset finding 1).
  - sparsemux (32-bit, N inputs): 4 → 20, 8 → 43, 16 → 65, 32 → 148, 64 → 273 LUT. An input count between two points is rounded up to the next one. More than 64 inputs cannot occur, because a bank count is at most 64; the code would scale linearly.
  - lshr (2.5 ns, latency 4): 3250 LUT / 2661 FF. This is a single library point that the report gives for both 512- and 2048-bit shifters.
  - shl (2.5 ns, pipelined): the identical 3250 LUT / 2661 FF. The report lists it under Expression; it places a lane and its byte enable into a word of ≥ 512 bits. It is taken from the HS aux list (Stage 1 rule X-SHL; on the true HS it comes from the extractor extension that reads the Expression table, `hs/expression_shifts.json`). It is subtracted from the Expression target before the fit. The shifter only places the lane and byte enable; the W-bit AND/OR merge with the old word stays combinational. For words written through a shl, the merge is therefore costed by its own fitted term, shl_merge_bits (LUT per bit: copy_in 1.6, compute 2.4, top 0.6), in place of that module's generic lane-merge terms (wide_inner/outer, dyn_write_bits, body_dyn_write, top_c_lane).
- **R2, memory.**
  - BRAM per bank = ⌈bits / maximum width(depth)⌉. Maximum width (bits):

    | Depth | ≤512 | 1K | 2K | 4K | 8K | 16K |
    |---|---|---|---|---|---|---|
    | Width | 36 SDP / 18 TDP | 18 | 9 | 4 | 2 | 1 |

  - True dual port (TDP) is needed when two reads or two writes hit a bank in the same cycle (1 read + 1 write fits SDP):
    - 2 accesses per bank in one module, or
    - a RAM_1WNR replica, or
    - an lA/lB value hoisted by the FSM, or
    - ≥ 2 words per bank accessed in the inlined top body.
  - RAM_1WNR replication: copies = read ports per bank − 1.
  - LUTRAM: ⌈words/64⌉·bits LUT plus bits·ports FF. Arrays in registers: bits FF.
- **R3, Expression / Multiplexer / Register.** Each module category = Σ (per-unit cost × structural quantity).
  - Each quantity is a count with a hardware meaning (features.py).
  - Each per-unit cost is a non-negative constant from least squares on all 510 samples, weighted by relative error of the sample total, (inst/T)² (fit.py).
  - Where physics fixes the cost it is not fitted: 1 FF per one-hot top FSM state, per top loop-counter bit and per hoisted data bit held by the top FSM.
  - No per-module-kind intercepts or per-kind slopes. These were removed after review rounds 1 and 2 because they hid structure.
  - The copy and compute classes keep one "module control" constant per category. It covers the ap_start/ap_done handshake, FSM and loop-exit logic present in every such module.
  - The top class has none: its control is counted by the n_loops, states and cnt_bits terms.

Fitted per-unit costs. The values below are indicative, rounded to 2–3 digits. The authoritative values are in `constants.json` and in **CONSTANTS.md**; both are regenerated by `fit.py`.

| Class / category | Quantity | Value | Meaning |
|---|---|---|---|
| copy_in / copy_out Expr LUT | one | 119 / 120 | copy-loop control: 13-bit index, exit compare, address |
| | wide_inner | 4.91 / 1.34 per bit | copy_in assembles W-bit words (lane merge); copy_out only selects lanes |
| | wide_outer | 4.85 / 4.12 per bit | read-modify-write lane select, min(W, 512) bits (byte-enable writes beyond 512) |
| | banks | 0.76 / 0.22 per bank | bank-select decode |
| copy Mux LUT | one | 72 | address/data/write-enable mux of the memory interface |
| copy Reg FF | one | 41 / 42 | index, FSM and handshake registers |
| | depth_x_32 | 0.80 / 0.99 per bit | 32-bit data register per extra pipeline stage |
| | wide_inner (copy_in) | 0.69 per bit | W-bit word assembly register |
| | wide_bits_x_depth | 0.24 / 0.32 per bit | wide word staged across extra stages |
| | banks | 1.3 / 1.4 per bank | per-bank enable registers |
| copy_in / compute / top Reg FF | shl_reg_bits | fixed 1 per bit | output register of each pipelined shl (2.5 ns): one W-bit register per shifter, Σ W = `params.total_bits` |
| copy_in / compute / top Expr LUT | shl_merge_bits | 1.6 / 2.4 / 0.6 per bit | W-bit merge of the shl-placed lane with the old word |
| compute Expr LUT | one | 25 | module control |
| | cnt_bits | 7.07 per bit | adder + compare per loop-index bit (flattened loops keep one index each) |
| | addr_ops | 0.77 per u·log2u | address logic of u unrolled accesses into one bank with compile-time constant offsets (0 when A or B is split along k) |
| | addr_partial | +7.4 per u·log2u | partial unroll (u < 64) only: the offsets k0 + t need run-time adders and compares |
| | dyn_read_pc | 11.6 per bit per cycle | part-select of narrow reshaped words (W < 1024), implemented as a combinational W-bit barrel shifter (cf. the library shifter, 3250 LUT / 512 bit = 6.3 LUT per bit, plus address and control logic); one select unit serves the II slots of an iteration |
| | dyn_read_wide | 1.07 per bit | 64:1 lane mux of whole-row words (W ≥ 1024), per element access |
| | dyn_hoist_bits | 3.0 per bit | lane select of hoisted narrow words, per element access |
| | dyn_write_bits | 2.13 per bit | read-modify-write merge of C words, per write |
| compute Mux LUT | one | 121 | module control |
| | mux_ii | 0.57 per bit | 32·P operand bits written in one of II slots (multi-source register input), II > 1 |
| | banks_sel | 3.0 per bank | bank-select fan-in |
| | II_states | 3.8 per state | pipeline FSM states |
| compute Reg FF | cnt_bits | 2.8 per bit | loop indices carried down the pipeline stages |
| | stages | fixed 1 per stage | one valid / enable bit per overlapped iteration, ⌈D/II⌉ (not fitted) |
| | depth_ov | 19.2 per stage | in-flight index, address and control bits per stage that holds another iteration, max(0, D − II), when the module owns its fp cores. The partial sums themselves sit in the fadd core's own pipeline registers, which are counted under Instance. |
| | depth_ov_ext | 3.6 per stage | same, when the fp cores are shared in the top (several loop_k copies): the cores' own pipeline registers carry the data outside the module, only index/control bits stay |
| | chain_wait | 1.00 per bit | unshared adders, II > 1: products wait for the serial add chain, 32·L_fadd·u(u−1)/2·R/II bits |
| | early_ops | 0.38 per bit | operands of those early-issued products, 32·P²/II. The P² counts operand slots summed over the issue order. At 0.38 FF per bit, fewer than half of those slots are really registered, because the scheduler reuses registers between products issued in different cycles. This magnitude is empirical and is disclosed in §5. |
| | core_regs_shared | 1.39 per bit | input/output registers of time-multiplexed cores, 32·(n_mul + n_add) |
| | P32_partial | 1.36 per bit | partial unroll: one 32-bit register per product term held while the serial chain consumes it |
| compute Reg LUT | sum_live | 0.41 per bit | shift-register (SRL) storage of long-lived running sums, 32·R·⌈D/II⌉. Weak term: the true Register LUT comes in steps (0 / 32 / 64 / 2048 / 2080), and this term explains little of it (residual 89 % of the small total). |
| top Expr LUT | n_loops | 53 per loop | loop counter, increment and exit compare |
| | body_P | 13.8 per term | address/compare logic of the inlined non-pipelined datapath (per product term) |
| | hoist_seq | 10.6 per word | per-word state, address constant and capture enable of the hoisting read sequence |
| | body_wide_sel(_wide) | 0.46 / 0.16 per bit | dynamic lane select in the inlined body (narrow / whole-row words) |
| | body_dyn_write | 1.29 per bit | C-word merge in the inlined body |
| | top_wide_bits | 1.61 per bit | wide words the top FSM selects or merges itself |
| | top_c_lane | 2.35 per bit | R lane writes into C words packed along j, written back by the top (R·W) |
| top Mux LUT | n_loops, states | 13.5 per loop, 2.85 per state | FSM next-state and enable muxing |
| | mem_arrays_banks | 40 per bank | memory-interface muxes (address/data/enable) shared by the copy, compute and top accessors |
| | shared_call_mux | 1.04 per bit | inputs of fp cores in the top that are shared by several call sites |
| | mux_body | 0.19 per bit | operand muxing of time-shared cores in the inlined body |
| | hoist_seq, hoist_reg_bits | 3.7 per word, 0.137 per bit | capture-enable muxes of the hoisted registers |
| top Reg FF | states, cnt_bits, hoist_reg_bits | fixed 1 | one-hot FSM, loop counters, and the hoisted values (rule R4). If a top loop fixes the outer index: one row of A or column of B, 64 × 32 bits; when the array is reshaped along its non-k index, the row is spread over 64 words (one lane each). For a complete reshape, the FSM keeps those 64 whole words, which is the whole array (64 × W). For a block or cyclic reshape it keeps a 2-row slice of every word (64 × 64 bits; empirical, low confidence). Otherwise the whole array (total_bits). |
| | top_cores | 55 per core | input operand registers of fp cores instantiated in the top (fewer than 2 × 32, because operands shared between cores are registered once) |
| | body_P, body_P_fast | 6.5 + 6.8 per term | inlined-datapath registers; below 10 ns intermediate values are registered |
| | top_wide_bits_fast | 1.13 per bit | wide-word register below 10 ns (chained combinationally at 10 ns) |
| | top_c_word | 0.96 per bit | W-bit C word register for lane writes along j |

- **Top FSM states.** `features.py` uses `states = body_states + 2·n_calls + 2·n_loops + 4`, where:
  - `body_states` is the L3 schedule length of an inlined non-pipelined body;
  - each call adds 2 states (start, wait for done);
  - each top loop adds 2 states (entry test, exit / increment);
  - 4 states are idle, entry, exit and done.

  These states feed Register FF at a fixed 1 FF per state (one-hot) and Multiplexer LUT at 2.86 per state.
- **fits** = LUT ≤ 70560, FF ≤ 141120, DSP ≤ 360 and BRAM_18K ≤ 432.

## 3. Stage 1 obligations (Stage 2 input = Stage 1 output spec)

Every field in §0, in particular:

- **II_achieved, depth**: HS_dataset findings 3, 4, 7.
- **Operator counts and impl, default latency**: finding 2.
- **Banks / words / bits**: from partition and reshape.
- **RAM_1WNR replication** (finding 5) and its read-port count.
- **Rule H, hoisting.**
  - Inside a pipelined module, an operand array whose index does not vary in the module's loops becomes scalar inputs. Example: A[i][k] with k fully unrolled and i fixed per call. They are read by the caller.
  - scalar_inputs = words per bank needed for one call.
  - When the array is not accessed through ports at all, it is hoisted as whole reshaped words.
- **Aux cores.** sparsemux = one per bank-selected access group, with inputs = banks. lshr = wide-word slicing at 2.5 ns.

### 3.1 Frozen input list and required precision (Step 3)

The fields in §0 are the frozen Stage 2 input list, which is the Stage 1 output spec. SENSITIVITY.md
gives the effect of an error in each field on the 216 train samples. In summary:

| field | required precision |
|---|---|
| II_achieved of compute loops | **exact**: +1 moves latency by 4 % median / 25 % at the 90th percentile; latency pass 100 → 69 % |
| fadd latency (recurrence designs) | exact: +1 → latency pass 88 % |
| pipeline depth | tolerant: ±10 % keeps latency pass at 100 %; FF moves about 2 % |
| fp core counts | **exact**: x2 doubles DSP and adds 25-45 % LUT/FF |
| fmul / fadd impl | **exact for DSP**: a wrong impl gives DSP pass 44 % |
| banks, ports per bank, RAM_1WNR | **exact for BRAM**: a missed replication or a wrong port count changes BRAM by up to 2-15x on the affected arrays |
| hoisted scalar count | tolerant: x2 changes LUT by 0 % median / 8 % at the 90th percentile |

## 4. Results on the TRUE HS

See VALIDATION.md (generated by `python3 report.py`).

## 5. Known limits (from review rounds 1-3)

- **BRAM shape.** 17 of 1530 arrays get the wrong SDP/TDP shape (×2 on that array). 10 are accessed only by the top FSM. 4 are lA 64 × 2048-bit words read on a single compute port (reshape_06/08/09, mixed_05; truth TDP). 3 are hoisted as scalar inputs (mixed_09, mixed_92, mixed_99). No port count in the HS separates them from the correctly predicted cases.
- **Latency.** 443 of 510 totals are cycle-exact and 18 are off by more than 1 %. One train sample fails: mixed_12, −12 %. It has an unsafe_math adder tree with lA reshaped along i; the list schedule under-estimates the tree. mixed_16 is at −9.9 %: its body is 9 predicted vs 10 true, with lB reshaped along k at 10 ns, a read-path register the δ rule misses.
- **Rule R4 (hoisted registers).**
  - **Words reshaped along the non-k index** (7 designs reach this branch). For a complete reshape, the FSM keeps the 64 whole words, which is the whole array: mixed_97 (2048-bit words) went from −51 % to about 0 % FF. For a block or cyclic reshape it keeps a 2-row slice, 64 × 64 bits. This is an empirical rule, supported by W = 64 (mixed_06/14/88, where it equals 64 × W), W = 128 (mixed_65) and W = 256 (S4d_wrong_dim_001). W = 512 (mixed_10) cannot be separated because two arrays are hoisted. The branch with W > 64 is low confidence and is flagged by `predict.py`. mixed_97 still has an unmodelled top lane select: 139 k LUT true against 52 predicted, because hoisted words are not included in `top_wide_bits`.
  - mixed_58 keeps lB twice (264 k vs 133 k FF; total FF −41 %).
  - S4a_lA_008..015 (lA reshaped along k) keep 1.0–2.0 k FF more than one row: top Register 3.1–4.0 k vs 2.07 k.
  - These coefficients are fixed at 1 FF per bit, so the fit cannot absorb the error.
- **Register FF of loop_j / loop_i_loop_j modules.** FF barely changes with II at large depth (about 8.6 k for II 1 to 8 at D = 326). The overlap terms approximate this.
- **Conflicting samples.** dyn_hoist_bits is equal (4096) on mixed_07 and mixed_90, but their compute Expression is 200 vs 15403 LUT, so no constant fits both. The lane-select structure differs (block reshape, 8 vs 4 scalars per bank).
- **Term support.** `fit.py` writes `term_support.json`: for every fitted term, the number of the 510 designs in which its quantity is non-zero. `predict.py` flags a design whose prediction uses a term supported by fewer than 10 designs. Terms in this group: dyn_hoist_bits (2), shl_merge_bits (top 2, compute 4, copy_in 8), body_wide_sel_wide (6), top_c_lane (6), top_c_word (7).
- **Sharing across II.** dyn_read_pc is divided by II: one part-select unit serves the II read slots of an iteration. dyn_read_wide and dyn_hoist_bits are not divided. A 64:1 lane mux of whole-row words, and the selects on hoisted words, are built per element access and are not time-shared in the reports. That assumption is supported by the 11 and 2 designs of those terms.
- **Fitted magnitudes that are high for their meaning.**
  - dyn_read_pc: 11.6 LUT per bit-cycle, about 30× a plain W→32 mux. It also absorbs address and control logic of the part-select.
  - copy wide_inner / wide_outer: about 5 LUT per bit.
  - compute cnt_bits: 7 LUT per bit, against 0 in the top, where loop control sits in n_loops.
- **Weak groups** (Judgement A pass rate, on the true HS; see VALIDATION.md for the current table):
  - train mixed: LUT 72, FF 70
  - train loop_flatten: LUT 60
  - extra S4a_lA: LUT 38
  - extra S4d: LUT 25
- **Rules tuned on few samples.** All 510 samples were inspected to write the rules, and LOGO refits only the R3 constants:
  - L4b concurrent calls: 2 designs.
  - Non-pipelined loop module: 1 design.
  - lshr cores: 11 designs, all at 2.5 ns.
  - meddsp +1: 19 designs.
- **Untested path.** Module instantiations > 1 never occur in the data.

## 6. Review log

- **Round 1:** per-kind intercepts removed; sanitizer made a strict whitelist; physical 1-FF-per-bit terms fixed; relative weighting; hoisting-sequence term; BRAM TDP rule refined.
- **Round 3:** reproducibility and the input restriction confirmed (junk-injection test into every forbidden field gives identical predictions); predict() now sanitizes its own input; rule R4 corrected for arrays reshaped along the non-k index (S4d_wrong_dim_001: 16.4 k -> 4.1 k FF, true 4.1 k); undocumented near-zero terms removed (copy_in Register LUT `fast`, top `one` terms, top Expression cnt_bits); unused per-kind features deleted; Known limits rewritten.
- **Round 2 (follow-up):** collapsing per-kind slopes lost the partial/full-unroll distinction, so it was re-added as two physical terms: addr_partial (run-time offset adders only when u < 64) and P32_partial (term registers in partially unrolled bodies).
- **Round 2:** per-kind slopes collapsed (addr_ops, mux_ii); dynamic-select terms corrected (select per access, time-shared over II); top C lane-write terms added; constants that contradicted their meaning removed (kfull_operands, sum_live FF, hoist_bits, copy_out wide_inner FF); non-pipelined module uses the L3 body; BRAM TDP rule: lC write-back counts R writes, hoisted ≥ 2 words per bank, inlined bodies only.
