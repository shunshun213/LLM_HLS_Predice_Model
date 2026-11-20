# Stage 1 model: HLS code + clock → hardware structure (HS)

Code: `stage1_parse.py` (pragma / Tcl parser), `stage1.py` (rules), `stage1_report.py` (validation).
Output: an HS in exactly the sanitized format that Stage 2 reads (`sanitize.py` whitelist), so the
Stage-2 input list stays frozen. `stage2.predict` re-sanitizes whatever it gets.

White-box rules. Each rule is a deterministic function of:
- the pragmas (`PIPELINE`, `UNROLL`, `ARRAY_PARTITION` / `ARRAY_RESHAPE`, `BIND_STORAGE`, `BIND_OP`,
  `LOOP_FLATTEN`, `config_compile -unsafe_math_optimizations`);
- the clock.

No tool report is read on the prediction path. All index arithmetic is the GEMM loop nest
`C[i][j] += A[i][k] * B[k][j]` with N = 64, plus the Vitis bank/lane mapping.

**Scope.** The rules below were derived by inspecting all 510 designs; most measured offsets were
characterized on the extra sweeps (the development set). No Stage-1 accuracy figure is
out-of-sample. Rule branches supported by fewer than ~10 designs are marked **[low confidence]**;
`VALIDATION_STAGE1.md` section 5 gives the support count for each branch.

**Domain.** Clocks 10 / 5 / 3.33 / 2.5 ns only; any other clock raises an error, because the
core latencies and depth offsets are tool measurements at these clocks.

Notation:
- u = k-unroll factor; R = j-unroll factor; T = clock (10 / 5 / 3.33 / 2.5 ns).
- La / Lm = fadd / fmul latency. Lt = latency of the fadd in the reassociated tree.

## 0. Parse (`stage1_parse.py`)

Reads the source pragmas, or the Tcl directives, and normalizes them into one cfg dict.

Implied settings:
- `PIPELINE` on loop_j implies `unroll_k = full`. On loop_i it would also imply `unroll_j = full`.
- `BIND_OP` with an impl and no latency gives `latency = auto`.

Check: 0 mismatches against the dataset config, on 510 designs from Tcl and 216 designs from source.

**Parser robustness.** The C++ front end handles the following without failing; an ignored item is recorded as a note that `predict.py` reports:
- pragmas apply to the innermost enclosing labeled loop, found by tracking brace depth;
- labels may stand alone or sit in front of the `for`;
- keywords and option names are case-insensitive;
- the old `ARRAY_PARTITION ... complete` syntax is accepted;
- `BIND_OP` without `op=` is resolved from the variable (`prod` → fmul, `acc` → fadd);
- unknown pragmas, pragmas on other arrays and loops, and `PIPELINE off` are ignored with a note.

## N0. Canonical form (`stage1.normalize`)

Each rewrite below is a documented HLS semantic and is recorded in `cfg['_notes']`:
- An unroll factor u that does not divide 64 keeps u body copies. The loop runs ⌈64/u⌉ iterations, and the copies in the last iteration are guarded by exit checks. The guard logic itself is not costed. [extrapolated: no such design in the data]
- `PIPELINE` on a loop that is fully unrolled has no loop left and is dropped. This covers loop_k with u = 64 and loop_j with R = 64. [extrapolated]
- `PIPELINE` on loop_j fully unrolls loop_k; `PIPELINE` on loop_i fully unrolls loop_j and loop_k.

## S. Structure (modules, loops, stages)

| rule | condition (pragma-traceable) | structure | hardware meaning |
|---|---|---|---|
| S4 [low confidence: 4 train designs; flagged] | A or B is completely reshaped on dim 2 **and** stored `ram_1p` | `copy_in` top loop calls `copy_in_j` pipeline | The whole-row word is built by read-modify-write on one port. The inner loop carries a dependence on that word, so the two copy loops cannot be merged. |
| S4' | otherwise | flattened `copy_in_copy_in_j` | perfect nest, II = 1 |
| S2 | pipeline k, R = 1, flatten not off, C not partitioned | one pipeline `loop_i_loop_j_loop_k` | perfect nest; the tool flattens it |
| S3 [the C-row-word branch: low confidence, 2 train designs; flagged] | pipeline k, other cases | R modules `loop_k`, `loop_k1`, … called from a top `loop_i_loop_j`, or from `loop_i` + `loop_j` when flatten is off or C is a whole-row word | R copies of the k pipeline after j-unroll; a partitioned C or a row word breaks the perfect nest |
| S1 | pipeline j, A complete (partition or reshape, either dim), R = 1, flatten not off | `loop_i_loop_j` pipeline | Every A element is in its own bank or lane, so no per-i row hoist into registers is needed and i, j flatten (13/13 designs; the cause is the missing hoist, not the dimension) |
| S1' | pipeline j, other cases | `loop_j` pipeline inside a top `loop_i` | A row hoisted once per i |
| S0 | no pipeline | `loop_i`, `loop_j` in the top FSM, plus R `loop_k` loops if u < 64 | sequential FSM |
| Sc | always | flattened `copy_out_copy_out_j` | |
| S5 | flp pipeline, TC ≤ 2, overlap (D − II) < 2 % of D | the loop module stays, **not pipelined**: 1 fmul + 1 fadd, one access per cycle | Pipelining would save < 2 % of a 2-iteration loop but adds the flush control, so the tool drops it. [low confidence: 1 supporting design (mixed_66, overlap 1.2 %), 0 contradicting in 510. The nearest pipelined cases are flp TC 4 at 5.0 % and stp TC 2 at 5.9 %, so the threshold could lie anywhere between 1.2 % and 5 %, and the flp restriction is untested.] |
| S6 | pipeline on loop_i | one `gemm_Pipeline_loop_i` (TC 64, j and k fully unrolled), called once | Same rules as the j kinds with R = 64; B is invariant in the module and hoisted (H1). [extrapolated: no such design in the data] |

## P. Initiation interval

`II = max(II_target, II_rec, II_mem)`.

**Recurrence (P2).** Applies to k pipelines only; j pipelines have II_rec = 1.
- II_rec = u·La without reassociation.
- II_rec = La with unsafe-math reassociation, because the tree feeds one accumulator.

**Memory (P3).** II_mem = max over arrays of ceil(reads_per_bank / ports).
- reads_per_bank: distinct words of one bank read in one iteration, from the Vitis mapping:
  - cyclic: bank = x mod f, address = x div f;
  - block: bank = x div (64/f);
  - reshape: lanes of one word.
- Explicit storage fixes the ports: 1P → 1, 2P / T2P → 2.
- Default storage gives `p = min(reads, max(2, ceil(reads / II_base)), 16)`:
  - a BRAM has 2 ports;
  - more than 2 reads per bank replicates the bank (RAM_1WNR);
  - at most 16 read ports per replicated bank.

## D. Pipeline depth

**k pipeline, no reassociation:** `u·La + Lm + 2 + extra`. **Chain cost per add.** Here a chained add costs La, while the j pipeline and the non-pipelined body (Stage 2 L3) cost La + 1. In a k pipeline, the u adds of one iteration go through the same recurrence register: the result of add t feeds add t + 1 inside the fadd core, with no register stage between them. With k fully unrolled (j pipeline) or in a sequential body, each add is a separate scheduled operation, and its result is registered at a stage boundary, which adds the +1. When u = 1, extra is set by the accumulator's first-iteration select (acc = 0 vs acc):
[low confidence: the loop_k branches at T < 10 ns rest on 8 train designs and 0 extra (all loop_k branches: 23 train, 5 extra); the ijk branches are 24/26, 85/88 and 45/48 on extra, so not exact]
- ijk module: extra = 2. These branches in effect separate a default fmul latency from a bound one. At 2.5 ns, Lm ≥ 7 is the default fulldsp or maxdsp; at 10 ns, Lm < 3 comes only from BIND_OP latency = 1 or 2. When La ≥ Lm (the select is on the fadd input path), add a further amount:
  - T < 10 ns: +2, reduced to +1 at 2.5 ns when the fmul has ≥ 7 stages (it absorbs one register);
  - 10 ns: +1, only when Lm < 3.
- loop_k module: extra = 1, plus 1 if La > Lm.

**k pipeline, reassociated (u > 1):** `Lm + 2 + (ceil(log2 u) + 1)·(Lt + 1) + fetch`.
- The term (ceil(log2 u) + 1)·(Lt + 1) covers the tree levels plus the accumulator, each level an fadd plus a register.
- fetch = max over arrays of ceil(reads_per_bank / ports), the cycles to gather the operands.

**j pipeline (k fully unrolled):** `levels·(La + 1) + Lm + 3 (+1 at T < 10)`.
- levels = 64, a serial chain, which consumes operands in order and overlaps the operand fetch.
- With reassociation, levels = 6, a balanced tree. A tree needs all its leaves, so the fetch is added in front: `+ (fetch − 1)`.

**Copy loops (D-copy).** Base depth is 2 at 10/5 ns and 4 at 3.33/2.5 ns.
- copy_in: +1 when writing a lane into a reshaped word (merge stage). At 10 ns this applies only to the 2048-bit row word.
- copy_out at 5 ns: +1 for a lane select or a 64:1 bank mux.
- copy_out at 3.33/2.5 ns:
  - −1 when C is partitioned along j or completely (the bank read is retimed into the mux);
  - +1 for a lane select on dim 1;
  - at 2.5 ns with a word ≥ 512 bits: (base − 1) + SHIFT_LAT = 3 + 4. The fast-clock read path, then the 4-stage shifter (see X-SH). [low confidence: 6 train designs]

**Note on clock scaling.** The latency in ns of a loop limited by the fadd chain can grow at a faster clock: in 99 of 2229 single clock steps of a design sweep, 22 of them inside the dataset's range. This is not a model artifact. The measured fadd latency jumps from 5 cycles at 5 ns to 12 cycles at 3.33 ns, which is 25 ns against 40 ns per add.

## O. Operators (fp cores)

**O1, default impl when there is no `BIND_OP`.**
- fmul: maxdsp, or fulldsp at 2.5 ns.
- fadd on the accumulation recurrence: the impl with the smaller latency, with fabric on ties. This applies only if the recurrence binds, i.e. II_target < f·max(L_fabric, L_fulldsp), where f = 1 when reassociated and u otherwise. Otherwise the off-recurrence default applies.
- fadd off the recurrence: fulldsp at T ≥ 5 ns, fabric below.

**O2, default latency.** Comes from table `DEFAULT_LAT[(op, impl)][T]`. These are *measured* tool characteristics (scheduled core latency per clock, HS_dataset finding 2), not derived values:

| core | 10 ns | 5 ns | 3.33 ns | 2.5 ns |
|---|---|---|---|---|
| fadd fabric | 3 | 5 | 12 | 13 |
| fadd fulldsp | 4 | 9 | 12 | 12 |
| fmul fabric | 3 | 5 | 9 | 9 |
| fmul meddsp | 3 | 6 | 9 | 9 |
| fmul fulldsp | 3 | 6 | 7 | 9 |
| fmul maxdsp | 3 | 5 | 7 | 7 |

**Counts in pipelined modules.** P = u·R operations per iteration, shared over II:
- stp: ceil(P / II);
- flp: P (a flushable pipeline does not share).

With reassociation, the tree and the accumulator share one pool if they have the same impl. Otherwise the tree uses share(u − 1) and the accumulator uses 1 core of the recurrence impl.

**Owner.** [O2-par: low confidence, 2 train designs]
- The operators belong to `gemm` (one shared pool) when there are several compute modules, or no pipeline.
- **Exception O2-par:** the R `loop_k` copies are independent and each owns its cores when all of these hold:
  - A reaches them as hoisted scalars (H1 / H3);
  - B is read from disjoint bank sets per copy. (A "B hoisted" branch had no supporting design and was removed.)

  In that case they share no memory port, so the tool runs them in parallel.

**Deviations from the Target formulas (Target_refined §6, key mechanisms), with evidence.** The Target gives n_fmul = ceil(u/II) and, under unsafe_math, n_fadd = ceil((u − 1)/II) + 1. The model departs from these in three places, each backed by the data:
- **(a) Shared tree/accumulator pool.** Under reassociation, when the tree and the accumulator use the same impl, they share one pool: n_fadd = ceil(u/II). Evidence: mixed_35 (u = 8, II = 12) and mixed_62 (u = 32, II = 32) both have 1 fadd, where the Target formula gives 2. [low confidence: 2 supporting designs, both with u ≤ II; `predict.py` flags the case u > II]
- **(b) Pipelined loop_j with reassociation.** There is no separate accumulator, so the 64-term tree of each j copy needs 63 adds: n_fadd = ceil(63·R/II). Evidence: compile_config_01 has 63 fadds at II = 1, where the Target formula gives 64.
- **(c) flp style.** A flushable pipeline does not share cores: n = P = u·R, whatever the II. Evidence: 47 flp designs with II > 1, for example mixed_22 and mixed_73, each with 8 fmul at u = 8, II = 8.

**Non-pipelined body (B-none).** [low confidence: each non-default branch rests on 1–7 train designs; the fadd = 2 branch is 2/4 right; the default branches rest on 33 (fmul) and 44 (fadd) designs. This rule causes all 5 remaining DSP misses (mixed_12, 18, 21, 54, 82).]
- fmul count is 2 in any of these cases, otherwise 1. Two products are needed in one cycle:
  - u = 64 (the full row of products, back to back);
  - reassociation with u ≥ 2 (the tree consumes products in pairs);
  - A split along k with u ≥ 4 (two A operands per cycle from separate banks or lanes).
- fadd count is 2 when u = 64 and R ≥ 2 (two interleaved accumulation chains), otherwise 1.

## M. Memories

- Banks, words and bits follow from the partition or reshape: partition f gives f banks of 4096/f words; reshape f gives 32·f-bit words.
- Ports: from `BIND_STORAGE`. T2P forces BRAM. RAM_1WNR is inferred when a default-storage bank needs more than 2 reads per iteration (P3).

## A. Access pattern (per compute module)

**Hoisting rules.** A hoisted array is passed to the module as scalar inputs and has no memory port.

| rule | condition | effect |
|---|---|---|
| H1 | the array's indices are constant inside the module (A in a loop_j module) | The whole row is passed as words. No entry if it is reshaped or split along i. |
| H2 | complete along an index that varies in the module, not k (for reshape: only if k does not address the word) | all 64 values are passed as scalars |
| H3 | k pipeline: each bank is read at the same word set in every iteration | The words are loop invariant, so the caller reads them once. scalar_inputs = words per bank, bits = words × word width. |

**Ports.** banks_accessed is every bank reachable over the module's whole iteration space, including the call-fixed i and j0. A bank selected by any run-time index must be wired to the module. read_ports_per_bank comes from P3.

**Writes to C.**
- A whole-row reshape of C in a loop_j module is held by the caller: 1 scalar.
- Otherwise:
  - write ports = min(stores per bank per iteration, 2);
  - the stores fall back to 1 port when any of these holds:
    - C is reshaped (one byte-enabled wide port);
    - the pipeline is flp (stores leave from the single flush-controlled stage);
    - an input is ram_1p (one operand per cycle). [low confidence: `VALIDATION_STAGE1.md` §5: min(stores, 2) 8/8 train + 1/1 extra; serialised to 1 16/18 train, 0 extra]
- A read port of 1 is added for read-modify-write of a reshaped C word when any of these holds:
  - several lanes of one word are written in one iteration;
  - the module is a flattened ijk module;
  - the module is a flattened i/j module with a whole-row word.

Non-pipelined designs report `{'gemm': {}}`.

## X. Aux cores

**sparsemux.** One bank-select mux per group of read slots that always hit the same bank sequence. Count = min(#slots, ports); hoisted values are not port-limited. Inputs = banks the slot can reach over all index values, as in A. A 2-way select is a plain select operator, so a sparsemux needs ≥ 3 inputs. Further cases:
- Non-pipelined bodies: one mux per j copy if the partitioned dimension is indexed by j, otherwise 1.
- copy_out: one mux over the C banks (≥ 3).

**shift (X-SH).** [clock condition well supported: no shifter on 14 wide words at 3.33 ns; the 512-bit threshold rests on one 256-bit counter-example; SHIFT_LAT = 4 is read off the data] Applies only at 2.5 ns. A lane select out of a reshaped word of ≥ 512 bits (≥ 16 lanes) does not fit one cycle as a mux, so the tool slices the word with a pipelined barrel shifter:
- core: lshr, width = word bits, shamt = log2(width), latency 4;
- one shifter per read slot whose lane changes at run time;
- for C, one shifter in copy_out;
- in non-pipelined bodies, one per j copy.

**shl (X-SHL).** Applies only at 2.5 ns, when one 32-bit lane is written into a reshaped word of ≥ 512 bits. The write uses two left shifters: a data shifter (32 → W bits) places the lane, and a mask shifter (W → W) places the byte enable. In the report these are pipelined Expression entries of 2661 FF / 3250 LUT each, the same cost as the lshr core, and each has one W-bit output register.

The rule counts the pipelined shifters per writer:
- copy_in, per reshaped A or B: 2 for a whole-row word (complete reshape on dim 2, data and mask both pipelined), otherwise 1 (the mask shifter only);
- the C writer, per lane write of one iteration: 2 for a whole-row word, otherwise 1.

The C writer is the compute module that stores C, or `gemm` when the top FSM stores C. The number of lane writes is R, or 1 in the flattened ijk nest. Because the shifters are merged per owner, the HS also carries `total_bits` (the sum of the output widths), which Stage 2 uses for the output registers.

[support: 11 designs, 8 exact (14 modules, 12 exact); all train, all at 2.5 ns. The 2 module misses, mixed_05 compute and mixed_64 gemm, are cases where a 512-bit data shifter is pipelined too; the rule does not predict that.]

## T. Transforms and latency breakdown

Each transform below is derived from the pragmas and the structure.

| field | contents |
|---|---|
| unroll | explicit unrolls, plus `implied_by_pipeline` for loop_k under pipeline j |
| array | one entry per partition or reshape directive |
| flatten | copy_in (if flattened), copy_out, loop_i and loop_j as in the structure |
| balanced_expressions | adds in the rebalanced chain: 63·R for a j pipeline, u·R for k or none when u > 2, otherwise 0 |

Stages and top-loop bodies follow directly from S.

## Constants and their hardware meaning

| constant | value | meaning |
|---|---|---|
| N | 64 | problem size (loop trip counts) |
| DEFAULT_LAT | table | tool's default scheduled latency of each fp core per clock |
| BRAM ports | 2 | dual-port block RAM |
| port cap | 16 | max read ports of a replicated (1WNR) bank |
| sparsemux min inputs | 3 | 2:1 selects are plain `select` operators |
| shift width threshold | 512 bit | 16 × 32-bit lanes: a lane mux this wide misses 2.5 ns in one cycle |
| lshr latency | 4 | pipelined barrel shifter stages at 2.5 ns |
| copy depth base | 2 / 4 | read, (address/data registers at fast clocks), write |
| depth offsets (+2, +3, +1…) | see D | memory read register, output register, init select, write |

## Known exceptions and biases on train (not modeled)

- **Compute-depth errors.** Compute depth is exact on only 60.3 % of train modules (94.2 % of extra modules). The signed error (Stage 1 − tool) per module kind is in `VALIDATION_STAGE1.md` §4b:
  - loop_k, chain: −2 on 22 modules and −1 on 4. These are R > 1 modules whose reads pass a wide bank mux or a lane select; the stage is not modeled.
  - loop_k, reassociated: mostly over-predicted, +1 to +4 on 25 modules.
  - loop_j: −1 to −3 on 21 modules, plus the large-II outliers (VALIDATION_STAGE1.md / FINAL_VALIDATION.md give the current signed-error table).
  - extra still has 17 misses: the ijk modules ±1, and loop_j −1.

  Latency still passes on 98.1 % of train, because II · trip count dominates.
- **Predicting a design the tool would refuse.** The only guard is S5, which rests on one design.

- **mixed_66** is now covered by rule S5.
- **mixed_79:** u_j = 2 with an A block reshape along k; the tool flattened i/j anyway.
- **mixed_10:** the caller-side shifters for an H1-hoisted reshaped A.
- **mixed_12 / mixed_18:** a reassociated non-pipelined body with many fp cores.
- **mixed_21 / mixed_54:** the B-none fmul count.
- **Large-II loop_j pipelines** (mixed_14, 23, 44, 49, 57, 93, …): depth exceeds the chain formula once memory starvation dominates (depth ≈ fetch time + chain). There are 10 of them; their latency still passes because II dominates.

The whole-HS check compares the full sanitized JSON with every list sorted. List order carries no meaning, and Stage 2 gives identical outputs regardless of it.

Validation is in `VALIDATION_STAGE1.md`, generated by `stage1_report.py`.
