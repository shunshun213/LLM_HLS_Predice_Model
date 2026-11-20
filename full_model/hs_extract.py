#!/usr/bin/env python3
"""
HS extractor: reads the TRUE hardware structure (HS) that Vitis HLS 2023.2 built for each
GEMM sample, from its synthesis outputs, and writes the Stage 1 and Stage 2 datasets.

Sources per sample (samples/<name>/):
  run.tcl, result.json              -> pragma configuration + clock (Stage 1 inputs)
  report/<module>_csynth.rpt        -> per-module latency, loop table, utilization by category,
                                       Instance / Memory / Expression / Multiplexer / Register tables,
                                       RTL port table (memory-port usage = access pattern)
  report/<module>_csynth.xml        -> loop PipelineDepth / PipelineII cross-check
  hls.log                           -> unroll / partition / flatten decisions, storage inference,
                                       pipelining result (target II, final II, depth) and the
                                       II-violation causes (recurrence vs memory port)

Every field is read from the tool output; nothing is fitted.

Usage:
  python3 hs_extract.py <dataset_root> <out_dir>
"""
import csv
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

DEVICE = {"LUT": 70560, "FF": 141120, "DSP": 360, "BRAM_18K": 432}
RES = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
CATS = ["DSP", "Expression", "FIFO", "Instance", "Memory", "Multiplexer", "Register"]

# GEMM loop nest in gemm_base.cpp: loop -> its enclosing loops (outermost first)
LOOP_ANCESTORS = {
    "copy_in": [], "copy_in_j": ["copy_in"],
    "loop_i": [], "loop_j": ["loop_i"], "loop_k": ["loop_i", "loop_j"],
    "copy_out": [], "copy_out_j": ["copy_out"],
}
LOOP_ORDER = ["copy_in", "copy_in_j", "loop_i", "loop_j", "loop_k", "copy_out", "copy_out_j"]

IMPL_MAP = {"no_dsp": "fabric", "full_dsp": "fulldsp", "max_dsp": "maxdsp",
            "med_dsp": "meddsp", "prim_dsp": "primitivedsp"}


# ----------------------------------------------------------------------------- helpers
def num(s):
    """'1,234' -> 1234 ; '3.5' -> 3.5 ; '-', '?', '' -> None"""
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if s in ("", "-", "?", "N/A", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def cells(line):
    parts = line.rstrip("\n").split("|")
    return parts[1:-1] if len(parts) >= 3 else []


def read_table(lines, i):
    """Table starting at a '+---' border. Returns (segments, next_index); a segment is a list
    of raw cell lists between two borders."""
    segs, cur = [], []
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("+-"):
            if cur:
                segs.append(cur)
                cur = []
        elif s.startswith("|"):
            cur.append(cells(lines[i]))
        else:
            break
        i += 1
    if cur:
        segs.append(cur)
    return segs, i


def dict_rows(segs):
    """Single-row header tables -> list of dicts (Total rows dropped)."""
    if len(segs) < 2:
        return []
    hdr = [h.strip() for h in segs[0][-1]]
    out = []
    for seg in segs[1:]:
        for r in seg:
            if not r or r[0].strip() in ("Total", ""):
                continue
            out.append({h: r[k].strip() if k < len(r) else "" for k, h in enumerate(hdr)})
    return out


# ----------------------------------------------------------------------------- report parser
def parse_rpt(path):
    lines = open(path, errors="replace").read().splitlines()
    rep = {"module": None, "timing": {}, "latency": {}, "perf_instances": [], "loops": [],
           "util": {}, "util_total": {}, "instances": [], "dsp": [], "memory": [],
           "expression": [], "multiplexer": [], "register": [], "rtl_ports": []}
    m = re.search(r"Vitis HLS Report for '([^']+)'", "\n".join(lines[:6]))
    rep["module"] = m.group(1) if m else os.path.basename(path).replace("_csynth.rpt", "")
    major, sub = None, None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("== "):
            major = s.strip("= ").strip()
            sub = None
        elif re.match(r"^[\*\+] [A-Za-z][A-Za-z /]*:\s*$", s):
            sub = s[2:].rstrip(":").strip()
        elif s.startswith("+-") and sub:
            segs, i = read_table(lines, i)
            _store(rep, major, sub, segs)
            sub = None
            continue
        i += 1
    return rep


def _store(rep, major, sub, segs):
    if major == "Performance Estimates":
        if sub == "Summary" and segs and segs[0] and "Clock" in "".join(segs[0][0]):
            r = segs[1][0]
            rep["timing"] = {"target_ns": num(r[1].replace("ns", "")),
                             "est_ns": num(r[2].replace("ns", "")),
                             "uncertainty_ns": num(r[3].replace("ns", ""))}
        elif sub == "Summary":
            r = segs[1][0]
            rep["latency"] = {"min": num(r[0]), "max": num(r[1]),
                              "interval_min": num(r[4]), "interval_max": num(r[5]),
                              "pipeline_type": r[6].strip()}
        elif sub == "Instance":
            for r in segs[1] if len(segs) > 1 else []:
                rep["perf_instances"].append({
                    "instance": r[0].strip(), "module": r[1].strip(),
                    "lat_min": num(r[2]), "lat_max": num(r[3]),
                    "ii_min": num(r[6]), "ii_max": num(r[7]), "pipeline_type": r[8].strip()})
        elif sub == "Loop":
            for r in segs[1] if len(segs) > 1 else []:
                raw = r[0]
                body = raw.lstrip(" ")
                level = len(raw) - len(body)
                name = re.sub(r"^[-+o*]+\s*", "", body).strip()
                piped = r[7].strip()
                style = None
                mm = re.match(r"yes\((\w+)\)", piped)
                if mm:
                    style = mm.group(1)
                rep["loops"].append({
                    "name": name, "level": level,
                    "lat_min": num(r[1]), "lat_max": num(r[2]),
                    "iteration_latency": num(r[3]), "ii_achieved": num(r[4]),
                    "ii_target": num(r[5]), "trip_count": num(r[6]),
                    "pipelined": piped.startswith("yes"), "pipeline_style": style})
    elif major == "Utilization Estimates":
        if sub == "Summary":
            hdr = [h.strip() for h in segs[0][-1]]
            for seg in segs[1:]:
                for r in seg:
                    nm = r[0].strip()
                    vals = {hdr[k]: num(r[k]) for k in range(1, len(hdr))}
                    if nm in CATS:
                        rep["util"][nm] = {k: (vals.get(k) or 0) for k in RES}
                    elif nm == "Total":
                        rep["util_total"] = {k: (vals.get(k) or 0) for k in RES}
        else:
            key = {"Instance": "instances", "DSP": "dsp", "Memory": "memory",
                   "Expression": "expression", "Multiplexer": "multiplexer",
                   "Register": "register"}.get(sub)
            if key:
                rep[key].extend(dict_rows(segs))
    elif major and major.startswith("Interface"):
        if sub == "Summary":
            rep["rtl_ports"].extend(dict_rows(segs))


def parse_xml_loops(path):
    """loop name -> {PipelineII, PipelineDepth, TripCount, Latency, IterationLatency}"""
    out = {}
    if not os.path.exists(path):
        return out
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return out
    sol = root.find("./PerformanceEstimates/SummaryOfLoopLatency")
    if sol is None:
        return out

    def walk(e):
        for c in e:
            if c.find("TripCount") is not None or c.find("Latency") is not None:
                out[c.tag] = {k: num(c.findtext(k)) for k in
                              ("TripCount", "Latency", "IterationLatency", "PipelineII", "PipelineDepth")}
                walk(c)
    walk(sol)
    return out


# ----------------------------------------------------------------------------- log parser
RX = {
    "module": re.compile(r"\[HLS 200-42\] -- Implementing module '([^']+)'"),
    "pipe": re.compile(r"\[HLS 200-1470\] Pipelining result : Target II = (\S+), Final II = (\S+), Depth = (\d+), loop '([^']+)'"),
    "carried": re.compile(r"\[HLS 200-880\] The II Violation in module '([^']+)' \(loop '([^']+)'\): Unable to enforce a carried dependence constraint \(II = (\d+), distance = (\d+).*?on (?:local )?variable '([^']+)'"),
    "memport": re.compile(r"\[HLS 200-885\] The II Violation in module '([^']+)' \(loop '([^']+)'\): Unable to schedule '(\w+)' operation.*?on array '([^']+)' due to limited memory ports \(II = (\d+)\)"),
    "timing886": re.compile(r"\[HLS 200-886\] Cannot meet target clock period in '(\w+)' operation.*?\(II=(\d+)\) in region 'loop '([^']+)''"),
    "unroll_p": re.compile(r"\[HLS 214-188\] Unrolling loop '([^']+)'.*?partially with a factor of (\d+)"),
    "unroll_c": re.compile(r"\[HLS 214-186\] Unrolling loop '([^']+)'.*?completely with a factor of (\d+)"),
    "unroll_i": re.compile(r"\[HLS 214-291\] Loop '([^']+)' is marked as complete unroll implied by the pipeline pragma"),
    "array": re.compile(r"\[HLS 214-248\] Applying (array_partition|array_reshape) to '([^']+)': (.*?)\. \("),
    "flat": re.compile(r"\[XFORM 203-541\] Flattening a loop nest '([^']+)'"),
    "noflat": re.compile(r"\[HLS 200-960\] Cannot flatten loop '([^']+)'.*?because (.*)$"),
    "storage": re.compile(r"\[HLS 200-1457\] Automatically inferring inferring (\S+) storage type for array '([^']+)'"),
    "estclk": re.compile(r"\[HLS 200-871\] Estimated clock period \(([\d.]+) ns\)"),
    "opdelay": re.compile(r"\[HLS 200-1015\] Estimated delay Estimated delay \(([\d.]+)ns\)\s+of '(\w+)' operation"),
    "balance": re.compile(r"\[XFORM 203-11\] Balancing expressions in function '([^']+)'.*?\.\.\.(\d+) expression\(s\) balanced"),
}


def parse_log(path):
    out = {"pipelining": [], "unroll": [], "array_xform": [], "flatten": [],
           "storage_inferred": {}, "op_delay_ns": {}, "balanced_expressions": 0,
           "violations": defaultdict(lambda: {"recurrence": [], "memory_port": [], "timing": []})}
    if not os.path.exists(path):
        return out
    cur = None
    seen_unroll = set()
    for line in open(path, errors="replace"):
        m = RX["module"].search(line)
        if m:
            cur = m.group(1)
            continue
        m = RX["pipe"].search(line)
        if m:
            out["pipelining"].append({"module": cur, "loop": m.group(4), "ii_target": num(m.group(1)),
                                      "ii_final": num(m.group(2)), "depth": int(m.group(3))})
            continue
        m = RX["carried"].search(line)
        if m:
            out["violations"][(m.group(1), m.group(2))]["recurrence"].append(
                {"ii": int(m.group(3)), "distance": int(m.group(4)), "variable": m.group(5)})
            continue
        m = RX["memport"].search(line)
        if m:
            out["violations"][(m.group(1), m.group(2))]["memory_port"].append(
                {"ii": int(m.group(5)), "op": m.group(3), "array": m.group(4)})
            continue
        m = RX["timing886"].search(line)
        if m:
            out["violations"][(cur, m.group(3))]["timing"].append({"ii": int(m.group(2)), "op": m.group(1)})
            continue
        for k, kind in (("unroll_p", "partial"), ("unroll_c", "complete")):
            m = RX[k].search(line)
            if m and (m.group(1), kind) not in seen_unroll:
                seen_unroll.add((m.group(1), kind))
                out["unroll"].append({"loop": m.group(1), "kind": kind, "factor": int(m.group(2))})
        m = RX["unroll_i"].search(line)
        if m and (m.group(1), "implied") not in seen_unroll:
            seen_unroll.add((m.group(1), "implied"))
            out["unroll"].append({"loop": m.group(1), "kind": "implied_by_pipeline", "factor": None})
        m = RX["array"].search(line)
        if m:
            out["array_xform"].append({"directive": m.group(1), "array": m.group(2), "detail": m.group(3)})
        m = RX["flat"].search(line)
        if m:
            out["flatten"].append({"loop": m.group(1), "flattened": True, "reason": None})
        m = RX["noflat"].search(line)
        if m:
            out["flatten"].append({"loop": m.group(1), "flattened": False, "reason": m.group(2).strip()})
        m = RX["storage"].search(line)
        if m:
            root = re.match(r"(l[ABC])", m.group(2))
            if root:
                out["storage_inferred"].setdefault(root.group(1), set()).add(m.group(1))
        m = RX["opdelay"].search(line)
        if m:
            out["op_delay_ns"][m.group(2)] = max(out["op_delay_ns"].get(m.group(2), 0), float(m.group(1)))
        m = RX["balance"].search(line)
        if m:
            out["balanced_expressions"] += int(m.group(2))
    out["storage_inferred"] = {k: sorted(v) for k, v in out["storage_inferred"].items()}
    # de-duplicate flatten records
    fl, seen = [], set()
    for f in out["flatten"]:
        key = (f["loop"], f["flattened"])
        if key not in seen:
            seen.add(key)
            fl.append(f)
    out["flatten"] = fl
    return out


# ----------------------------------------------------------------------------- HS assembly
def op_kind(module):
    """Classify an Instance-table module name."""
    m = re.match(r"^(fadd|fmul|fsub|fdiv|dadd|dmul)_(\d+)ns_(\d+)ns_(\d+)_(\d+)_(\w+?)_1$", module)
    if m:
        # The number in the core name is (scheduled latency + 1): a core built with
        # `bind_op -latency L` is named ..._<L+1>_... (verified on all 94 S1 samples that request
        # a latency; the wrapper adds an input register in front of the IP's c_latency).
        return {"class": "fp", "op": m.group(1), "latency": int(m.group(5)) - 1,
                "name_stages": int(m.group(5)),
                "impl_core": m.group(6), "impl": IMPL_MAP.get(m.group(6), m.group(6)),
                "width": int(m.group(4))}
    m = re.match(r"^sparsemux_(\d+)_(\d+)_(\d+)_\d+_\d+$", module)
    if m:  # sparsemux_<n_ports>_<sel_bits>_<data_bits>; n_ports = 2*inputs+1
        return {"class": "sparsemux", "op": "sparsemux", "inputs": (int(m.group(1)) - 1) // 2,
                "sel_bits": int(m.group(2)), "width": int(m.group(3))}
    m = re.match(r"^(lshr|shl|ashr)_(\d+)ns_(\d+)ns_(\d+)_(\d+)_\d+$", module)
    if m:
        return {"class": "shift", "op": m.group(1), "width": int(m.group(2)),
                "shamt_bits": int(m.group(3)), "latency": int(m.group(5))}
    m = re.match(r"^(mul|mac_muladd|am_addmul|ama_addmuladd|urem|srem|udiv|sdiv)_", module)
    if m:
        return {"class": "int", "op": m.group(1)}
    return None


def loop_root(name):
    """'loop_k1' -> 'loop_k' ; 'loop_i_loop_j_loop_k' -> outermost 'loop_i'."""
    toks = re.findall(r"(copy_in_j|copy_out_j|copy_in|copy_out|loop_[ijk])", name)
    return toks


def storage_from_core(core_module):
    # e.g. lA_RAM_1WNR_AUTO_1R1W, lB_RAM_T2P_BRAM_1R1W, lC_RAM_AUTO_1R1W
    m = re.search(r"_(RAM|ROM)_(?:(1P|2P|T2P|S2P|1WNR|NP)_)?(AUTO|BRAM|LUTRAM|URAM|DISTRAM)_", core_module)
    if not m:
        return {"kind": None, "ports": None, "impl": None}
    return {"kind": m.group(1), "ports": m.group(2) or "default", "impl": m.group(3)}


def build_sample(sdir):
    name = os.path.basename(sdir)
    res = json.load(open(os.path.join(sdir, "result.json")))
    tcl = open(os.path.join(sdir, "run.tcl")).read()
    rdir = os.path.join(sdir, "report")
    reps = {}
    for p in sorted(glob.glob(os.path.join(rdir, "*_csynth.rpt"))):
        r = parse_rpt(p)
        r["xml_loops"] = parse_xml_loops(p.replace(".rpt", ".xml"))
        reps[r["module"]] = r
    top = reps["gemm"]
    log = parse_log(os.path.join(sdir, "hls.log"))

    # ---- module hierarchy and multiplicity (how many times each module is instantiated)
    children = {m: [i["Module"] for i in r["instances"] if i.get("Module") in reps] for m, r in reps.items()}
    mult = Counter()

    def walk(m, k):
        mult[m] += k
        for c in children[m]:
            walk(c, k)
    walk("gemm", 1)

    # ---- per-module records
    modules = []
    for mname, r in reps.items():
        regs = {x["Name"]: num(x.get("Bits")) for x in r["register"]}
        fsm_bits = sum(v or 0 for k, v in regs.items() if re.match(r"ap_CS(_iter\d+)?_fsm$", k))
        ports = defaultdict(lambda: {"banks": set(), "read_ports": Counter(), "write_ports": Counter(),
                                     "scalar_inputs": 0, "scalar_bits": 0})
        for p in r["rtl_ports"]:
            obj = p.get("Source Object", "")
            pname = p.get("RTL Ports", "")
            proto = p.get("Protocol", "")
            mroot = re.match(r"^(l[ABC]|[ABC])(?:_\d+)*$", obj) or re.match(r"^(l[ABC])_load", obj)
            if not mroot:
                continue
            arr = mroot.group(1)
            if proto == "ap_memory":
                ports[arr]["banks"].add(obj)
                pm = re.search(r"_(q|d)(\d+)$", pname)
                if pm and pm.group(1) == "q":
                    ports[arr]["read_ports"][obj] += 1
                elif pm and pm.group(1) == "d":
                    ports[arr]["write_ports"][obj] += 1
            elif proto in ("ap_none", "ap_vld") and p.get("Dir") == "in":
                ports[arr]["scalar_inputs"] += 1
                ports[arr]["scalar_bits"] += num(p.get("Bits")) or 0
        port_summary = {}
        for arr, d in ports.items():
            nb = len(d["banks"])
            rp = [d["read_ports"][b] for b in d["banks"]]
            wp = [d["write_ports"][b] for b in d["banks"]]
            port_summary[arr] = {
                "banks_accessed": nb,
                "read_ports_per_bank_max": max(rp) if rp else 0,
                "read_ports_total": sum(rp),
                "write_ports_per_bank_max": max(wp) if wp else 0,
                "write_ports_total": sum(wp),
                "scalar_inputs": d["scalar_inputs"], "scalar_input_bits": d["scalar_bits"]}
        mux = r["multiplexer"]
        modules.append({
            "module": mname, "is_top": mname == "gemm", "instantiations": mult[mname],
            "children": children[mname],
            "latency_min": r["latency"].get("min"), "latency_max": r["latency"].get("max"),
            "interval_max": r["latency"].get("interval_max"),
            "est_clock_ns": r["timing"].get("est_ns"),
            "fsm_state_bits": fsm_bits,
            "n_registers": len(r["register"]),
            "register_bits": sum(num(x.get("Bits")) or 0 for x in r["register"]),
            "n_mux": len(mux),
            "mux_total_input_bits": sum(num(x.get("Total Bits")) or 0 for x in mux),
            "n_expressions": len(r["expression"]),
            "expression_ops": dict(Counter(x.get("Operation", "").strip() for x in r["expression"])),
            "memory_ports": port_summary,
            "util": r["util"], "util_total": r["util_total"]})

    # ---- loops (with log-derived II limiter and xml depth)
    loops = []
    pipe_log = {(p["module"], p["loop"]): p for p in log["pipelining"]}
    for mname, r in reps.items():
        for lp in r["loops"]:
            x = r["xml_loops"].get(lp["name"], {})
            pl = pipe_log.get((mname, lp["name"]), {})
            v = log["violations"].get((mname, lp["name"]), {"recurrence": [], "memory_port": [], "timing": []})
            last = {k: max((e["ii"] for e in v[k]), default=None) for k in v}
            limiter = "none"
            iia, iit = lp["ii_achieved"], lp["ii_target"]
            if lp["pipelined"] and iia and iit and iia > iit:
                cands = [k for k in ("recurrence", "memory_port", "timing") if last[k] == iia - 1]
                if not cands:
                    cands = [k for k in ("recurrence", "memory_port", "timing") if last[k] is not None]
                limiter = "+".join(cands) if cands else "unknown"
            toks = loop_root(lp["name"])
            loops.append({
                "module": mname, "module_instantiations": mult[mname], "loop": lp["name"],
                "flattened_from": toks if len(toks) > 1 else None,
                "level": lp["level"], "trip_count": lp["trip_count"],
                "latency": lp["lat_max"], "iteration_latency": lp["iteration_latency"],
                "pipelined": lp["pipelined"], "pipeline_style": lp["pipeline_style"],
                "ii_target": iit, "ii_achieved": iia,
                "pipeline_depth": x.get("PipelineDepth") or (lp["iteration_latency"] if lp["pipelined"] else None),
                "log_depth": pl.get("depth"),
                "ii_limiter": limiter,
                "last_failed_ii": {k: last[k] for k in last},
                "recurrence_vars": sorted({e["variable"] for e in v["recurrence"]}),
                "memport_arrays": sorted({e["array"] for e in v["memory_port"]})})

    # ---- operators (fp IP + aux instances), with owner module and multiplicity
    ops = Counter()
    unit = {}
    aux = Counter()
    aux_unit = {}
    for mname, r in reps.items():
        for inst in r["instances"]:
            k = op_kind(inst.get("Module", ""))
            if not k:
                continue
            cost = {c: num(inst.get(c)) or 0 for c in ("DSP", "FF", "LUT", "BRAM_18K")}
            if k["class"] == "fp":
                key = (mname, k["op"], k["impl"], k["latency"], inst["Module"])
                ops[key] += 1
                unit[key] = cost
            else:
                key = (mname, k["class"], inst["Module"])
                aux[key] += 1
                aux_unit[key] = (cost, k)
    operators = []
    for (mname, op, impl, lat, core), n in sorted(ops.items()):
        operators.append({"owner": mname, "op": op, "impl": impl, "latency": lat, "core": core,
                          "count_in_module": n, "owner_instantiations": mult[mname],
                          "count_effective": n * mult[mname], "unit_cost": unit[(mname, op, impl, lat, core)]})
    aux_ops = []
    for (mname, cls, core), n in sorted(aux.items()):
        cost, k = aux_unit[(mname, cls, core)]
        aux_ops.append({"owner": mname, "class": cls, "core": core, "params": k,
                        "count_in_module": n, "count_effective": n * mult[mname], "unit_cost": cost})
    op_totals = defaultdict(lambda: {"count": 0, "impl": set(), "latency": set()})
    for o in operators:
        t = op_totals[o["op"]]
        t["count"] += o["count_effective"]
        t["impl"].add(o["impl"])
        t["latency"].add(o["latency"])
    op_totals = {k: {"count": v["count"], "impl": sorted(v["impl"]), "latency": sorted(v["latency"])}
                 for k, v in op_totals.items()}
    # where the fp operators live: shared at top (muxed grp_fu) or inside loop modules
    fp_location = sorted({o["owner"] for o in operators})

    # ---- memories per local array
    memories = {}
    for mname, r in reps.items():
        for row in r["memory"]:
            mm = re.match(r"^(l[ABC])", row.get("Memory", ""))
            if not mm:
                continue
            a = mm.group(1)
            st = storage_from_core(row.get("Module", ""))
            d = memories.setdefault(a, {"owner": mname, "core": row.get("Module"), "storage_ports": st["ports"],
                                        "storage_impl": st["impl"], "rows": []})
            d["rows"].append({c: num(row.get(c)) or 0 for c in ("BRAM_18K", "FF", "LUT", "URAM", "Words", "Bits", "Banks")})
    for a, d in memories.items():
        rows = d.pop("rows")
        shapes = Counter((x["Words"], x["Bits"], x["BRAM_18K"], x["LUT"], x["FF"]) for x in rows)
        d.update({
            "banks": sum(x["Banks"] for x in rows),
            "words_per_bank": sorted({x["Words"] for x in rows}),
            "bits_per_word": sorted({x["Bits"] for x in rows}),
            "bram18k_per_bank": sorted({x["BRAM_18K"] for x in rows}),
            "uniform_banks": len(shapes) == 1,
            "bank_shapes": [{"words": w, "bits": b, "bram18k": br, "lut": l, "ff": f, "n": n}
                            for (w, b, br, l, f), n in shapes.items()],
            "BRAM_18K": sum(x["BRAM_18K"] for x in rows), "LUT": sum(x["LUT"] for x in rows),
            "FF": sum(x["FF"] for x in rows), "URAM": sum(x["URAM"] for x in rows),
            "total_bits": sum(x["Words"] * x["Bits"] * x["Banks"] for x in rows),
            "in_registers": False})
    for a in ("lA", "lB", "lC"):
        if a not in memories:
            regbits = sum(num(x.get("Bits")) or 0 for r in reps.values() for x in r["register"]
                          if re.match(rf"^{a}_", x.get("Name", "")))
            memories[a] = {"owner": None, "core": None, "storage_ports": None, "storage_impl": None,
                           "banks": 0, "in_registers": True, "register_bits": regbits,
                           "BRAM_18K": 0, "LUT": 0, "FF": 0, "URAM": 0}
        memories[a]["storage_inferred_by_tool"] = log["storage_inferred"].get(a)

    # ---- access pattern: compute-module port usage per array (where the MAC loop lives)
    compute_modules = [m["module"] for m in modules
                       if re.search(r"loop_[ijk]", m["module"]) and not m["is_top"]]
    access = {}
    for m in modules:
        if m["module"] in compute_modules or (m["is_top"] and not compute_modules):
            access[m["module"]] = {a: v for a, v in m["memory_ports"].items() if a.startswith("l")}

    # ---- latency breakdown at top: stages called directly vs inside top-level loops
    top_loops = top["loops"]

    def enclosing_top_loop(first_token, own_level=None):
        """Deepest top-level loop (possibly a flattened nest such as loop_i_loop_j) that encloses
        a loop whose outermost source loop is `first_token`."""
        anc = set(LOOP_ANCESTORS.get(first_token, []))
        best = None
        for L in top_loops:
            if own_level is not None and L["level"] != own_level - 1:
                continue
            if anc & set(loop_root(L["name"])):
                if best is None or L["level"] > best["level"]:
                    best = L
        return best["name"] if best else None
    calls = []
    for pi in top["perf_instances"]:
        mod = pi["module"]
        toks = loop_root(mod.replace("gemm_Pipeline_", ""))
        calls.append({"instance": pi["instance"], "module": mod, "latency": pi["lat_max"],
                      "inside_top_loop": enclosing_top_loop(toks[0]) if toks else None})
    stages = []
    for c in calls:
        if c["inside_top_loop"] is None:
            stages.append({"stage": c["module"], "kind": "call", "latency": c["latency"]})
    for l in top_loops:
        if l["level"] == 0:
            stages.append({"stage": l["name"], "kind": "top_loop", "latency": l["lat_max"],
                           "trip_count": l["trip_count"], "iteration_latency": l["iteration_latency"]})
    order = {n: i for i, n in enumerate(LOOP_ORDER)}

    def stage_key(s):
        t = loop_root(s["stage"].replace("gemm_Pipeline_", ""))
        return order.get(t[0], 99) if t else 99
    stages.sort(key=stage_key)
    top_lat = top["latency"]["max"]
    known = sum(s["latency"] or 0 for s in stages)
    # per-iteration body of top loops: inner top loops + calls placed in them
    loop_bodies = []
    for l in top_loops:
        inner = [x for x in top_loops if x["level"] == l["level"] + 1
                 and enclosing_top_loop(loop_root(x["name"])[0], x["level"]) == l["name"]]
        called = [c for c in calls if c["inside_top_loop"] == l["name"]]
        body = sum(x["lat_max"] or 0 for x in inner) + sum(c["latency"] or 0 for c in called)
        loop_bodies.append({"loop": l["name"], "trip_count": l["trip_count"],
                            "iteration_latency": l["iteration_latency"],
                            "body_children_latency": body,
                            "children": [x["name"] for x in inner] + [c["module"] for c in called],
                            "per_iteration_overhead": (l["iteration_latency"] or 0) - body})
    by_phase = {"copy_in": 0, "compute": 0, "copy_out": 0}
    for st in stages:
        nm = st["stage"].replace("gemm_Pipeline_", "")
        ph = "copy_in" if nm.startswith("copy_in") else "copy_out" if nm.startswith("copy_out") else "compute"
        st["phase"] = ph
        by_phase[ph] += st["latency"] or 0
    latency_breakdown = {"top_latency": top_lat, "stages": stages, "by_phase": by_phase,
                         "fsm_overhead": (top_lat - known) if top_lat is not None else None,
                         "top_loop_bodies": loop_bodies}

    # ---- assemble
    cfg = res["config"]
    met = res["metrics"]
    hs = {
        "module_hierarchy": [{"module": m["module"], "instantiations": m["instantiations"],
                              "children": m["children"], "latency": m["latency_max"],
                              "fsm_state_bits": m["fsm_state_bits"]} for m in modules],
        "loops": loops,
        "operators": operators,
        "operator_totals": op_totals,
        "fp_operator_owner_modules": fp_location,
        "aux_operators": aux_ops,
        "memories": memories,
        "access_pattern": access,
        "transforms_applied": {"unroll": log["unroll"], "array": log["array_xform"],
                               "flatten": log["flatten"], "balanced_expressions": log["balanced_expressions"]},
        "timing": {"target_ns": top["timing"].get("target_ns"), "est_ns": top["timing"].get("est_ns"),
                   "uncertainty_ns": top["timing"].get("uncertainty_ns"),
                   "op_delay_ns_unpipelined": log["op_delay_ns"]},
        "latency_breakdown": latency_breakdown,
    }
    targets = {
        "latency_cycles": top_lat,
        "interval": top["latency"].get("interval_max"),
        "LUT": top["util_total"].get("LUT"), "FF": top["util_total"].get("FF"),
        "DSP": top["util_total"].get("DSP"), "BRAM_18K": top["util_total"].get("BRAM_18K"),
        "URAM": top["util_total"].get("URAM"),
        "fits": all((top["util_total"].get(k) or 0) <= v for k, v in DEVICE.items()),
        "top_by_category": top["util"],
        "per_module": {m["module"]: {"instantiations": m["instantiations"], "latency": m["latency_max"],
                                     "total": m["util_total"], "by_category": m["util"]} for m in modules},
        "est_clock_ns": top["timing"].get("est_ns"),
    }
    inputs = {
        "config": cfg,
        "clock_ns": cfg.get("clock_ns"),
        "directives": res.get("directives"),
        "tcl": tcl,
        "source_with_pragmas": open(os.path.join(sdir, "gemm_pragmas.cpp")).read()
        if os.path.exists(os.path.join(sdir, "gemm_pragmas.cpp")) else None,
    }
    suite = res.get("sweep") or re.sub(r"_\d+_[0-9a-f]+$", "", name)
    source = res.get("source") or ("extra" if res.get("sweep") else "train")
    return {"sample": name, "suite": suite, "source": source, "key": res.get("key"),
            "inputs": inputs, "hs": hs, "targets": targets,
            "_metrics_result_json": met, "_reps": reps, "_log": log}


# ----------------------------------------------------------------------------- checks
def checks(s):
    """Internal consistency of the extracted HS against the report totals."""
    out = {}
    reps = s["_reps"]
    top = reps["gemm"]
    # 1. Instance category at every module = sum of its Instance rows
    ok = True
    for m, r in reps.items():
        for c in ("LUT", "FF", "DSP"):
            rows = sum(num(i.get(c)) or 0 for i in r["instances"])
            if rows != (r["util"].get("Instance", {}).get(c, 0)):
                ok = False
    out["instance_rows_sum_to_category"] = ok
    # 2. Memory BRAM by array sums to top Memory category
    mem = sum(v.get("BRAM_18K", 0) for v in s["hs"]["memories"].values())
    out["memory_bram_sums"] = mem == top["util"].get("Memory", {}).get("BRAM_18K", 0)
    # 3. Bottom-up resource sum: own categories (excl. Instance submodules) x multiplicity, + fp/aux
    # instance costs, reproduces top totals
    for c in ("LUT", "FF", "DSP", "BRAM_18K"):
        tot = 0
        for m, r in reps.items():
            k = s["hs"]["module_hierarchy"]
            mult = next(x["instantiations"] for x in k if x["module"] == m)
            own = sum(v.get(c, 0) for cat, v in r["util"].items() if cat != "Instance")
            leaf = sum(num(i.get(c)) or 0 for i in r["instances"] if i.get("Module") not in reps)
            tot += mult * (own + leaf)
        out[f"bottom_up_{c}"] = tot == top["util_total"].get(c)
    # 4. top totals agree with result.json metrics
    met = s["_metrics_result_json"]
    out["targets_match_result_json"] = (
        s["targets"]["latency_cycles"] == met.get("latency_cycles")
        and s["targets"]["LUT"] == met.get("LUT_used") and s["targets"]["FF"] == met.get("FF_used")
        and s["targets"]["DSP"] == met.get("DSP_used") and s["targets"]["BRAM_18K"] == met.get("BRAM_18K_used"))
    out["fits_matches"] = s["targets"]["fits"] == met.get("fits")
    # 5. log pipelining result agrees with report loop table
    # (known tool behaviour: for the fully flattened loop_i_loop_j_loop_k pipeline the log's
    #  scheduling depth is 1 more than the final report's; the report value is authoritative)
    agree = True
    for l in s["hs"]["loops"]:
        if l["pipelined"] and l["log_depth"] is not None:
            d = l["log_depth"] - (l["pipeline_depth"] or 0)
            if not (d == 0 or (d == 1 and l["loop"] == "loop_i_loop_j_loop_k")):
                agree = False
    out["log_depth_matches_report"] = agree
    out["ii_limiter_resolved"] = all(l["ii_limiter"] != "unknown" for l in s["hs"]["loops"])
    return out


# ----------------------------------------------------------------------------- flat summary
def summary_row(s):
    """One row per sample: the key HS scalars (Stage 1 targets / Stage 2 inputs) + final targets."""
    hs, t, cfg = s["hs"], s["targets"], s["inputs"]["config"]
    row = {"sample": s["sample"], "suite": s["suite"], "source": s["source"]}
    for k, v in cfg.items():
        row[f"cfg_{k}"] = v
    # compute loop = innermost pipelined compute loop if any, else innermost compute loop
    comp = [l for l in hs["loops"] if re.search(r"loop_[ijk]", l["loop"])]
    piped = [l for l in comp if l["pipelined"]]
    main = piped[0] if piped else (comp[-1] if comp else None)
    row.update({
        "n_modules": len(hs["module_hierarchy"]),
        "n_compute_module_instances": sum(x["instantiations"] for x in hs["module_hierarchy"]
                                          if re.search(r"loop_[ijk]", x["module"])),
        "compute_loop": main["loop"] if main else None,
        "compute_loop_module": main["module"] if main else None,
        "compute_loop_tc": main["trip_count"] if main else None,
        "compute_loop_pipelined": main["pipelined"] if main else None,
        "compute_ii_target": main["ii_target"] if main else None,
        "compute_ii_achieved": main["ii_achieved"] if main else None,
        "compute_depth": main["pipeline_depth"] if main else None,
        "compute_iteration_latency": main["iteration_latency"] if main else None,
        "compute_ii_limiter": main["ii_limiter"] if main else None,
    })
    for op in ("fmul", "fadd"):
        o = hs["operator_totals"].get(op, {})
        row[f"n_{op}"] = o.get("count", 0)
        row[f"{op}_impl"] = "|".join(o.get("impl", []))
        row[f"{op}_latency"] = "|".join(str(x) for x in o.get("latency", []))
    row["fp_ops_owner"] = "|".join(hs["fp_operator_owner_modules"])
    for cls in ("sparsemux", "shift", "int"):
        row[f"n_aux_{cls}"] = sum(a["count_effective"] for a in hs["aux_operators"] if a["class"] == cls)
    for a in ("lA", "lB", "lC"):
        m = hs["memories"][a]
        row[f"{a}_banks"] = m["banks"]
        row[f"{a}_words"] = "|".join(map(str, m.get("words_per_bank", [])))
        row[f"{a}_bits"] = "|".join(map(str, m.get("bits_per_word", [])))
        row[f"{a}_storage"] = f'{m["storage_ports"]}/{m["storage_impl"]}' if m["core"] else "registers"
        row[f"{a}_BRAM"] = m["BRAM_18K"]
        row[f"{a}_LUT"] = m["LUT"]
        cm = row["compute_loop_module"]
        ap = hs["access_pattern"].get(cm, {}).get(a, {}) if cm else {}
        row[f"{a}_banks_read_by_compute"] = ap.get("banks_accessed", 0)
        row[f"{a}_read_ports_per_bank"] = ap.get("read_ports_per_bank_max", 0)
        row[f"{a}_scalar_inputs_to_compute"] = ap.get("scalar_inputs", 0)
    lb = hs["latency_breakdown"]
    for ph, v in lb["by_phase"].items():
        row[f"lat_{ph}"] = v
    row["lat_fsm_overhead"] = lb["fsm_overhead"]
    row["compute_stage"] = "|".join(st["stage"].replace("gemm_Pipeline_", "") for st in lb["stages"]
                                    if st["phase"] == "compute")
    row["est_clock_ns"] = hs["timing"]["est_ns"]
    for k in ("latency_cycles", "LUT", "FF", "DSP", "BRAM_18K", "fits"):
        row[f"y_{k}"] = t[k]
    for cat in ("Expression", "Instance", "Memory", "Multiplexer", "Register"):
        for r in ("LUT", "FF", "DSP", "BRAM_18K"):
            v = t["top_by_category"].get(cat, {}).get(r, 0)
            row[f"y_top_{cat}_{r}"] = v
    return row


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    sdirs = sorted(glob.glob(os.path.join(root, "samples", "*")))
    skipped = [os.path.basename(d) for d in sdirs if not os.path.exists(os.path.join(d, "report", "gemm_csynth.rpt"))]
    sdirs = [d for d in sdirs if os.path.basename(d) not in skipped]
    if skipped:
        print("skipped (no synthesis report):", ", ".join(skipped))
    st1, st2, summ, loops_rows, ops_rows, mem_rows, mod_rows, chk_rows = [], [], [], [], [], [], [], []
    for sd in sdirs:
        s = build_sample(sd)
        c = checks(s)
        chk_rows.append({"sample": s["sample"], **c})
        st1.append({"sample": s["sample"], "suite": s["suite"], "source": s["source"],
                    "inputs": s["inputs"], "target_hs": s["hs"]})
        st2.append({"sample": s["sample"], "suite": s["suite"], "source": s["source"],
                    "input_hs": s["hs"], "targets": s["targets"]})
        summ.append(summary_row(s))
        for l in s["hs"]["loops"]:
            loops_rows.append({"sample": s["sample"], **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                                       for k, v in l.items()}})
        for o in s["hs"]["operators"]:
            ops_rows.append({"sample": s["sample"], "clock_ns": s["inputs"]["clock_ns"],
                             **{k: v for k, v in o.items() if k != "unit_cost"},
                             **{f"unit_{k}": v for k, v in o["unit_cost"].items()}})
        for o in s["hs"]["aux_operators"]:
            ops_rows.append({"sample": s["sample"], "clock_ns": s["inputs"]["clock_ns"], "owner": o["owner"],
                             "op": o["class"], "core": o["core"], "count_in_module": o["count_in_module"],
                             "count_effective": o["count_effective"],
                             **{f"unit_{k}": v for k, v in o["unit_cost"].items()}})
        for a, m in s["hs"]["memories"].items():
            mem_rows.append({"sample": s["sample"], "array": a,
                             **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in m.items()}})
        for m in s["targets"]["per_module"]:
            pm = s["targets"]["per_module"][m]
            h = next(x for x in s["hs"]["module_hierarchy"] if x["module"] == m)
            row = {"sample": s["sample"], "module": m, "instantiations": pm["instantiations"],
                   "latency": pm["latency"], "fsm_state_bits": h["fsm_state_bits"]}
            for cat in ("Expression", "Instance", "Memory", "Multiplexer", "Register"):
                for r in ("LUT", "FF", "DSP", "BRAM_18K"):
                    row[f"{cat}_{r}"] = pm["by_category"].get(cat, {}).get(r, 0)
            for r in ("LUT", "FF", "DSP", "BRAM_18K"):
                row[f"total_{r}"] = pm["total"].get(r, 0)
            mod_rows.append(row)
    with open(os.path.join(out, "stage1_dataset.jsonl"), "w") as f:
        for r in st1:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "stage2_dataset.jsonl"), "w") as f:
        for r in st2:
            f.write(json.dumps(r) + "\n")
    write_csv(os.path.join(out, "hs_summary.csv"), summ)
    write_csv(os.path.join(out, "hs_loops.csv"), loops_rows)
    write_csv(os.path.join(out, "hs_operators.csv"), ops_rows)
    write_csv(os.path.join(out, "hs_memories.csv"), mem_rows)
    write_csv(os.path.join(out, "hs_modules.csv"), mod_rows)
    write_csv(os.path.join(out, "extraction_checks.csv"), chk_rows)
    # aggregate check report
    agg = {k: sum(1 for r in chk_rows if r[k]) for k in chk_rows[0] if k != "sample"}
    print(json.dumps({"samples": len(chk_rows), "checks_passed": agg}, indent=1))


if __name__ == "__main__":
    main()
