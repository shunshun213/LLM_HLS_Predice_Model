"""HS extractor extension: shift expressions from the per-module csynth reports.

The Expression table of `report/<module>_csynth.rpt` lists every shl / lshr / ashr expression
with its FF / LUT and bit widths. The pipelined 2.5 ns shl (2661 FF / 3250 LUT) that writes a lane
into a wide reshaped word is part of the true hardware structure; `evaluate.load()` adds it to the
true HS as an aux 'shift' core (op 'shl'), as Stage 1 rule X-SHL predicts it.

Usage: python3 extract_expression_shifts.py <dataset_root>/samples <out.json>
(the repository copy of the output is full_dataset/hs/expression_shifts.json)
"""
import glob, json, os, re, sys


def extract(samples_dir):
    out = {}
    for d in sorted(os.listdir(samples_dir)):
        for f in glob.glob(os.path.join(samples_dir, d, 'report', '*_csynth.rpt')):
            m = os.path.basename(f)[:-len('_csynth.rpt')]
            if m == 'csynth':
                continue
            txt = open(f, errors='ignore').read()
            sec = re.search(r'\* Expression:(.*?)\n\s*\* ', txt, re.S)
            if not sec:
                continue
            rows = re.findall(r'\|\s*(\S+)\s*\|\s*(shl|lshr|ashr)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|',
                              sec.group(1))
            if rows:
                out.setdefault(d, {})[m] = [dict(name=r[0], op=r[1], FF=int(r[3]), LUT=int(r[4]),
                                                 w0=int(r[5]), w1=int(r[6])) for r in rows]
    return out


if __name__ == '__main__':
    json.dump(extract(sys.argv[1]), open(sys.argv[2], 'w'), indent=0)
