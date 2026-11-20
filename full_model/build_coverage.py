"""Design-space coverage of the dataset (inputs only: the pragma configs, no tool output).
Writes coverage.json: for every config field, and for pipeline_loop x {unroll_k, unroll_j,
pipeline_ii, pipeline_style}, how many of the 510 designs use each value. predict.py flags a design
whose values are absent from the data (the rules were never exercised there) or sparse (< 3 designs).
Usage: python3 build_coverage.py"""
import json, os, collections
import evaluate

PAIRS = [('pipeline_loop', 'unroll_k'), ('pipeline_loop', 'unroll_j'), ('pipeline_loop', 'pipeline_ii'),
         ('pipeline_loop', 'pipeline_style')]


def main():
    path = evaluate.DATA.replace('stage2_dataset', 'stage1_dataset')
    single = collections.defaultdict(collections.Counter)
    pair = collections.defaultdict(collections.Counter)
    for l in open(path):
        cf = json.loads(l)['inputs']['config']
        for k, v in cf.items():
            single[k][str(v)] += 1
        for a, b in PAIRS:
            pair[f'{a}|{b}'][f'{cf[a]}|{cf[b]}'] += 1
    out = {'single': {k: dict(v) for k, v in single.items()}, 'pair': {k: dict(v) for k, v in pair.items()}}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coverage.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
