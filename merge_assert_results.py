#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import sys
sys.set_int_max_str_digits(0)

curdir = osp.dirname(osp.abspath(__file__))
eval_dir = osp.join(curdir, 'gen_tests', 'gen_asserts', 'eval_results_faster_mp')
test_dir = osp.join(curdir, 'gen_tests',)

def merge(fn):
    if osp.exists(osp.join(test_dir, fn)) and osp.exists(osp.join(test_dir, fn.replace('.json', '_unique.json'))):
        return
    print('Merging %s' % fn)
    with open(osp.join(eval_dir, fn), 'r') as f:
        gen = json.load(f) # list of list of list of dicts
    asserts = [[{'assert_line': test['assert_line'], 'print_line': test['use_line'], 'out': test['out']} for sample in task for test in sample] for task in gen]
    with open(osp.join(test_dir, fn), 'w') as f:
        json.dump(asserts, f, indent=4)
    unique_asserts = []

    for task in asserts:
        unique_asserts.append([])
        unique_assert_lines = set()
        for test in task:
            if test['assert_line'] not in unique_assert_lines:
                unique_assert_lines.add(test['assert_line'])
                unique_asserts[-1].append(test)
    with open(osp.join(test_dir, fn.replace('.json', '_unique.json')), 'w') as f:
        json.dump(unique_asserts, f, indent=4)

def main():
    filenames = os.listdir(eval_dir)
    filenames = [fn for fn in filenames if fn.endswith('.json')]
    for fn in filenames:
        merge(fn)

if __name__ == '__main__':
    main()
