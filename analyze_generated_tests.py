#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import numpy as np
import sys
sys.set_int_max_str_digits(0)

def print_statistics(results, valid_check_func):
    print("Number of tasks: {}".format(len(results)))
    print("Number of samples per task: {}".format(set([len(task) for task in results])))
    validity_num = [
        [
            sum([valid_check_func(test) for test in sample])
            for sample in task
        ] for task in results
    ] # [task-num, sample-num]
    print("Statistics of number of tests per sample: mean={}, std={}, min={}, max={}".format(
        np.mean(validity_num),
        np.std(validity_num),
        np.min(validity_num),
        np.max(validity_num),
    ))
    print("Statistics of number of tests per task: mean={}, std={}, min={}, max={}".format(
        np.mean(np.mean(validity_num, axis=1)),
        np.std(np.mean(validity_num, axis=1)),
        np.min(np.mean(validity_num, axis=1)),
        np.max(np.mean(validity_num, axis=1)),
    ))

    validity_ratio = [
        [
            np.mean([valid_check_func(test) for test in sample])
            for sample in task
        ] for task in results
    ]
    print("Statistics of ratio of valid tests per sample: mean={}, std={}, min={}, max={}".format(
        np.mean(validity_ratio),
        np.std(validity_ratio),
        np.min(validity_ratio),
        np.max(validity_ratio),
    ))
    print("Statistics of ratio of valid tests per task: mean={}, std={}, min={}, max={}".format(
        np.mean(np.mean(validity_ratio, axis=1)),
        np.std(np.mean(validity_ratio, axis=1)),
        np.min(np.mean(validity_ratio, axis=1)),
        np.max(np.mean(validity_ratio, axis=1)),
    ))

    unique_results = []
    for task in results:
        unique_results.append([])
        unique_ids = set()
        for sample in task:
            for test in sample:
                id = str(test['args']) + '<<>>' + str(test['out'])
                if id not in unique_ids:
                    unique_results[-1].append(test)
                    unique_ids.add(id)
    unique_validity_num = [
        sum([valid_check_func(test) for test in sample])
        for sample in unique_results
    ]
    print("Statistics of number of unique tests per task: mean={}, std={}, min={}, max={}".format(
        np.mean(unique_validity_num),
        np.std(unique_validity_num),
        np.min(unique_validity_num),
        np.max(unique_validity_num),
    ))
    unique_validity_ratio = [
        np.mean([valid_check_func(test) for test in sample])
        for sample in unique_results
    ]
    print("Statistics of ratio of unique valid tests per task: mean={}, std={}, min={}, max={}".format(
        np.mean(unique_validity_ratio),
        np.std(unique_validity_ratio),
        np.min(unique_validity_ratio),
        np.max(unique_validity_ratio),
    ))

def analyze_tests(fn):
    print('====================')
    print('Analyzing {}'.format(fn))
    with open(fn, 'r') as f:
        # list of list of list of dict, [task-num, sample-num, test-num]
        # dict of keys {'use_line', 'print_line', 'assert_line', 'syntax_out',
        # 'exec_out', 'assert_out', 'args', 'out', 'line'}
        results = json.load(f)

    # print()
    # print('Statistics of Generated Tests:')
    # valid_check_func = lambda x: True
    # print_statistics(results, valid_check_func)

    # print()
    # print('Statistics of Generated Tests with Valid Inputs:')
    # valid_check_func = lambda x: 'pass' in x['syntax_out']
    # print_statistics(results, valid_check_func)

    # print()
    print('Statistics of Generated Tests with Valid Inputs & Outputs:')
    valid_check_func = lambda x: 'pass' in x['assert_out']
    print_statistics(results, valid_check_func)

def main(path):
    if not osp.exists(path):
        print('Path not exists: {}'.format(path))
        return

    filenames = os.listdir(path)
    filenames = [fn for fn in filenames if fn.endswith('.json') and not fn.endswith('_tmp.json')]
    filenames = [osp.join(path, fn) for fn in filenames]
    for fn in filenames:
        analyze_tests(fn)

if __name__ == '__main__':
    if sys.argv[1:]:
        for fn in sys.argv[1:]:
            analyze_tests(fn)
    else:
        curdir = osp.dirname(osp.abspath(__file__))
        curdir = osp.join(curdir, 'gen_tests')
        main(osp.join(curdir, 'gen_asserts', 'eval_results_faster_mp'))
        main(osp.join(curdir, 'gen_doctests', 'eval_results_faster_mp'))
        main(osp.join(curdir, 'gen_unittest', 'eval_results_faster_mp'))
