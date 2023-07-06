#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import numpy as np
import sys
sys.set_int_max_str_digits(0)

def print_statistics(results, valid_check_func, prefix):
    statistics = dict()
    print("Number of tasks: {}".format(len(results)))
    print("Number of samples per task: {}".format(set([len(task) for task in results])))
    statistics.update({
        prefix+'num_tasks': len(results),
        prefix+'num_samples_per_task': list(set([len(task) for task in results])),
    })

    validity_num = [
        [
            sum([1 for test in sample])
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
    statistics.update({
        prefix+'num_tests_per_sample': {
            'mean': np.mean(validity_num),
            'std': np.std(validity_num),
            'min': np.min(validity_num),
            'max': np.max(validity_num),
        },
        prefix+'num_tests_per_task': {
            'mean': np.mean(np.mean(validity_num, axis=1)),
            'std': np.std(np.mean(validity_num, axis=1)),
            'min': np.min(np.mean(validity_num, axis=1)),
            'max': np.max(np.mean(validity_num, axis=1)),
        },
    })

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
    statistics.update({
        prefix+'ratio_valid_tests_per_sample': {
            'mean': np.mean(validity_ratio),
            'std': np.std(validity_ratio),
            'min': np.min(validity_ratio),
            'max': np.max(validity_ratio),
        },
        prefix+'ratio_valid_tests_per_task': {
            'mean': np.mean(np.mean(validity_ratio, axis=1)),
            'std': np.std(np.mean(validity_ratio, axis=1)),
            'min': np.min(np.mean(validity_ratio, axis=1)),
            'max': np.max(np.mean(validity_ratio, axis=1)),
        },
    })

    unique_results = []
    for task in results:
        unique_results.append(dict())
        for sample in task:
            for test in sample:
                id = str(test['args']) + '<<>>' + str(test['out'])
                if id not in unique_results[-1]:
                    test['count'] = 1
                    unique_results[-1][id] = test
                else:
                    unique_results[-1][id]['count'] += 1
        unique_results[-1] = list(unique_results[-1].values())
        sum_count = sum([test['count'] for test in unique_results[-1]])
        # assert sum_count == len(task), "sum_count={}, len(task)={}".format(sum_count, len(task))
        for test in unique_results[-1]:
            test['ratio'] = test['count'] / sum_count
        unique_results[-1] = sorted(unique_results[-1], key=lambda x: x['count'], reverse=True)

    unique_validity_num = [
        sum([1 for test in sample])
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

    statistics.update({
        prefix+'num_unique_tests_per_task': {
            'mean': np.mean(unique_validity_num),
            'std': np.std(unique_validity_num),
            'min': np.min(unique_validity_num),
            'max': np.max(unique_validity_num),
        },
        prefix+'ratio_unique_valid_tests_per_task': {
            'mean': np.mean(unique_validity_ratio),
            'std': np.std(unique_validity_ratio),
            'min': np.min(unique_validity_ratio),
            'max': np.max(unique_validity_ratio),
        },
    })

    for k in [1, 2, 3, 4, 5, 10,]:
        topk_results = [res[:k] for res in unique_results]
        topk_validity_num = [
            sum([1 for test in sample])
            for sample in topk_results
        ]
        print("Statistics of number of top-{} unique tests per task: mean={}, std={}, min={}, max={}".format(
            k,
            np.mean(topk_validity_num),
            np.std(topk_validity_num),
            np.min(topk_validity_num),
            np.max(topk_validity_num),
        ))
        topk_validity_ratio = [
            np.mean([valid_check_func(test) for test in sample])
            for sample in topk_results
        ]
        print("Statistics of ratio of top-{} unique valid tests per task: mean={}, std={}, min={}, max={}".format(
            k,
            np.mean(topk_validity_ratio),
            np.std(topk_validity_ratio),
            np.min(topk_validity_ratio),
            np.max(topk_validity_ratio),
        ))
        statistics.update({
            prefix+'num_top{}_unique_tests_per_task'.format(k): {
                'mean': np.mean(topk_validity_num),
                'std': np.std(topk_validity_num),
                'min': np.min(topk_validity_num),
                'max': np.max(topk_validity_num),
            },
            prefix+'ratio_top{}_unique_valid_tests_per_task'.format(k): {
                'mean': np.mean(topk_validity_ratio),
                'std': np.std(topk_validity_ratio),
                'min': np.min(topk_validity_ratio),
                'max': np.max(topk_validity_ratio),
            },
        })

        if k == 1:
            # Print out wrong asserts
            for task in topk_results:
                for test in task:
                    if not valid_check_func(test):
                        print(test)

    return statistics

def analyze_tests(fn):
    filedir = os.path.dirname(fn)
    filename = os.path.basename(fn)
    outdir = os.path.join(filedir, 'statistics')
    os.makedirs(outdir, exist_ok=True)
    outfn = os.path.join(outdir, filename)
    if os.path.exists(outfn) and os.path.getmtime(outfn) > os.path.getmtime(fn):
        return

    print('====================')
    print('Analyzing {}'.format(fn))
    with open(fn, 'r') as f:
        # list of list of list of dict, [task-num, sample-num, test-num]
        # dict of keys {'use_line', 'print_line', 'assert_line', 'syntax_out',
        # 'exec_out', 'assert_out', 'args', 'out', 'line'}
        results = json.load(f)

    statistics = {}

    print()
    print('Statistics of Generated Tests:')
    valid_check_func = lambda x: True
    statistics.update(print_statistics(results, valid_check_func, 'all_'))

    print()
    print('Statistics of Generated Tests with Valid Inputs:')
    valid_check_func = lambda x: 'pass' in x['syntax_out']
    statistics.update(print_statistics(results, valid_check_func, 'valid_inputs_'))

    print()
    print('Statistics of Generated Tests with Valid Inputs & Outputs:')
    valid_check_func = lambda x: 'pass' in x['assert_out']
    statistics.update(print_statistics(results, valid_check_func, 'valid_inputs_outputs_'))

    # Make it serializable by converting numpy types to python types
    for k, v in statistics.items():
        if isinstance(v, np.float64):
            statistics[k] = float(v)
        elif isinstance(v, np.int64):
            statistics[k] = int(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, np.float64):
                    statistics[k][k2] = float(v2)
                elif isinstance(v2, np.int64):
                    statistics[k][k2] = int(v2)
                else:
                    assert not isinstance(v2, np.generic), type(v2)
        else:
            assert not isinstance(v, np.generic), type(v)
    with open(outfn, 'w') as f:
        json.dump(statistics, f, indent=2)

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
