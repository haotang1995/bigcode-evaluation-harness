#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import numpy as np
import sys
sys.set_int_max_str_digits(0)

import matplotlib.pyplot as plt
import seaborn as sns

curdir = osp.dirname(osp.abspath(__file__))
curdir = osp.join(curdir, 'gen_tests', 'gen_asserts', 'eval_results_faster_mp')
visdir = osp.join(curdir, 'visualization')
os.makedirs(visdir, exist_ok=True)

def visualize_statistics(results, valid_check_func, fn):
    valid_ratio_list = [
        np.mean([valid_check_func(test) for test in task])
        for task in results
    ]
    sns.displot(valid_ratio_list, kde=True,)
    plt.xlabel('Valid ratio')
    plt.ylabel('Density')
    plt.title(fn.replace('.json', ''))
    plt.savefig(osp.join(visdir, fn.replace('.json', '.png')))
    plt.close()

def visualize_tests(fn):
    print('====================')
    print('Visualizing {}'.format(fn))
    with open(osp.join(curdir, fn), 'r') as f:
        # list of list of list of dict, [task-num, sample-num, test-num]
        # dict of keys {'use_line', 'print_line', 'assert_line', 'syntax_out',
        # 'exec_out', 'assert_out', 'args', 'out', 'line'}
        results = json.load(f)

    # Merge tests
    results = [[test for sample in task for test in sample] for task in results]
    valid_check_func = lambda x: 'pass' in x['assert_out']
    visualize_statistics(results, valid_check_func, fn)

    # Get unique tests
    unique_results = []
    for task in results:
        unique_results.append([])
        unique_ids = set()
        for test in task:
            id = str(test['args']) + '<<>>' + str(test['out'])
            if id not in unique_ids:
                unique_results[-1].append(test)
                unique_ids.add(id)
    visualize_statistics(unique_results, valid_check_func, fn.replace('.json', '_unique.json'))

def main():
    assert osp.exists(curdir)
    filenames = os.listdir(curdir)
    filenames = [fn for fn in filenames if fn.endswith('.json') and not fn.endswith('_tmp.json')]
    for fn in filenames:
        visualize_tests(fn)

if __name__ == '__main__':
    main()
