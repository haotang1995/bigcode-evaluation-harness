#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import random

from evaluate import load
from datasets import load_dataset

os.environ['HF_ALLOW_CODE_EVAL'] = '1'
datasets = load_dataset('openai_humaneval')
code_eval = load('code_eval',)

def eval_test_case(data, test_case):
    # test_case: {'args': List, 'out': Any, 'line': str,}
    assert(data['entry_point'] in test_case['line'])
    use_line = data['entry_point'] + '(' + ', '.join(test_case['args']) + ')'
    print_line = 'assert False, '+use_line
    assert_line = test_case['line']

    sol = data['prompt'] + data['canonical_solution']

    references = [use_line, print_line, assert_line]
    predictions = [[sol], [sol], [sol]]
    _, out = code_eval.compute(references=references, predictions=predictions, k=[1,])

    use_out = out[0][0][1]
    print_out = out[1][0][1]
    assert_out = out[2][0][1]

    syntax_error = use_out['result']
    if syntax_error.startswith('failed:'):
        syntax_error = syntax_error[7:].strip()

    exec_out = print_out['result']
    if exec_out.startswith('failed:'):
        exec_out = exec_out[7:].strip()
    elif 'timed out' in exec_out:
        exec_out = 'timed out'
    else:
        assert False, (exec_out, print_line, test_case, data)

    assert_out = assert_out['result']

    return {
        'use_line': use_line,
        'print_line': print_line,
        'assert_line': assert_line,

        'syntax_out': syntax_error,
        'exec_out': exec_out,
        'assert_out': assert_out,

        'args': test_case['args'],
        'out': test_case['out'],
        'line': test_case['line'],
    }


def eval_parse_file(fn, parse_dir, eval_dir):
    print(f'evaluating {fn}')
    if osp.exists(osp.join(eval_dir, fn)) and osp.getctime(osp.join(eval_dir, fn)) > osp.getctime(osp.join(parse_dir, fn)):
        print(f'{fn} already exists')
        return
    with open(osp.join(parse_dir, fn), 'r') as f:
        parse_results = json.load(f)
        assert(len(parse_results) == len(datasets['test']))

    tmp_eval_results_filename = osp.join(eval_dir, fn.replace('.json', '_tmp.json'))
    if osp.exists(tmp_eval_results_filename):
        with open(tmp_eval_results_filename, 'r') as f:
            tmp_eval_results = json.load(f)
        eval_results = tmp_eval_results['eval_results']
        prv_di = tmp_eval_results['di']
        print(f'loading {tmp_eval_results_filename}, prv_di={prv_di}')
    else:
        eval_results = []
        prv_di = -1

    for di, (data, ps) in enumerate(zip(datasets['test'], parse_results)):
        if di <= prv_di:
            continue
        for pi, pps in enumerate(ps):
            for pppi, ppps in enumerate(pps):
                eval_results.append(eval_test_case(data, ppps))
        with open(tmp_eval_results_filename, 'w') as f:
            json.dump({
                'eval_results': eval_results,
                'di': di,
            }, f)
    with open(osp.join(eval_dir, fn), 'w') as f:
        json.dump(eval_results, f)

def main(curdir):
    parse_dir = osp.join(curdir, 'parse_results')
    if not osp.exists(parse_dir):
        print(f'{parse_dir} does not exist')
        return
    eval_dir = osp.join(curdir, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)

    filenames = os.listdir(parse_dir)
    filenames = [f for f in filenames if f.endswith('.json')]
    random.shuffle(filenames)
    for fn in filenames:
        eval_parse_file(fn, parse_dir, eval_dir)

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.fn is None:
        curdir = osp.dirname(osp.abspath(__file__))
        curdir = osp.join(curdir, 'gen_tests')
        main(osp.join(curdir, 'gen_asserts'))
        main(osp.join(curdir, 'gen_doctest'))
        main(osp.join(curdir, 'gen_unittest'))
    else:
        fn = args.fn
        assert osp.isfile(fn)
        parse_dir = osp.join(osp.dirname(osp.abspath(fn)), 'parse_results')
        eval_dir = osp.join(osp.dirname(osp.abspath(fn)), 'eval_results')
        eval_parse_file(osp.basename(fn), parse_dir, eval_dir)

