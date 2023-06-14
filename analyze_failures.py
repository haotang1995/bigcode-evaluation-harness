#!/usr/bin/env python
# coding=utf-8

# Given a filename in generation_results,
# Retrieve its eval_results & check_syntax_results
# Analyze the results and print out the statistics (including the pass@1,
# failure patterns and their counts, pass@1 before and after filtering out the
# syntax errors)
# Usage: python analyze_failures.py <filename>

import argparse
import os
import json
import numpy as np

from datasets import load_dataset

def get_failure_pattern(generations, eval_results, data=None,):
    verbose_data = False
    if data is not None:
        verbose_data = True
        # Show data samples
        print("@Task-ID:", data['task_id'])
        print("@Prompt:", data['prompt'])
        print("@Test:", data['test'])
        print("@Canonical-Solution:", data['canonical_solution'])

    # Print pass@1
    print('Pass@1: %.4f' % (np.mean([er['passed'] for er in eval_results])))

    total_samples = len(eval_results)
    failed_samples = len([er for er in eval_results if not er['passed']])

    failed_reasons = [er['result'].lower().strip() for er in eval_results if not er['passed']]
    failed_reasons_count = sorted([(fr, failed_reasons.count(fr)) for fr in set(failed_reasons)], key=lambda x: x[1], reverse=True)
    print('Failure patterns:')
    for fi, (fr, count) in enumerate(failed_reasons_count):
        if fi >= 10 and count/total_samples < 0.01:
            break
        print('----')
        # Print failure, count, and the potential improvement if solved
        print('%s: %d (%.2f%%)' % (fr, count, count / total_samples * 100))
        if verbose_data:
            # Print the corresponding generations
            generation_cnt = 0
            print('Corresponding generations:')
            for generation, er in zip(generations, eval_results):
                if not er['passed'] and er['result'].lower().strip() == fr:
                    print(generation)
                    generation_cnt += 1
                    if generation_cnt >= 5:
                        break
            print('')
        print('----')

def main(args):
    dataset = load_dataset('openai_humaneval',)['test']

    current_dir = os.path.dirname(os.path.realpath(__file__))
    generation_results_dir = os.path.join(current_dir, 'generation_results')
    generation_filenames = os.listdir(generation_results_dir)
    eval_results_dir = os.path.join(generation_results_dir, 'eval_results')
    eval_results_filenames = os.listdir(eval_results_dir)

    fn = os.path.basename(args.filename)
    if not fn.endswith('.json') or fn not in generation_filenames:
        print('Error: Invalid filename: %s' % fn)
        return

    # Read generations
    generation_fn = os.path.join(generation_results_dir, fn)
    with open(generation_fn, 'r') as f:
        generations = json.load(f)
        assert(len(generations) == 164), "Error: This script is only valid for testing human-eval"

    # Read eval_results
    eval_results_fn = os.path.join(eval_results_dir, fn)
    with open(eval_results_fn, 'r') as f:
        eval_results = json.load(f)
        pass_at_k = eval_results['results']
        # print('Eval results:', pass_at_k)
        eval_results = list(eval_results['out'].values())
        eval_results = [[er[1] for er in eval_result] for eval_result in eval_results]
        assert(len(eval_results) == 164), "Error: This script is only valid for testing human-eval"
        assert(len(eval_results[0]) == len(generations[0])), "Error: The number of generations and eval_results are not equal"

    if args.task_id is None:
        # Analayze the failure patterns of the whole dataset, first
        print('Failure patterns of the whole dataset:')
        get_failure_pattern(None, [er for eval_result in eval_results for er in eval_result])

    if args.task_id is not None:
        # Analyze the failure patterns of each task
        task_id = args.task_id
        print('====================')
        print('Failure patterns of task %d:' % task_id)
        get_failure_pattern(generations[task_id], eval_results[task_id], data=dataset[task_id])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='The filename in generation_results')
    parser.add_argument('--task_id', type=int, default=None, help='The task_id of the filename')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
