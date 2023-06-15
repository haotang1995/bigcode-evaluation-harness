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
    total_samples = len(eval_results)
    failed_samples = len([er for er in eval_results if not er['passed']])
    pass_at_1 = 1 - failed_samples / total_samples

    verbose_data = False
    if data is not None:
        verbose_data = True
        # Show data samples
        print("@Task-ID:", data['task_id'])
        print("@Prompt:", data['prompt'])
        if pass_at_1 <= 0.98:
            print("@Test:", data['test'])
            print("@Canonical-Solution:", data['canonical_solution'])

    # Print pass@1
    print('Pass@1: %.2f%%' % (pass_at_1*100))

    if not verbose_data:
        # Print more analysis task-wise
        assert(len(eval_results) % 164 == 0), "Error: The number of generations is not a multiple of 164"
        sample_num = len(eval_results) // 164
        reshaped_eval_results = [eval_results[i:i+sample_num] for i in range(0, len(eval_results), 164)]
        pass_at_1_list = [np.mean([1 if er['passed'] else 0 for er in eval_result]) for eval_result in reshaped_eval_results]
        # Percentage of pass@1 == 0
        print('Pass@1 <= 5%%: %.2f%%' % (len([p for p in pass_at_1_list if p <= 0.05]) / len(pass_at_1_list) * 100))
        # Percentage of pass@1 == 1
        print('Pass@1 >= 95%%: %.2f%%' % (len([p for p in pass_at_1_list if p >= 0.95]) / len(pass_at_1_list) * 100))
        # Quantile of pass@1
        for q in [0.25, 0.5, 0.75]:
            print('Pass@1 quantile == %.2f: %.2f%%' % (q, np.quantile(pass_at_1_list, q) * 100))

    failed_reasons = [er['result'].lower().strip() for er in eval_results if not er['passed']]
    failed_reasons_count = sorted([(fr, failed_reasons.count(fr)) for fr in set(failed_reasons)], key=lambda x: x[1], reverse=True)
    if pass_at_1 <= 0.98:
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
    elif verbose_data:
        print('No failure patterns')
        print(generations[0])


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
