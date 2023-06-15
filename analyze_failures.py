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

def get_failure_pattern(generations, eval_results, empty_errors, data=None,):
    if data is None:
        reshaped_generations = generations
        reshaped_eval_results = eval_results
        reshaped_empty_errors = empty_errors
        generations = [gg for g in generations for gg in g]
        eval_results = [er for er in eval_results for er in er]
        empty_errors = [ee for ee in empty_errors for ee in ee]
    assert(len(eval_results) == len(empty_errors)), "Error: The number of eval_results and empty_errors should be the same"
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
    print('Empty Code Ratio: %.2f%%' % (len([er for er in empty_errors if er]) / total_samples * 100))

    if not verbose_data:
        # Print more analysis task-wise
        assert(len(eval_results) % 164 == 0), "Error: The number of generations is not a multiple of 164"
        pass_at_1_list = [np.mean([1 if er['passed'] else 0 for er in eval_result]) for eval_result in reshaped_eval_results]
        assert(len(reshaped_empty_errors) == len(reshaped_eval_results) and len(reshaped_empty_errors[0]) == len(reshaped_eval_results[0])), "Error: The number of reshaped empty errors is not the same as the number of reshaped eval results"
        empty_error_ratio_list = [np.mean([1 if er else 0 for er in empty_error]) for empty_error in reshaped_empty_errors]
        # Percentage of pass@1 == 0%
        print('Pass@1 == 0%%: %.2f%%' % (len([p for p in pass_at_1_list if p <= 0.00]) / len(pass_at_1_list) * 100))
        # Percentage of pass@1 == 0% and empty code ratio >= 95%
        print('Pass@1 == 0%% and empty code ratio >= 95%%: %.2f%%' % (len([p for p, e in zip(pass_at_1_list, empty_error_ratio_list) if p <= 0.00 and e >= 0.95]) / len(pass_at_1_list) * 100))
        # Percentage of pass@1 <= 5%
        print('Pass@1 <= 5%%: %.2f%%' % (len([p for p in pass_at_1_list if p <= 0.05]) / len(pass_at_1_list) * 100))
        # Percentage of pass@1 >= 95%
        print('Pass@1 >= 95%%: %.2f%%' % (len([p for p in pass_at_1_list if p >= 0.95]) / len(pass_at_1_list) * 100))
        # Percentage of pass@1 == 100%
        print('Pass@1 == 100%%: %.2f%%' % (len([p for p in pass_at_1_list if p >= 1.]) / len(pass_at_1_list) * 100))
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
        # Print examples without empty errors
        if verbose_data and pass_at_1 <= 0.05:
            print('Examples without empty errors:')
            count = 0
            for generation, er in zip(generations, eval_results):
                if not er['passed'] and not empty_errors[eval_results.index(er)]:
                    print(generation)
                    count += 1
                    if count >= 2:
                        break
            print('')
    elif verbose_data:
        print('No failure patterns')
        print(generations[0])

def is_empty_code(gen, data):
    gen = gen.replace(data['prompt'], '').strip()
    if 'pass' in gen:
        return True
    lines = [l for l in gen.split('\n') if l.strip() != '' and not l.strip().startswith('#') and not l.strip().startswith('"""')]
    if len(lines) <= 1:
        return True
    if 'return' in lines and len(lines) <= 3:
        return True
    return False

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

    # Get errors that generate empty codes
    empty_errors = [[is_empty_code(gg, data) for gg in gen] for gen, data in zip(generations, dataset)]

    if args.task_id is None:
        # Analayze the failure patterns of the whole dataset, first
        print('Failure patterns of the whole dataset:')
        get_failure_pattern(
            generations, eval_results, empty_errors,
            # [gg for generations_per_task in generations for gg in generations_per_task],
            # [er for eval_result in eval_results for er in eval_result],
            # [ee for empty_errors_per_task in empty_errors for ee in empty_errors_per_task],
        )

    if args.task_id is not None:
        # Analyze the failure patterns of each task
        task_id = args.task_id
        print('====================')
        print('Failure patterns of task %d:' % task_id)
        get_failure_pattern(generations[task_id], eval_results[task_id], empty_errors[task_id], data=dataset[task_id])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='The filename in generation_results')
    parser.add_argument('--task_id', type=int, default=None, help='The task_id of the filename')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
