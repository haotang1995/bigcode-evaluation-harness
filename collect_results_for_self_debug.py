#!/usr/bin/env python
# coding=utf-8

# Given filenames in generation results
# Retrieve their eval_results & check_syntax_results
# Analyze the results and print out the statistics

import os
import sys
import json
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
generation_results_dir = os.path.join(current_dir, 'generation_results')
generation_filenames = os.listdir(generation_results_dir)
eval_results_dir = os.path.join(generation_results_dir, 'eval_results')
eval_results_filenames = os.listdir(eval_results_dir)
check_syntax_results_dir = os.path.join(generation_results_dir, 'check_syntax_results')
check_syntax_results_filenames = os.listdir(check_syntax_results_dir)
out_dir = os.path.join(generation_results_dir, 'refinement_results', 'gt_test_cases')
os.makedirs(out_dir, exist_ok=True,)

def retrieve_results(fn):
    print('Retrieving results for %s:' % fn)
    fn = os.path.basename(fn).strip()
    if not fn.endswith('.json') or fn not in generation_filenames:
        print('Error: Invalid filename: %s' % fn)
        return

    # Read eval_results
    eval_results_fn = os.path.join(eval_results_dir, fn)
    with open(eval_results_fn, 'r') as f:
        eval_results = json.load(f)
        pass_at_k = eval_results['results']
        print('Eval results:', pass_at_k)
        eval_results = list(eval_results['out'].values())
        eval_results = [[er[1] for er in eval_result] for eval_result in eval_results]
        assert(len(eval_results) == 164), "Error: This script is only valid for testing human-eval"

    # Read check_syntax_results
    check_syntax_results_fn = os.path.join(check_syntax_results_dir, fn)
    with open(check_syntax_results_fn, 'r') as f:
        check_syntax_results = json.load(f) # a list of a list of a list of errors (str)
        assert(len(check_syntax_results) == 164), "Error: This script is only valid for testing human-eval"
        assert(len(check_syntax_results[0]) == len(eval_results[0])), (len(check_syntax_results[0]), len(eval_results[0]), check_syntax_results[0], eval_results[0])

    # Print the syntax error rates
    syntax_error_rates = np.mean([np.mean([len(sr) > 0 for sr in syntax_results]) for syntax_results in check_syntax_results])
    print('Syntax error rate: %.4f' % syntax_error_rates)

    # Filter out the syntax errors and print the pass@one
    filtered_eval_results = [[er for er, sr in zip(eval_result, syntax_result) if len(sr) == 0] for eval_result, syntax_result in zip(eval_results, check_syntax_results)]
    pass_at_one = np.mean([np.mean([er['passed'] for er in eval_result]) if len(eval_result) > 0 else 0 for eval_result in filtered_eval_results])
    print('After filtering out syntax errors, pass@1: %.4f' % pass_at_one)
    print()

    return eval_results, check_syntax_results, pass_at_k

def print_statistics(eval_results, check_syntax_results, prefix, fn):
    # Filter out the syntax errors
    no_syntax_error_eval_results = [[
        er for er, sr in zip(eval_result, syntax_result) if len(sr) == 0
    ] for eval_result, syntax_result in zip(eval_results, check_syntax_results)]
    with_syntax_error_eval_results = [er for er, no_sr_er in zip(eval_results, no_syntax_error_eval_results) if len(no_sr_er) > 0]
    no_syntax_error_eval_results = [er for er in no_syntax_error_eval_results if len(er) > 0]
    print("Number of examples with full syntax errors: %d" % (len(eval_results) - len(no_syntax_error_eval_results)))
    print('Pass@1 before filtering out syntax errors: %.4f' % (np.sum([np.mean([er['passed'] for er in eval_result]) for eval_result in with_syntax_error_eval_results])/164))
    print('Pass@1 after filtering out syntax errors: %.4f' % (np.sum([np.mean([er['passed'] for er in eval_result]) for eval_result in no_syntax_error_eval_results])/164))
    print('Pass@1 before filtering out syntax errors & all syntax error tasks: %.4f' % (np.mean([np.mean([er['passed'] for er in eval_result]) for eval_result in with_syntax_error_eval_results])))
    print('Pass@1 after filtering out syntax errors & all syntax error tasks: %.4f' % (np.mean([np.mean([er['passed'] for er in eval_result]) for eval_result in no_syntax_error_eval_results])))
    print()

    # Save current results
    cur_out_dir = os.path.join(out_dir, prefix)
    os.makedirs(cur_out_dir, exist_ok=True,)
    combined_out = [
        [{**er, 'syntax_errors': sr} for er, sr in zip(eval_result, syntax_result)]
        for eval_result, syntax_result in zip(eval_results, check_syntax_results)
    ]
    results = {
        'pass@1': np.mean([np.mean([er['passed'] for er in eval_result]) for eval_result in combined_out]),
        'pass@1 (no syntax error tasks)': np.sum([np.mean([er['passed'] for er in eval_result]) for eval_result in no_syntax_error_eval_results])/164,
    }
    with open(os.path.join(cur_out_dir, fn), 'w') as f:
        json.dump({
            'results': results,
            'out': combined_out,
        }, f)

def main():
    filenames = sys.argv[1:]
    assert(len(filenames) >= 2), "Error: At least 2 filenames are required"

    results = [retrieve_results(fn) for fn in filenames]

    # Merge results
    eval_results, check_syntax_results, _ = results[0]
    print("Round 0 eval results:")
    print_statistics(eval_results, check_syntax_results, 'round0', filenames[0])

    for ri, (next_round_eval_results, next_round_check_syntax_results, _) in enumerate(results[1:]):
        eval_results = [
            [
                ner if not er['passed'] else er
                for er, ner, csr in zip(eval_result, next_round_eval_result, check_syntax_results)
            ] for eval_result, next_round_eval_result, check_syntax_results in zip(eval_results, next_round_eval_results, check_syntax_results)
        ]
        check_syntax_results = [
            [
                next_round_csr if not er['passed'] else csr
                for er, csr, next_round_csr in zip(eval_result, check_syntax_result, next_round_check_syntax_result)
            ] for eval_result, check_syntax_result, next_round_check_syntax_result in zip(eval_results, check_syntax_results, next_round_check_syntax_results)
        ]
        print("Round %d eval results:" % (ri + 1))
        print_statistics(eval_results, check_syntax_results, 'round%d' % (ri + 1), filenames[ri+1])

if __name__ == '__main__':
    main()
