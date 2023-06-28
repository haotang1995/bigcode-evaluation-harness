#!/usr/bin/env python
# coding=utf-8

from datasets import load_dataset, concatenate_datasets
from evaluate import load

import random
import json
import os, os.path as osp
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

def main():
    dataset = load_dataset('mbpp', 'sanitized', use_auth_token=True)
    dataset = dataset.map(mbpp_preprocess)
    dataset['test'] = concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test'],
        dataset['prompt'],
    ])
    references = ['\n'+data['test']+'\n'+f'check({data["entry_point"]})' for data in dataset['test']]
    code_eval = load('code_eval')

    curdir = osp.dirname(osp.abspath(__file__))
    generation_dir = osp.join(curdir, 'mbpp_generation_results')
    assert osp.exists(generation_dir), f'Generation directory {generation_dir} does not exist'
    eval_dir = osp.join(generation_dir, 'eval_results')
    if not osp.exists(eval_dir):
        os.makedirs(eval_dir)
    eval_filename_list = os.listdir(eval_dir)

    generation_filenames = os.listdir(generation_dir)
    random.shuffle(generation_filenames)
    for fn in generation_filenames:
        if fn.endswith('.json') and (fn not in eval_filename_list or osp.getctime(osp.join(generation_dir, fn)) > osp.getctime(osp.join(eval_dir, fn))):
            print('Evaluating {}'.format(fn))
            with open(osp.join(generation_dir, fn), 'r') as f:
                gen = json.load(f)
            # Remove '\nAnswer: ' from gen
            gen = [[gg.replace('\nAnswer: ', '') for gg in g] for g in gen]
            # Remove '<commit_after>' from gen
            gen = [[gg.replace('<commit_after>', '') for gg in g] for g in gen]
            results, out = code_eval.compute(references=references, predictions=gen, k=[1, 10])
            print(results)
            print()
            with open(osp.join(eval_dir, fn), 'w') as f:
                json.dump({'results': results, 'out': out}, f,)

def mbpp_preprocess(example):
    prompt = example["prompt"]
    code = example["code"]
    test_list = example["test_list"]
    test_imports = example["test_imports"]

    func_name_list = [
        c[4:].split('(')[0].strip()
        for c in code.split('\n')
        if 'def ' == c[:4]
    ]
    func_name = [
        fn
        for fn in func_name_list
        if all([fn in test for test in test_list])
    ]
    assert(len(func_name) == 1), f"func_name: {func_name}, func_name_list: {func_name_list}, test_list: {test_list}"
    func_name = func_name[0]

    code_blocks = [[],]
    for c in code.split('\n'):
        if c.startswith('def '):
            code_blocks.append([])
        code_blocks[-1].append(c)
    test_imports += [c for c in code_blocks[0] if c.startswith('import ') or c.startswith('from ')]
    code_blocks = code_blocks[1:]
    assert(all([c[0].startswith('def ') for c in code_blocks]))

    func_def = [c for c in code.split('\n') if f'def {func_name}(' in c or f'def {func_name} (' in c]
    assert(len(func_def) == 1), f"func_def: {func_def}, func_name: {func_name}, code: {code}"
    func_def = func_def[0]

    func_def_index = [cbi for cbi, cb in enumerate(code_blocks) if func_def.strip() == cb[0].strip()]
    assert(len(func_def_index) == 1), f"func_def_index: {func_def_index}, func_def: {func_def}, code_blocks: {code_blocks}"
    func_def_index = func_def_index[0]
    code_blocks = code_blocks[:func_def_index] + code_blocks[func_def_index+1:] + [code_blocks[func_def_index][1:]]

    main_cb_line = code_blocks[-1][0]
    indent = main_cb_line[:len(main_cb_line) - len(main_cb_line.lstrip())]

    assert(all([test.strip().startswith('assert') for test in test_list]))
    running_examples = [test.strip()[6:].strip() for test in test_list]

    new_prompt = '\n'.join(test_imports) + '\n'
    new_prompt += func_def + '\n'
    new_prompt += indent+'"""\n'
    new_prompt += '\n'.join([indent+p for p in prompt.split('\n')]) + '\n'
    new_prompt += '\n'.join([indent+'>>> '+p for p in running_examples]) + '\n'
    new_prompt += indent+'"""\n'

    new_code = '\n'.join(['\n'.join([indent+c for c in cb]) for cb in code_blocks[:-1]])
    new_code += '\n'.join(code_blocks[-1])

    example["prompt"] = new_prompt
    example["code"] = new_code
    example["canonical_solution"] = new_code
    example["entry_point"] = func_name
    assert all([func_name in test for test in example["test_list"]])
    example["test"] = 'def check(candidate)\n' + '\n'.join([indent+test.replace(func_name, 'candidate') for test in example["test_list"]])
    return example

if __name__ == '__main__':
    main()
