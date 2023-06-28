#!/usr/bin/env python
# coding=utf-8

from datasets import load_dataset
from evaluate import load

import random
import json
import os, os.path as osp
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

def main():
    dataset = load_dataset('openai_humaneval')
    references = ['\n'+data['test']+'\n'+f'check({data["entry_point"]})' for data in dataset['test']]
    code_eval = load('code_eval')

    curdir = osp.dirname(osp.abspath(__file__))
    generation_dir = osp.join(curdir, 'generation_results')
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

if __name__ == '__main__':
    main()
