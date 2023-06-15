#!/usr/bin/env python
# coding=utf-8

from datasets import load_dataset
import argparse
import json
import os, os.path as osp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

# Remove fold$i from cv_gen_filenames to create gen_filenames
def remove_fold_from_filename(filename):
    index = filename.find('fold')
    assert index != -1
    return filename[:index-1] + filename[index+5:]

def main(args):
    dataset = load_dataset('openai_humaneval')
    shuffled_dataset = dataset.shuffle(seed=args.seed)['test']
    dataset = dataset['test']
    # Get the inverse order of shuffling
    order = [None,]*len(shuffled_dataset)
    for i, data in enumerate(shuffled_dataset):
        order[int(data['task_id'].split('/')[-1])] = i
    for i, data in enumerate(dataset):
        assert(data['task_id'] == shuffled_dataset[order[i]]['task_id'])

    cur_dir = osp.dirname(osp.abspath(__file__))
    cv_gen_dir = osp.join(cur_dir, 'cv_generation_results')
    gen_dir = osp.join(cur_dir, 'generation_results')

    cv_gen_filenames = [fn for fn in os.listdir(cv_gen_dir) if fn.endswith('.json')]
    gen_filenames = list(set([remove_fold_from_filename(fn) for fn in cv_gen_filenames]))

    for gen_filename in gen_filenames:
        cur_cv_gen_filenames = sorted([fn for fn in cv_gen_filenames if remove_fold_from_filename(fn) == gen_filename])
        if osp.exists(osp.join(gen_dir, gen_filename)) and osp.getctime(osp.join(gen_dir, gen_filename)) > osp.getctime(osp.join(cv_gen_dir, cur_cv_gen_filenames[0])):
            continue
        if len(cur_cv_gen_filenames) < 5:
            continue
        assert(len(cur_cv_gen_filenames) == 5), (gen_filename, cur_cv_gen_filenames)
        print("Merging %s" % gen_filename)

        generations = []
        for cv_gen_filename in cur_cv_gen_filenames:
            with open(osp.join(cv_gen_dir, cv_gen_filename), 'r') as f:
                generations += json.load(f)
        generations = [generations[i] for i in order]
        for di, (data, gen) in enumerate(zip(dataset, generations)):
            for g in gen:
                if (data['prompt'].strip() not in g):
                    print(di, data)
                    print('---')
                    print(data['prompt'].strip())
                    print('---')
                    print(g)
                    assert(False)

        assert(len(generations) == 164)
        with open(osp.join(gen_dir, gen_filename), 'w') as f:
            json.dump(generations, f)

if __name__ == '__main__':
    args = get_args()
    main(args)
