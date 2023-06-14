#!/usr/bin/env python
# coding=utf-8

import json
import os, os.path as osp


# Remove fold$i from cv_gen_filenames to create gen_filenames
def remove_fold_from_filename(filename):
    index = filename.find('fold')
    assert index != -1
    return filename[:index-1] + filename[index+5:]

def main():
    cur_dir = osp.dirname(osp.abspath(__file__))
    cv_gen_dir = osp.join(cur_dir, 'cv_generation_results')
    gen_dir = osp.join(cur_dir, 'generation_results')

    cv_gen_filenames = [fn for fn in os.listdir(cv_gen_dir) if fn.endswith('.json')]
    gen_filenames = list(set([remove_fold_from_filename(fn) for fn in cv_gen_filenames]))

    for gen_filename in gen_filenames:
        if osp.exists(osp.join(gen_dir, gen_filename)):
            continue
        cur_cv_gen_filenames = sorted([fn for fn in cv_gen_filenames if remove_fold_from_filename(fn) == gen_filename])
        assert(len(cur_cv_gen_filenames) == 5), (gen_filename, cur_cv_gen_filenames)

        generations = []
        for cv_gen_filename in cur_cv_gen_filenames:
            with open(osp.join(cv_gen_dir, cv_gen_filename), 'r') as f:
                generations += json.load(f)
        assert(len(generations) == 164)
        with open(osp.join(gen_dir, gen_filename), 'w') as f:
            json.dump(generations, f)

if __name__ == '__main__':
    main()
