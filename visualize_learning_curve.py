#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

import json
import os, os.path as osp

curdir = osp.dirname(osp.abspath(__file__))
gen_dir = osp.join(curdir, 'generation_results')
eval_dir = osp.join(gen_dir, 'eval_results')
vis_dir = osp.join(curdir, 'visualization')

def visualize_learning_curve(config_name, fn_list,):
    eval_results = dict()
    for fn in fn_list:
        with open(osp.join(eval_dir, fn), 'r') as f:
            out = json.load(f)
        pass_at_one = out['results']['pass@1']

        epoch_num = [c for c in fn.split('_') if c.startswith('epoch')]
        assert len(epoch_num) == 1
        epoch_num = int(epoch_num[0].strip('epoch'))
        eval_results[epoch_num] = pass_at_one

    sorted_epoch_num = sorted(eval_results.keys())
    sorted_pass_at_one = [eval_results[epoch_num] for epoch_num in sorted_epoch_num]

    plt.figure()
    plt.plot(sorted_epoch_num, sorted_pass_at_one, 'b-x')
    plt.xlabel('Epoch')
    plt.ylabel('Pass@1')
    plt.title(config_name)
    plt.savefig(osp.join(vis_dir, config_name + '.png'))
    plt.close()


def main():
    gen_filenames = os.listdir(gen_dir)
    gen_filenames = [fn for fn in gen_filenames if '_epoch' in fn]

    configs = dict()
    for fn in gen_filenames:
        if not fn.endswith('.json') or not osp.exists(osp.join(eval_dir, fn)):
            continue
        config_name = fn.split('.')[0].strip()
        config_name = '_'.join([c for c in config_name.split('_') if not c.startswith('epoch')])
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(fn)

    for config_name, fn_list in configs.items():
        visualize_learning_curve(config_name, fn_list,)

if __name__ == '__main__':
    main()
