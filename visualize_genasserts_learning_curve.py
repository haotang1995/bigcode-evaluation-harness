#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

import json
import os, os.path as osp

curdir = osp.dirname(osp.abspath(__file__))
eval_dir = osp.join(curdir, 'gen_tests', 'gen_asserts', 'eval_results_faster_mp', 'statistics')
assert osp.exists(eval_dir)
vis_dir = osp.join(curdir, 'visualization', 'gen_asserts',)
os.makedirs(vis_dir, exist_ok=True)

def visualize_learning_curve(config_name, fn_list,):
    print('Visualizing learning curve for %s' % config_name)
    eval_results = dict()
    for fn in fn_list:
        with open(osp.join(eval_dir, fn), 'r') as f:
            out = json.load(f)

        epoch_num = [c for c in fn.split('_') if c.startswith('epoch')]
        assert len(epoch_num) == 1
        epoch_num = int(epoch_num[0].strip('epoch'))
        eval_results[epoch_num] = out

    sorted_epoch_num = sorted(eval_results.keys())
    level1_keys = [k for k, v in eval_results[5].items() if isinstance(v, dict)]
    for level1_key in level1_keys:
        values = [eval_results[epoch_num][level1_key]['mean'] for epoch_num in sorted_epoch_num]
        plt.figure()
        plt.plot(sorted_epoch_num, values, 'b-x')
        plt.xlabel('Epoch')
        plt.ylabel('Mean %s' % level1_key)
        plt.title(config_name)
        plt.savefig(osp.join(vis_dir, config_name+'_'+level1_key+'.png'))
        plt.close()


def main():
    gen_filenames = os.listdir(eval_dir)
    gen_filenames = [fn for fn in gen_filenames if '_epoch' in fn]

    configs = dict()
    for fn in gen_filenames:
        config_name = fn.split('.')[0].strip()
        config_name = '_'.join([c for c in config_name.split('_') if not c.startswith('epoch')])
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(fn)

    for config_name, fn_list in configs.items():
        visualize_learning_curve(config_name, fn_list,)

if __name__ == '__main__':
    main()
