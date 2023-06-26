#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import random
import itertools
import multiprocessing as mp

import sys
sys.set_int_max_str_digits(0)

curdir = osp.dirname(osp.abspath(__file__))
gen_dir = osp.join(curdir, 'generation_results')
assert(osp.exists(gen_dir))
test_dir = osp.join(curdir, 'gen_tests')
assert(osp.exists(test_dir))
eval_dir = osp.join(gen_dir, 'eval_results_on_gen_tests')
os.makedirs(eval_dir, exist_ok=True)

# ==================== EVALUATION ====================

import contextlib, io, signal
@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

class TimeoutException(Exception):
    pass

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

def eval_sol(line, timeout=3.):
    try:
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                exec(line, exec_globals)
        return exec_globals
    except TimeoutException:
        return 'timed out'
    except BaseException as e:
        return f"failed: {e}"

def eval_print(line, exec_globals, timeout=3.):
    try:
        with swallow_io():
            with time_limit(timeout):
                exec('hhhhhh='+line, exec_globals)
        return exec_globals['hhhhhh']
    except TimeoutException:
        return 'timed out'
    except BaseException as e:
        return f"failed: {e}"

def eval_assert(line, exec_globals, timeout=3.):
    try:
        with swallow_io():
            with time_limit(timeout):
                exec(line, exec_globals)
        return 'passed'
    except TimeoutException:
        return 'timed out'
    except BaseException as e:
        return f"failed: {e}"

# ==================== EVALUATION Finished ====================

def eval_test_case(tt, exec_globals, history):
    assert_line = tt['assert_line']
    print_line = tt['print_line']
    out = tt['out']

    if assert_line in history:
        return history[assert_line]

    print_out = str(eval_print(print_line, exec_globals))
    assert_out = eval_assert(assert_line, exec_globals)
    history[assert_line] = {
        'print_out': print_out,
        'assert_out': assert_out,
        'out': out,
    }
    return history[assert_line]

def eval_per_sol(ti, gi, gg, t, tmp_eval_results_filename):
    tmp_eval_results_filename = tmp_eval_results_filename.replace('.json', f'_t{ti}_g{gi}.json')
    if osp.exists(tmp_eval_results_filename):
        with open(tmp_eval_results_filename, 'r') as f:
            return json.load(f)

    exec_globals = eval_sol(gg)
    history = dict()

    eval_results = [eval_test_case(tt, exec_globals, history) for tt in t]
    with open(tmp_eval_results_filename, 'w') as f:
        json.dump(eval_results, f)
    return eval_results

def eval_gen_file_on_test(gen_fn, test_fn):
    eval_test_dir = osp.join(eval_dir, test_fn.replace('.json', ''))
    os.makedirs(eval_test_dir, exist_ok=True)
    eval_results_filename = osp.join(eval_test_dir, gen_fn)
    if osp.exists(eval_results_filename):
        return

    print(f'evaluating {gen_fn} on {test_fn}')
    tmp_dir = osp.join(eval_test_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_eval_results_filename = osp.join(tmp_dir, gen_fn.replace('.json', '_tmp.json'))

    with open(osp.join(gen_dir, gen_fn), 'r') as f:
        gen = json.load(f) # list of list of strings, [test_num, prog_sample_num]
    with open(osp.join(test_dir, test_fn), 'r') as f:
        test = json.load(f) # list of list of dicts, [test_num, test_sample_num]
    assert(len(gen) == len(test))

    eval_results = []
    for ti, (g, t) in enumerate(zip(gen, test)):
        with mp.Pool(8) as pool:
            eval_results.append(
                pool.starmap(eval_per_sol, [(ti, gi, gg, t, tmp_eval_results_filename) for gi, gg in enumerate(g)])
            )
    with open(eval_results_filename, 'w') as f:
        json.dump(eval_results, f)

    for ti in range(len(gen)):
        for gi in range(len(gen[ti])):
            tmp_eval_results_filename = tmp_eval_results_filename.replace('.json', f'_t{ti}_g{gi}.json')
            os.remove(tmp_eval_results_filename)

def main():
    gen_filenames = os.listdir(gen_dir)
    gen_filenames = [f for f in gen_filenames if f.endswith('.json')]
    random.shuffle(gen_filenames)
    test_filenames = os.listdir(test_dir)
    test_filenames = [f for f in test_filenames if f.endswith('.json')]
    random.shuffle(test_filenames)

    for gen_fn, test_fn in itertools.product(gen_filenames, test_filenames):
        eval_gen_file_on_test(gen_fn, test_fn)

if __name__ == '__main__':
    main()

