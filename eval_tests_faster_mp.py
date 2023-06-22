#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import random
import tqdm
import multiprocessing as mp
import sys
sys.set_int_max_str_digits(0)

from datasets import load_dataset

os.environ['HF_ALLOW_CODE_EVAL'] = '1'
datasets = load_dataset('openai_humaneval')

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

def eval_test_case(data, test_case, exec_globals, history):
    # test_case: {'args': List, 'out': Any, 'line': str,}
    assert(data['entry_point'] in test_case['line'])
    use_line = data['entry_point'] + '(' + ', '.join(test_case['args']) + ')'
    assert_line = test_case['line']
    if assert_line in history:
        return history[assert_line]

    out = eval_print(use_line, exec_globals,)
    if out != 'timed out' and (not isinstance(out, str) or not out.startswith('failed: ')):
        exec_out = out
        syntax_error = 'passed'
    else:
        exec_out = None
        syntax_error = out
    assert(isinstance(syntax_error, str) and (
        syntax_error.startswith('failed: ') or
        syntax_error.startswith('timed out') or
        syntax_error.startswith('passed')
    )), f'{syntax_error} is not a valid syntax error'

    assert_out = eval_assert(assert_line, exec_globals,)

    history[assert_line] = {
        'use_line': use_line,
        'assert_line': assert_line,

        'syntax_out': syntax_error,
        'exec_out': exec_out,
        'assert_out': assert_out,

        'args': test_case['args'],
        'out': test_case['out'],
        'line': test_case['line'],
    }
    return history[assert_line]

def eval_per_data(di, data, ps, tmp_eval_results_filenames):
    tmp_eval_results_filename = tmp_eval_results_filenames.replace('tmp.json', f'tmp_{di}.json')
    if osp.exists(tmp_eval_results_filename):
        with open(tmp_eval_results_filename, 'r') as f:
            try:
                eval_results = json.load(f)
                return eval_results
            except json.decoder.JSONDecodeError as e:
                print(f'error in {tmp_eval_results_filename}: {e}')
                os.remove(tmp_eval_results_filename)

    sol = data['prompt'] + '\n' + data['canonical_solution']
    exec_globals = eval_sol(sol)
    history = dict()

    eval_results = []
    for pi, pps in enumerate(ps):
        eval_results.append([])
        for pppi, ppps in enumerate(pps):
            eval_results[-1].append(eval_test_case(data, ppps, exec_globals, history))
    with open(tmp_eval_results_filename, 'w') as f:
        json.dump(eval_results, f)
    return eval_results

def eval_parse_file(fn, parse_dir, eval_dir):
    print(f'evaluating {fn}')
    if osp.exists(osp.join(eval_dir, fn)) and osp.getctime(osp.join(eval_dir, fn)) > osp.getctime(osp.join(parse_dir, fn)):
        print(f'{fn} already exists')
        return
    with open(osp.join(parse_dir, fn), 'r') as f:
        parse_results = json.load(f)
        assert(len(parse_results) == len(datasets['test']))

    tmp_eval_dir = osp.join(eval_dir, 'tmp')
    os.makedirs(tmp_eval_dir, exist_ok=True)
    tmp_eval_results_filename = osp.join(tmp_eval_dir, fn.replace('.json', '_tmp.json'))

    with mp.Pool(8) as pool:
        eval_results = pool.starmap(eval_per_data, [(di, data, ps, tmp_eval_results_filename) for di, (data, ps) in enumerate(zip(datasets['test'], parse_results))])
    with open(osp.join(eval_dir, fn), 'w') as f:
        json.dump(eval_results, f)

def main(curdir):
    parse_dir = osp.join(curdir, 'parse_results')
    if not osp.exists(parse_dir):
        print(f'{parse_dir} does not exist')
        return
    eval_dir = osp.join(curdir, 'eval_results_faster_mp')
    os.makedirs(eval_dir, exist_ok=True)

    filenames = os.listdir(parse_dir)
    filenames = [f for f in filenames if f.endswith('.json')]
    random.shuffle(filenames)
    for fn in filenames:
        eval_parse_file(fn, parse_dir, eval_dir)

if __name__ == '__main__':
    curdir = osp.dirname(osp.abspath(__file__))
    curdir = osp.join(curdir, 'gen_tests')
    main(osp.join(curdir, 'gen_asserts'))
    main(osp.join(curdir, 'gen_doctest'))
    main(osp.join(curdir, 'gen_unittest'))

