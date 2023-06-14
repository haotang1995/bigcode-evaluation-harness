import json
import os, os.path as osp
import pyflakes.api
import pyflakes.reporter

FalsePositiveErrors = ['is assigned to but never used', 'imported but unused',]

class StringReporter(pyflakes.reporter.Reporter):
    def __init__(self):
        self.messages = []

    def unexpectedError(self, filename, msg):
        self.messages.append(str(msg))

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.messages.append("Syntax Error: {0}".format(msg))

    def flake(self, message):
        self.messages.append(str(message))

def check_code(code):
    reporter = StringReporter()
    pyflakes.api.check(code, 'input', reporter)
    errors = reporter.messages
    errors = [e for e in errors if not any([f in e for f in FalsePositiveErrors])]
    return errors

def main():
    curdir = osp.dirname(osp.abspath(__file__))
    generation_dir = osp.join(curdir, 'generation_results')
    eval_dir = osp.join(generation_dir, 'check_syntax_results')
    if not osp.exists(eval_dir):
        os.makedirs(eval_dir)
    eval_filename_list = os.listdir(eval_dir)
    for fn in os.listdir(generation_dir):
        if fn.endswith('.json') and (fn not in eval_filename_list or osp.getctime(osp.join(generation_dir, fn)) > osp.getctime(osp.join(eval_dir, fn))):
            print('Evaluating {}'.format(fn))
            with open(osp.join(generation_dir, fn), 'r') as f:
                gen = json.load(f)
            # Remove '\nAnswer: ' from generations
            gen = [[g.replace('\nAnswer: ', '') for g in gg] for gg in gen]
            errors = [[check_code(gg) for gg in g] for g in gen]
            with open(osp.join(eval_dir, fn), 'w') as f:
                json.dump(errors, f)

if __name__ == '__main__':
    main()
