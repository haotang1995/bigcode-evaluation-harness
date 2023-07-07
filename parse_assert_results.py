#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import ast

curdir = osp.dirname(osp.abspath(__file__))
curdir = osp.join(curdir, 'gen_tests', 'gen_asserts')
parsedir = osp.join(curdir, 'parse_results')
os.makedirs(parsedir, exist_ok=True)

def parse_comparator(comp, entry_point):
    # Translate comparators back to strings
    if not isinstance(comp, ast.Call) or not isinstance(comp.func, ast.Name):
        out = ast.unparse(comp)
        assert isinstance(out, str)
    else:
        assert isinstance(comp.func, ast.Name), (comp.func, ast.unparse(comp))
        if not comp.func.id == entry_point:
            out = {'out': None}
        else:
            assert comp.func.id == entry_point, (l, comp.func.id)
            args2 = comp.args
            # Translate args back to strings
            args2 = tuple(ast.unparse(arg) for arg in args2)
            out = {'args2': args2}
    return out

def parse_line(tree, entry_point,):
    if not isinstance(tree, ast.Assert):
        return None
    l = ast.unparse(tree)
    assert isinstance(tree, ast.Assert)
    tree = tree.test

    if isinstance(tree, ast.Compare):
        assert isinstance(tree, ast.Compare), (l, tree)
        if not isinstance(tree.left, ast.Call):
            return None
        assert isinstance(tree.left, ast.Call), (l, tree.left)
        assert isinstance(tree.left.func, ast.Name)
        if not tree.left.func.id == entry_point:
            return None
        assert tree.left.func.id == entry_point, (l, tree.left.func.id, entry_point)
        args = tree.left.args
        # Translate args back to strings
        args = tuple(ast.unparse(arg) for arg in args)

        if isinstance(tree.ops[0], ast.Eq):
            assert isinstance(tree.ops[0], ast.Eq), (l, tree)
            if len(tree.comparators) == 1:
                out = parse_comparator(tree.comparators[0], entry_point)
            else:
                out = {'comparators': [parse_comparator(comp, entry_point) for comp in tree.comparators]}
        elif isinstance(tree.ops[0], ast.Is):
            assert isinstance(tree.ops[0], ast.Is), (l, tree)
            assert len(tree.comparators) == 1
            assert isinstance(tree.comparators[0], ast.NameConstant)
            out = tree.comparators[0].value
        elif isinstance(tree.ops[0], ast.NotEq):
            out = {'noteq': ast.unparse(tree.comparators[0])}
        elif isinstance(tree.ops[0], ast.Gt):
            assert isinstance(tree.ops[0], ast.Gt), (l, tree)
            if len(tree.comparators) == 1:
                out = parse_comparator(tree.comparators[0], entry_point)
            else:
                out = {'comparators': [parse_comparator(comp, entry_point) for comp in tree.comparators]}
            out = {'gt': out}
        else:
            assert False, (l, tree.ops[0])
    elif isinstance(tree, ast.Call):
        assert isinstance(tree.func, ast.Name)
        if tree.func.id == entry_point:
            args = tree.args
            # Translate args back to strings
            args = tuple(ast.unparse(arg) for arg in args)
            out = True
        elif tree.func.id == 'isinstance':
            assert len(tree.args) == 2
            if isinstance(tree.args[0], ast.Name) and tree.args[0].id == entry_point:
                return None
            assert isinstance(tree.args[0], ast.Call), (l, tree.args[0])
            assert isinstance(tree.args[0].func, ast.Name)
            assert tree.args[0].func.id == entry_point
            args = tree.args[0].args
            # Translate args back to strings
            args = tuple(ast.unparse(arg) for arg in args)
            assert isinstance(tree.args[1], ast.Name)
            out = {'isinstance': ast.unparse(tree.args[1])}
        elif tree.func.id == 'callable' or tree.func.id == 'all':
            return None
        else:
            return None
            assert False, (l, tree.func.id)
    elif isinstance(tree, ast.UnaryOp):
        assert isinstance(tree.op, ast.Not)
        if isinstance(tree.operand, ast.Call):
            assert isinstance(tree.operand.func, ast.Name)
            if tree.operand.func.id != entry_point:
                return None
            assert tree.operand.func.id == entry_point, (l, tree.operand.func.id, entry_point)
            args = tree.operand.args
            # Translate args back to strings
            args = tuple(ast.unparse(arg) for arg in args)
            out = False
        else:
            return None
    elif isinstance(tree, ast.Name):
        # assert tree.id == entry_point, (l, tree.id)
        return None
    elif isinstance(tree, ast.Subscript):
        return None
    elif isinstance(tree, ast.BinOp):
        return None
    elif isinstance(tree, ast.Constant):
        return None
    else:
        assert False, (l, tree)

    return {'args': args, 'out': out, 'line': l,}

def get_entry_point(prompt):
    prompt = prompt[prompt.rfind('\ndef ') + 4:].strip()
    entry_point = prompt[:prompt.find('(')].strip()
    assert entry_point, prompt
    return entry_point

def split_gen(g):
    assert('pass' in g), g
    g = g.split('\n')
    sg = [gg.strip() for gg in g]
    # index = len(sg) - 1 - sg[::-1].index('pass')
    index = sg.index('pass')
    assert index != len(sg), g
    prompt = '\n'.join(sg[:index+1])
    gen = '\n'.join(sg[index+1:])
    return prompt, gen

def parse_gen(g):
    # print('---')
    # print(g)
    prompt, g = split_gen(g)
    entry_point = get_entry_point(prompt)

    g = g.split('\n')

    # Remove blocked comments surrounded by """ """
    tmp_g = []
    comment_start = False
    for gg in g:
        if gg.strip() == '"""':
            comment_start = not comment_start
            continue
        if comment_start:
            continue
        tmp_g.append(gg)
    g = tmp_g

    # Remove blocked comments surrounded by ''' '''
    tmp_g = []
    comment_start = False
    for gg in g:
        if gg.strip() == "'''":
            comment_start = not comment_start
            continue
        if comment_start:
            continue
        tmp_g.append(gg)
    g = tmp_g

    # Remove any lines that are empty or start with # or print
    g = [gg for gg in g if gg.strip() and not gg.strip().startswith('#') and not gg.strip().startswith('print') and not gg.strip().startswith('"""')]

    # Remove any lines after import statements
    for i, gg in enumerate(g):
        if gg.strip().startswith('import') or gg.strip().startswith('from') or gg.strip().startswith('def') or gg.strip().startswith('class') or gg.strip().startswith('help'):
            g = g[:i]
            break

    asserts = '\n'.join(g)
    syntax_error = True
    while syntax_error:
        try:
            trees = ast.parse(asserts)
            syntax_error = False
        except SyntaxError:
            asserts = asserts[:asserts.rfind('assert')]

    # assert len(trees.body) > 0, (g, trees.body)

    new_gen = []
    for tree in trees.body:
        new_gg = parse_line(tree, entry_point)
        if new_gg:
            new_gen.append(new_gg)
        else:
            break
    return new_gen

def parse(fn):
    assert fn.endswith('.json')
    parsed_fn = osp.join(parsedir, fn)
    if osp.exists(parsed_fn):
        return
    print('Parsing %s' % fn)
    with open(osp.join(curdir, fn), 'r') as f:
        gen = json.load(f)
    gen = [[parse_gen(gg) for gg in g] for g in gen]
    with open(parsed_fn, 'w') as f:
        json.dump(gen, f, indent=4)

def main():
    filenames = os.listdir(curdir)
    filenames = [fn for fn in filenames if fn.endswith('.json')]
    for fn in filenames:
        parse(fn)

if __name__ == '__main__':
    main()
