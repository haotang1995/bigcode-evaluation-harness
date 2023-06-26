from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, mbpp, multiple, instruct_humaneval,
               humaneval_postprompt,
               humaneval_simple_feedback,
               humaneval_simple_feedback_cf,
               humaneval_pyflakes_feedback,
               humaneval_pyflakes_feedback_cf,
               humaneval_cv,
               humaneval_gen_assert,
               humaneval_gen_assert_cmt,
               humaneval_gen_unittest,
               humaneval_gen_doctest,
               humaneval_git_commit,
               humaneval_simple_feedback_git,
               )

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    "humaneval": humaneval.HumanEval,
    "mbpp": mbpp.MBPP,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),

    "humaneval_postprompt": humaneval_postprompt.HumanEval,
    "humaneval_simple_feedback": humaneval_simple_feedback.HumanEval,
    "humaneval_simple_feedback_cf": humaneval_simple_feedback_cf.HumanEval,
    "humaneval_pyflakes_feedback": humaneval_pyflakes_feedback.HumanEval,
    "humaneval_pyflakes_feedback_cf": humaneval_pyflakes_feedback_cf.HumanEval,
    "humaneval_cv": humaneval_cv.HumanEval,
    "humaneval_gen_assert": humaneval_gen_assert.HumanEval,
    "humaneval_gen_assert_cmt": humaneval_gen_assert_cmt.HumanEval,
    "humaneval_gen_unittest": humaneval_gen_unittest.HumanEval,
    "humaneval_gen_doctest": humaneval_gen_doctest.HumanEval,
    "humaneval_git_commit": humaneval_git_commit.HumanEval,
    "humaneval_simple_feedback_git": humaneval_simple_feedback_git.HumanEval,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
