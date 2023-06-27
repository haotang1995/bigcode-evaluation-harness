"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests.
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import re, hashlib

from evaluate import load
from datasets import load_dataset, concatenate_datasets

from lm_eval.base import Task

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

def mbpp_preprocess(example):
    prompt = example["prompt"]
    code = example["code"]
    test_list = example["test_list"]
    test_imports = example["test_imports"]

    func_name_list = [
        c[4:].split('(')[0].strip()
        for c in code.split('\n')
        if 'def ' == c[:4]
    ]
    func_name = [
        fn
        for fn in func_name_list
        if all([fn in test for test in test_list])
    ]
    assert(len(func_name) == 1), f"func_name: {func_name}, func_name_list: {func_name_list}, test_list: {test_list}"
    func_name = func_name[0]

    code_blocks = [[],]
    for c in code.split('\n'):
        if c.startswith('def '):
            code_blocks.append([])
        code_blocks[-1].append(c)
    test_imports += [c for c in code_blocks[0] if c.startswith('import ') or c.startswith('from ')]
    code_blocks = code_blocks[1:]
    assert(all([c[0].startswith('def ') for c in code_blocks]))

    func_def = [c for c in code.split('\n') if f'def {func_name}(' in c or f'def {func_name} (' in c]
    assert(len(func_def) == 1), f"func_def: {func_def}, func_name: {func_name}, code: {code}"
    func_def = func_def[0]

    func_def_index = [cbi for cbi, cb in enumerate(code_blocks) if func_def.strip() == cb[0].strip()]
    assert(len(func_def_index) == 1), f"func_def_index: {func_def_index}, func_def: {func_def}, code_blocks: {code_blocks}"
    func_def_index = func_def_index[0]
    code_blocks = code_blocks[:func_def_index] + code_blocks[func_def_index+1:] + [code_blocks[func_def_index][1:]]

    main_cb_line = code_blocks[-1][0]
    indent = main_cb_line[:len(main_cb_line) - len(main_cb_line.lstrip())]

    assert(all([test.strip().startswith('assert') for test in test_list]))
    running_examples = [test.strip()[6:].strip() for test in test_list]

    new_prompt = '\n'.join(test_imports) + '\n'
    new_prompt += func_def + '\n'
    new_prompt += indent+'"""\n'
    new_prompt += '\n'.join([indent+p for p in prompt.split('\n')]) + '\n'
    new_prompt += '\n'.join([indent+'>>> '+p for p in running_examples]) + '\n'
    new_prompt += indent+'"""\n'

    new_code = '\n'.join(['\n'.join([indent+c for c in cb]) for cb in code_blocks[:-1]])
    new_code += '\n'.join(code_blocks[-1])

    example["prompt"] = new_prompt
    example["code"] = new_code
    example["canonical_solution"] = new_code
    example["entry_point"] = func_name
    assert all([func_name in test for test in example["test_list"]])
    example["test"] = 'def check(candidate)\n' + '\n'.join([indent+test.replace(func_name, 'candidate') for test in example["test_list"]])
    return example


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "EMPTY-NULL"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
        )
        self.dataset = load_dataset(
            'mbpp',
            'sanitized',
            use_auth_token=True,
        )
        self.dataset = self.dataset.map(mbpp_preprocess,)
        self.dataset['test'] = concatenate_datasets([
            self.dataset['train'],
            self.dataset['validation'],
            self.dataset['test'],
            self.dataset['prompt']
        ])

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
