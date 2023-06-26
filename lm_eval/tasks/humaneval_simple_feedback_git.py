#@Hao @Jun 12, 2023
#Just add comment mark before the feedback
"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests.
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import json
import re
import copy

from evaluate import load

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

class HumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nFeedback", '\n"""',],
            requires_execution=True,
        )

    def get_dataset(self, previous_generation_path=[]):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        self.previous_generation_path = previous_generation_path
        if len(previous_generation_path) == 0:
            return self.dataset['test']

        previous_generations = [] # [round_num, doc_num, n_samples]
        previous_n_samples = []
        for gen_path in previous_generation_path:
            with open(gen_path, "r") as f:
                gen = json.load(f)
                assert len(self.dataset['test']) == len(gen), "Previous generation must have the same length as the dataset"
            n_samples_list = [len(ggen) for ggen in gen]
            assert len(set(n_samples_list)) == 1, "All previous generations must have the same number of samples"
            assert len(previous_n_samples) == 0 or previous_n_samples[-1] % n_samples_list[0] == 0, "The number of samples in the previous generation must be a multiple of the number of samples in the current generation"
            previous_n_samples.append(n_samples_list[0])
            previous_generations.append(gen)

        self.num_of_rounds = len(previous_generations)
        self.previous_n_samples = previous_n_samples
        self.n_samples = previous_n_samples[-1]

        dataset = []
        for doc_num in range(len(self.dataset["test"])):
            doc = self.dataset["test"][doc_num]
            for sample_num in range(self.n_samples):
                sample = copy.deepcopy(doc)
                for round_num in range(len(previous_generations)):
                    sample[f"prv_gen_round_{round_num}"] = previous_generations[round_num][doc_num][sample_num//self.previous_n_samples[round_num]]
                dataset.append(sample)
        assert(len(dataset) == len(self.dataset["test"]) * self.n_samples)

        self.processed_dataset = dataset
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        original_prompt = doc["prompt"].strip()
        prompt = ''
        for round_num in range(self.num_of_rounds):
            previous_generation = doc[f"prv_gen_round_{round_num}"]
            previous_generation = previous_generation.strip()
            prompt += '<commit_before>'+previous_generation + '<commit_msg>Feedback: The code above is incorrect. Please fix it.'
        prompt += '<commit_after>'+original_prompt
        return prompt

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
        prompt = self.get_prompt(self.processed_dataset[idx])
        generation = generation[len(prompt) :]
        prompt = prompt[prompt.rfind('# Feedback'):].strip()
        prompt = prompt[prompt.find('\n') + 1 :].strip()
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def postprocess(self, generations, references):
        """Defines the postprocessing for a list of LM generations and their references.
        :param generations: list(str)
        :param references: list(str)
        """
        assert len(generations) == len(references), "Generations and references must have the same length"
        assert len(generations) % self.n_samples == 0, f"Generations must have a length that is a multiple of n_samples, but got {len(generations)} and n_samples={self.n_samples}"
        # generations of shape [len(dataset) * n_samples, cur_n_samples].
        # change its shape to [len(dataset), n_samples * cur_n_samples]
        generations = [[
            gen
            for si in range(self.n_samples)
            for gen in generations[di * self.n_samples + si]
        ] for di in range(len(generations)//self.n_samples)]
        references = [[
            references[di * self.n_samples + si]
            for si in range(self.n_samples)
        ] for di in range(len(references)//self.n_samples)]
        for ref in references:
            assert len(set(ref)) == 1, "All samples in a doc must have the same reference"
        references = [ref[0] for ref in references]
        return generations, references

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
