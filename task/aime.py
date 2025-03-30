from datasets import load_from_disk, load_dataset
from typing import List, Dict, Tuple, Union, Optional
from .base import Dataset
import re, math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIME(Dataset):
    def __init__(self, name: str = "AI-MO/aimo-validation-aime", split="train"):
        super().__init__(name, split)
        """
        Load the dataset of problems specifically for 2024.
        By default, this loads from 'AI-MO/aimo-validation-aime' and filters for 2024.
        Adjust if you want to pull from your own AIME_2024 dataset or anything else.
        """
        dataset_original = load_dataset(self.dataset_name)
        # Filter out problems that are not from 2024
        dataset = dataset_original[split].filter(lambda example: "2024" in example["url"])
        logging.debug(f"Filtered dataset size: {len(dataset)}.")
        assert (
            len(dataset) == 30
        ), f"Expected 30 problems from 2024, but found {len(dataset)}"
        self.data = dataset
        
    def __iter__(self):
        """
        Iterate through the dataset.
        """
        for example in self.data:
            yield example["problem"], example["answer"]
    
    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)
        
    def extract_answer(self, response: str) -> Optional[int]:
        """
        Extract the numerical answer from the LLM's solution text.
        Looks for \\boxed{} or other final-answer patterns. Returns the *last* match found.
        """
        if not response:
            return None

        response = " ".join(response.split())  # remove extra whitespace

        patterns = [
            r"\$n=\\boxed{(\d+)}\$",
            r"\\\[\\boxed{(\d+)}\\\]",
            r"\\\[\\boxed{(\d+)}\.\\\]",
            r"\\boxed{(\d+)}",
            r"\$\\boxed{(\d+)}\$",
            r"boxed{(\d+)}",
            r"\\boxed\s*{\s*(\d+)\s*}",
            r"\bboxed\s*{\s*(\d+)\s*}",
            r"final answer is[^\d]*(\d+)",
            r"answer is[^\d]*(\d+)",
            r"answer:[^\d]*(\d+)",
            r"= ?(\d+)$",
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                try:
                    return last_match.group(1)
                except (ValueError, IndexError):
                    continue

        numbers = re.findall(r"(\d+)", response)
        if numbers:
            try:
                return numbers[-1]
            except ValueError:
                pass

        return None
    
    def is_correct(self, completion, answer):
        '''
        Check if the completion is correct based on the answer.
        :param completion: The generated answer from the model.
        :param answer: The ground truth answer.
        :return: True if the completion is correct, False otherwise.
        '''
        gold = self.extract_answer(answer)
        assert gold is not None, "No ground truth answer found in the document."

        def number_equal(answer, pred):
            if pred is None:
                return False
            try:
                return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
            except:
                print(
                    f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
                )
                return False

        return number_equal(gold, self.extract_answer(completion))