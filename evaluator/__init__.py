from task import Dataset
import torch
import logging
from time import time
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator(object):
    def __init__(self, model, tokenizer, dataset: Dataset, prompt_template: str, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.prompt_template = prompt_template
        self.verbose = verbose

    def __repr__(self):
        return f"Evaluator(model={self.model}, dataset={self.dataset}, prompt_template={self.prompt_template})"
    
    def evaluate(self, output_info: Optional[Dict] = None):
        """
        Evaluate the model on the dataset.
        """
        output_info["generation_config"] = self.model.generation_config.to_dict()
        for sample_problem, sample_answer in self.dataset:
            start_time = time()
            output_text, score, raw_text_len, gen_length = self.run_single(sample_problem, sample_answer)
            time_taken = time() - start_time
            if output_info is not None:
                assert "instances" in output_info, "output_info must contain 'instances' key."
                output_info["instances"].append({
                    "question": sample_problem,
                    "answer": sample_answer,
                    "output_text": output_text,
                    "score": score,
                    "raw_text_len": raw_text_len,
                    "gen_length": gen_length,
                    "time_taken": time_taken,
                })
            if self.verbose:
                logger.info(f"Score: {score}, Raw text length: {raw_text_len}, Generated length: {gen_length}, time taken: {time_taken:.2f}s")
            break  # Remove this break to evaluate all samples
    
    def run_single(self, question: str, answer: str):
        context = self.prompt_template.format(Question=question)
        input_ids = self.tokenizer.encode(context)
        raw_text_len = len(input_ids)
        context_enc = torch.tensor([input_ids]).to(self.model.device)
        if self.verbose:
            logger.info(f"\nQuestion: {context}\n")
        outputs = self.model.generate(context_enc)
        output_text, gen_length = self.decode(outputs, self.tokenizer, raw_text_len)
        if self.verbose:
            logger.info(f"\nOutput text: {output_text}\n")
        score = self.dataset.is_correct(output_text, answer)
        return output_text, score, raw_text_len, gen_length
    
    def decode(self, outputs, tokenizer, raw_text_len):
        new_tokens = outputs[0, raw_text_len:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_length = len(new_tokens)
        if self.verbose:
            logger.info(f"prompt length: {raw_text_len}, generated {gen_length} tokens for this problem.")
        return out_text.strip(), gen_length