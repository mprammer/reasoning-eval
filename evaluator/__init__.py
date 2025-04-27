from task import Dataset
import torch
import logging
from time import time
from typing import Dict, Optional
from vllm import SamplingParams
import json
import os

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
        for idx, (sample_problem, sample_answer) in enumerate(self.dataset):
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
            logger.info(f"{idx+1}/{len(self.dataset)} Score: {score}, Raw text length: {raw_text_len}, Generated length: {gen_length}, time taken: {time_taken:.2f}s")
    
    def evaluate_vllm(self, output_info: Optional[Dict] = None, sample_output_file: str = None):
        """
        Evaluate the model on the dataset.
        """
        start_idx = 0
        output_info["generation_config"] = self.model[1]
        if os.path.isfile(sample_output_file):
            logger.info(f"Loading existing output info from {sample_output_file}")
            with open(sample_output_file, "r") as f:
                output_info = json.load(f)
                start_idx = len(output_info["instances"])
        for idx, (sample_problem, sample_answer) in enumerate(self.dataset):
            if idx < start_idx:
                continue
            start_time = time()
            context, output_text, score, raw_text_len, gen_length = self.run_vllm_single(sample_problem, sample_answer)
            time_taken = time() - start_time
            if output_info is not None:
                assert "instances" in output_info, "output_info must contain 'instances' key."
                output_info["instances"].append({
                    "question": sample_problem,
                    "context": context,
                    "answer": sample_answer,
                    "output_text": output_text,
                    "score": score,
                    "raw_text_len": raw_text_len,
                    "gen_length": gen_length,
                    "time_taken": time_taken,
                })
            logger.info(f"{idx+1}/{len(self.dataset)} Score: {score}, Raw text length: {raw_text_len}, Generated length: {gen_length}, time taken: {time_taken:.2f}s")
            with open(sample_output_file, "w") as f:
                json.dump(output_info, f, indent=4)
        return output_info
    
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
    
    def run_vllm_single(self, question: str, answer: str):
        output_text = []
        scores = []
        gen_length = []
        context = self.prompt_template.format(Question=question)
        sampling_params = SamplingParams(**self.model[1])
        outputs = self.model[0].generate([context], sampling_params)[0]
        input_ids = self.tokenizer.encode(context)
        raw_input_len = len(input_ids)
        prompt = outputs.prompt
        if self.verbose:
            print(f"Prompt:    {prompt!r}")
        for idx, output in enumerate(outputs.outputs):
            generated_text = output.text
            if self.verbose:
                print(f"Output[{idx}]:    {generated_text!r}")
                print("-" * 60)
            output_ids = self.tokenizer.encode(generated_text)
            raw_output_len = len(output_ids)
            score = self.dataset.is_correct(generated_text, answer)
            output_text.append(generated_text.strip())
            gen_length.append(raw_output_len)
            scores.append(score)
        return context, output_text, scores, raw_input_len, gen_length
    
    def decode(self, outputs, tokenizer, raw_text_len):
        new_tokens = outputs[0, raw_text_len:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_length = len(new_tokens)
        if self.verbose:
            logger.info(f"prompt length: {raw_text_len}, generated {gen_length} tokens for this problem.")
        return out_text.strip(), gen_length