import json
import re
from pathlib import Path
import argparse
import requests
import math
import numpy as np
import tqdm
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from typing import List, Dict, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
python eval/evaluate_chat_gsm8k.py [--use-fewshot]
"""

INVALID_ANS = "[invalid]"
DEVICE = "cuda:0"

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

def load_2024_dataset() -> List[dict]:
    """
    Load the dataset of problems specifically for 2024.
    By default, this loads from 'AI-MO/aimo-validation-aime' and filters for 2024.
    Adjust if you want to pull from your own AIME_2024 dataset or anything else.
    """
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")
    # Filter out problems that are not from 2024
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])
    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert (
        len(dataset) == 30
    ), f"Expected 30 problems from 2024, but found {len(dataset)}"
    return dataset

def doc_to_text(doc, use_fewshot):
    context = MATH_QUERY_TEMPLATE.format(Question=doc["problem"])
    return context

def decode(outputs, tokenizer, raw_text_len):
    new_tokens = outputs[0, raw_text_len:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    gen_length = len(new_tokens)
    logger.info(f"prompt length: {raw_text_len}, generated {gen_length} tokens for this problem.")
    return out_text.strip()

def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc)
    output_text = decode(outputs, tokenizer, raw_text_len)
    print(f"\nOutput text: {output_text}\n")
    return output_text

def extract_answer(response: str) -> Optional[int]:
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
                return int(last_match.group(1))
            except (ValueError, IndexError):
                continue

    numbers = re.findall(r"(\d+)", response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    return None

def is_correct(completion, answer):
    gold = extract_answer(answer)
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

    return number_equal(gold, extract_answer(completion))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        help="Checkpoint path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )
    parser.add_argument("--use-fewshot", action="store_true")

    args = parser.parse_args()

    # if args.sample_input_file is not None:
    #     dataset = load_from_disk(args.sample_input_file)  # or:
    # else:
    #     dataset = load_dataset("gsm8k", "main")
    
    dataset = load_2024_dataset()

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, bf16=True, use_flash_attn=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="auto", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.temperature = 0.6  
    model.generation_config.top_p = 0.95
    model.generation_config.max_new_tokens = 32768

    f_output = open(args.sample_output_file, "w", encoding="utf-8")
    acc_res = []
    for doc in tqdm.tqdm(dataset):
        context = doc_to_text(doc, args.use_fewshot)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        print(acc)
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()
        acc_res.append(acc)

    f_output.close()
    # print("4-shot Acc: " if args.use_fewshot else "Zero-shot Acc", np.mean(acc_res))