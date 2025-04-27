import json
from pathlib import Path
import argparse
from utils import get_evaluator


"""
python lm-eval.py --model-path [model path on HF] --dataset-name [dataset name]

Supported models:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

Supported datasets:
- AIME-2024
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lm-eval args.")
    parser.add_argument(
        "-c",
        "--model-path",
        type=str,
        help="Checkpoint path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        help="Dataset name",
        default="AIME-2024",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="qwen-14b-aime.json"
    )

    args = parser.parse_args()
    
    output_info = {
        "instances": [],
        "model_name": args.model_path,
        "dataset_name": args.dataset_name,
    }
    
    evaluator = get_evaluator(
        model_name=args.model_path,
        dataset_name=args.dataset_name,
        tensor_parallel_size=4,
        verbose=True,
    )
    
    evaluator.evaluate_vllm(output_info=output_info, sample_output_file=args.sample_output_file)
    
    