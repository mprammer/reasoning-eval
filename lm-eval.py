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
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        help="Dataset name",
        default="AIME-2024",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="aime.json"
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
        verbose=True,
    )
    
    evaluator.evaluate_vllm(output_info=output_info)
    
    with open(args.sample_output_file, "w") as f:
        json.dump(output_info, f, indent=4)
    
    