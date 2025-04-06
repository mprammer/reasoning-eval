"""Utility functions for the project."""
from evaluator import Evaluator
from task import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from typing import List, Dict, Tuple, Optional, Union
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name: str):
    print(f"Loading model and tokenizer for {model_name} ...")
    if "deepseek-ai" in model_name: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        model.generation_config.temperature = 0.6  
        model.generation_config.top_p = 0.95
        model.generation_config.max_new_tokens = 8192 # 32768
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, bf16=True, use_flash_attn=True
        )
        return model, tokenizer
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
def load_vllm_model(model_name: str):
    print(f"Loading vllm model and tokenizer for {model_name} ...")
    if "deepseek-ai" in model_name: 
        model = LLM(model=model_name, max_model_len=8192)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, bf16=True, use_flash_attn=True
        )
        generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 8192,
            "n": 4,
        }
        return (model, generation_config), tokenizer
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    
def load_dataset(dataset_name):
    """
    Get the dataset for the specified name.
    """
    print(f"Loading dataset and prompt template for {dataset_name} ...")
    if dataset_name == "AIME-2024":
        from task.aime import AIME
        from template import MATH_QUERY_TEMPLATE as template
        dataset = AIME()
        return dataset, template
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
def get_evaluator(model_name: str, dataset_name: str, verbose: bool = False):
    """
    Get the evaluator for the specified model and dataset.
    """

    # Load the model and tokenizer
    model, tokenizer = load_vllm_model(model_name)
    # Load the dataset and prompt template
    dataset, prompt_template = load_dataset(dataset_name)

    # Create the evaluator
    evaluator = Evaluator(model, tokenizer, dataset, prompt_template, verbose=verbose)

    return evaluator
