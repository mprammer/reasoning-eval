# reasoning-eval

Usage:

```python
python lm-eval.py --model-path [model path on HF] --dataset-name [dataset name]
```

Supported models:
- models under deepseek-ai (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

Supported datasets:
- AIME-2024

To add a model please add the corresponding generation config under the load_model function in file `utils.py`, for instance:
```python
if "deepseek-ai" in model_name: 
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
    model.generation_config.temperature = 0.6  
    model.generation_config.top_p = 0.95
    model.generation_config.max_new_tokens = 32768
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, bf16=True, use_flash_attn=True
    )
    return model, tokenizer
```

To add a customized dataset please add the corresponding dataset class in folder `task`.

To add a customized prompt please add the template in `template.py` and also update the `load_dataset` function under `utils.py`. 