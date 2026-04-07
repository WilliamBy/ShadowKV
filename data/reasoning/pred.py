import os
import time
import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import random


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models import choose_model_class
from models.base import LLM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_common_args(parser):
    """Parse common arguments for model loading and evaluation."""
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--max_gen", type=int, default=2048, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument("--data_from", type=int, default=None, help="Start index for data")
    parser.add_argument("--data_idx", type=int, default=None, help="Specific data index")
    parser.add_argument("--data_idx_to", type=int, default=None, help="End index for data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--attn_mode", type=str, default="full", help="Attention mode")
    parser.add_argument("--sparse_budget", type=int, default=4096, help="Sparse budget for KV cache")
    parser.add_argument("--rank", type=int, default=32, help="Rank for compression")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size")
    parser.add_argument("--output_dir", type=str, default="archive/reasoning", help="Output directory")
    return parser


def build_chat(tokenizer, prompt, model_name):
    """Build chat prompt for specific models."""
    if "llama-2" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "llama-3" in model_name.lower():
        # Llama-3 uses chat template by default
        pass
    return prompt


def load_model_and_tokenizer(model_path, args):
    """Load model and tokenizer using project's LLM interface."""
    tokenizer = None  # Will be set by LLM class
    step_updater = None  # Not used in LLM interface
    eos_token_ids = None  # Will be set by LLM class
    config = {"model_path": model_path}
    
    # Load LLM implementation
    LLMClass = choose_model_class(model_path)
    print(f"Loading model: {model_path}")
    print(f"Using LLM class: {LLMClass.__name__}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LLMClass(
        model_name=model_path,
        batch_size=args.batch_size,
        device=device,
        max_length=128000,  # Large enough for reasoning tasks
        attn_mode=args.attn_mode,
        dtype=torch.bfloat16,
        sparse_budget=args.sparse_budget,
        rank=args.rank,
        chunk_size=args.chunk_size,
    )
    
    # Extract tokenizer and eos_token_ids from model
    tokenizer = model.tokenizer
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    
    # Add special EOS tokens for specific models
    if "llama" in model_path.lower():
        eos_token_ids.extend([tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    elif "glm" in model_path.lower():
        eos_token_ids.extend([151329, 151336, 151338])
    elif "phi" in model_path.lower():
        eos_token_ids.extend([tokenizer.convert_tokens_to_ids("<|end|>")])
    
    return model, tokenizer, step_updater, eos_token_ids, config


def get_out_path(args, config):
    """Generate output path for predictions."""
    model_name = args.model_name.split('/')[-1]
    method_name = args.attn_mode
    os.makedirs(args.output_dir, exist_ok=True)
    
    filename = f"{args.dataset}-{model_name}-{method_name}.jsonl"
    out_path = os.path.join(args.output_dir, filename)
    
    # Clear the file if it exists
    open(out_path, 'w').close()
    
    return out_path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    return parser.parse_args(args)


def get_pred(
    model: LLM,
    tokenizer,
    eos_token_ids,
    data,
    answer_field_id,
    max_length,
    max_gen,
    prompt_format,
    model_name,
    temperature,
    top_p,
    step_updater,
    out_path
):
    """
    Generate predictions using LLM interface (single sample at a time).
    
    Note: step_updater parameter is kept for compatibility but not used.
    """
    preds = []
    for di, json_obj in enumerate(tqdm(data)):
        prompt = prompt_format.format(**json_obj)
        
        # Tokenize and check length
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            print(f"Warning: prompt length {len(tokenized_prompt)} exceeds max_length {max_length}")
            # Truncate to fit
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        # Apply chat template if needed
        chat_prompt = build_chat(tokenizer, prompt, model_name)
        input_ids = tokenizer.encode(chat_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        # Generate using LLM interface
        outputs = model.generate(
            input_ids=input_ids,
            gen_len=max_gen,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
            benchmark=False
        )
        
        pred = outputs[0] if isinstance(outputs, list) else outputs
        
        pred_entry = {
            "qid": di + (args.data_from or 0),
            "input": prompt,
            "pred": pred,
            "answer": json_obj[answer_field_id],
            "input_len": input_ids.shape[1],
            "output_len": len(tokenizer.encode(pred)),
        }
        
        preds.append(pred_entry)
        
        # Write to file incrementally
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(pred_entry, f, ensure_ascii=False)
            f.write("\n")
    
    return preds


def get_pred_batched(
    model: LLM,
    tokenizer,
    eos_token_ids,
    data,
    answer_field_id,
    max_gen,
    prompt_format,
    model_name,
    temperature,
    top_p,
    step_updater,
    out_path,
    batch_size,
):
    """
    Generate predictions using LLM interface with batching.
    
    Note: step_updater parameter is kept for compatibility but not used.
    """
    preds_all = []
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_json_objs = data.select(range(i, min(i+batch_size, len(data))))
        current_batch_size = len(batch_json_objs)
        
        # Format prompts
        batch_prompts_text = [prompt_format.format(**json_obj) for json_obj in batch_json_objs]
        batch_chat_prompts_text = [
            build_chat(tokenizer, p_text, model_name) for p_text in batch_prompts_text
        ]
        
        # Tokenize all prompts
        input_ids_list = []
        max_len = 0
        for chat_prompt in batch_chat_prompts_text:
            tokens = tokenizer.encode(chat_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids_list.append(tokens)
            max_len = max(max_len, tokens.shape[1])
        
        # Pad to same length
        padded_inputs = []
        for tokens in input_ids_list:
            if tokens.shape[1] < max_len:
                pad_len = max_len - tokens.shape[1]
                padded = torch.nn.functional.pad(tokens, (0, pad_len), value=tokenizer.pad_token_id or tokenizer.eos_token_id)
                padded_inputs.append(padded)
            else:
                padded_inputs.append(tokens)
        
        # Stack into batch
        batch_input_ids = torch.cat(padded_inputs, dim=0).to(model.device)
        
        # Generate using LLM batch_generate
        outputs = model.batch_generate(
            input_ids=batch_input_ids,
            gen_len=max_gen,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
            benchmark=False
        )
        
        # Process outputs
        for k in range(current_batch_size):
            json_obj = batch_json_objs[k]
            original_prompt_text = batch_prompts_text[k]
            chat_prompt_text = batch_chat_prompts_text[k]
            pred_text = outputs[k] if isinstance(outputs, list) else outputs
            
            pred_entry = {
                "qid": i + k,
                "input": original_prompt_text,
                "chat_input": chat_prompt_text,
                "pred": pred_text,
                "answer": json_obj[answer_field_id],
                "input_len": input_ids_list[k].shape[1],
                "output_len": len(tokenizer.encode(pred_text)),
            }
            preds_all.append(pred_entry)
            
            # Write to file incrementally
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(pred_entry, f, ensure_ascii=False)
                f.write("\n")
    
    return preds_all


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    _config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
    model2path = json.load(open(os.path.join(_config_dir, 'model2path.json'), "r"))
    model2maxlen = json.load(open(os.path.join(_config_dir, 'model2maxlen.json'), "r"))
    
    model_name = args.model_name
    assert model_name in model2path, f"Model {model_name} not found in model2path.json"
    
    # Load model using project's LLM interface
    model, tokenizer, step_updater, eos_token_ids, config = load_model_and_tokenizer(
        model2path[model_name], args
    )
    
    max_length = model2maxlen[model_name.split('/')[-1]]
    max_gen = args.max_gen
    
    dataset = args.dataset
    ds_dir = "data/reasoning/datasets"
    answer_field_id = "answer"
    
    if dataset == "AIME":
        ds_path = f"{ds_dir}/aime.jsonl"
    elif dataset == "AIME24":
        ds_path = f"{ds_dir}/aime24.jsonl"
    elif dataset == "GPQAd":
        ds_path = f"{ds_dir}/gpqa_diamond.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "GPQAm":
        ds_path = f"{ds_dir}/gpqa_main.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "GPQA50":
        ds_path = f"{ds_dir}/gpqa50.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "GPQA50c":
        ds_path = f"{ds_dir}/gpqa50c.jsonl"
        answer_field_id = "Correct Answer"
    elif dataset == "MATH500":
        ds_path = f"{ds_dir}/math500.jsonl"
    elif dataset == "MATH50":
        ds_path = f"{ds_dir}/math50.jsonl"
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    data = load_dataset("json", data_files=ds_path, split="train")
    
    dataset2prompt = json.load(open(os.path.join(_config_dir, "dataset2prompt.json"), "r"))
    prompt_format = dataset2prompt[dataset]
    
    # Add chain-of-thought template if specified
    if "cot" in model_name:
        prompt_format += "<Thought> {thought} </Thought>\n"
    
    # Filter data based on arguments
    if args.data_idx is not None:
        data = data.select(range(args.data_idx, args.data_idx+1))
    elif args.data_idx_to is not None:
        data = data.select(range(0, args.data_idx_to))
    elif args.data_from is not None:
        data = data.select(range(args.data_from, len(data)))
    
    out_path = get_out_path(args, config)
    
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Max length: {max_length}")
    print(f"Max generation: {max_gen}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output path: {out_path}")
    print(f"Attn mode: {args.attn_mode}")
    print("-" * 80)
    
    if args.batch_size == 1:
        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            answer_field_id,
            max_length,
            max_gen,
            prompt_format,
            model_name,
            args.temperature,
            args.top_p,
            step_updater,
            out_path
        )
    else:
        preds = get_pred_batched(
            model,
            tokenizer,
            eos_token_ids,
            data,
            answer_field_id,
            max_gen,
            prompt_format,
            model_name,
            args.temperature,
            args.top_p,
            step_updater,
            out_path,
            args.batch_size,
        )
    
    print(f"\nCompleted! Processed {len(preds)} samples.")
    print(f"Results saved to: {out_path}")
