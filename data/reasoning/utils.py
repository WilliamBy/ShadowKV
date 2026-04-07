import os, argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import torch.nn.functional as F


def parse_common_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_root_dir", type=str, required=True, help="Root directory for output files")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")

    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--method", type=str, default="full", 
                        choices=["full", "duo_attn", "razor", 
                                 "quest", "arkv", "spec_ret", "raas"],
                        help="KV cache eviction/selection method")
    parser.add_argument("--page_rep", type=str, default="quest", choices=["quest", "arkv"],
                        help="Page representation method")
    parser.add_argument("--GQA_policy", type=str, default="avgS",
                        choices=["maxQ", "avgQ", "maxS", "avgS", "maxSM", "avgSM"],
                        help="Grouped-query attention aggregation policy")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--max_gen", type=int, default=8192,
                        help="Maximum number of tokens to generate")

    parser.add_argument("--data_from", type=int, default=None,
                        help="Starting index offset for data samples")
    parser.add_argument("--data_idx", type=int, default=None,
                        help="Specific data sample index to evaluate")
    parser.add_argument("--data_idx_to", type=int, default=None,
                        help="End index (exclusive) for data sample range")

    parser.add_argument(
        "--attn_load_dir", type=str, default="manual", help="attention pattern directory"
    )
    parser.add_argument("--sink", type=int, default=512,
                        help="Number of sink (initial) tokens to keep")
    parser.add_argument("--recent", type=int, default=512,
                        help="Number of recent tokens to keep")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Token budget for dynamic KV cache selection")
    parser.add_argument("--page_size", type=int, default=32, 
                        help="For Quest, ArkVale, SpecRet and RaaS")

    parser.add_argument("--sparsity", type=float, default=1, 
                        help="Head-level sparsity, i.e., 1 - (ratio of full heads)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Attention score threshold for classifying full vs. streaming heads")

    parser.add_argument("--raas_alpha", type=float, default=1e-4,
                        help="Alpha parameter for RaaS method")

    parser.add_argument("--spec_ret_steps", type=int, default=1,
                        help="Number of past queries to use for SpecRet")
    parser.add_argument("--last_layer_budget", type=int, default=0,
                        help="Extra token budget for the last layer in SpecRet")
    parser.add_argument("--spec_ret_corr", type=float, default=None,
                        help="Cosine similarity threshold to trigger correction in SpecRet")
    parser.add_argument("--corr_group", type=str, default="avg", choices=["max", "avg"],
                        help="Aggregation method for correction grouping in SpecRet")

    parser.add_argument("--skip_layer", type=int, default=1,
                        help="Number of initial layers to skip (use full attention)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--repeat_bsz", type=int, default=None, 
                        help="set for efficiency eval")

    parser.add_argument("--kv8", action="store_true",
                        help="Use 8-bit quantized KV cache (FP16 instead of BF16)")
    return parser


def get_out_path(args, config, out_root_dir=None, mkdir=True):
    method = args.method
    dataset = args.dataset or ""
    model_name = args.model
    if out_root_dir is None:
        out_root_dir = args.out_root_dir
    if args.data_idx is not None or args.data_idx_to is not None:
        out_root_dir = str(Path(out_root_dir).parent / "res-test")
    out_dir = f"{out_root_dir}/{model_name}-{method}"
    if mkdir:
        os.makedirs(out_dir, exist_ok=True)
    if method in ["duo_attn", "razor"]:
        sparsity = config["sparsity"]
        sink = config["sink"]
        recent = config["recent"]
        out_path = f"{out_dir}/{dataset}-s{sink}-r{recent}-{sparsity:.2f}"
        attn_load = os.path.basename(args.attn_load_dir.rstrip("/"))
        pattern_id = attn_load if os.path.isdir(args.attn_load_dir) else attn_load.split(".")[0]
        if sparsity < 1:
            out_path += f"-{pattern_id}"
    elif method != "full":
        sink = config["sink"]
        recent = config["recent"]
        sparsity = config["sparsity"]
        attn_load = os.path.basename(args.attn_load_dir.rstrip("/"))
        pattern_id = attn_load if os.path.isdir(args.attn_load_dir) else attn_load.split(".")[0]
        out_path = (f"{out_dir}/{dataset}-s{sink}-r{recent}-{sparsity:.2f}-{pattern_id}")
        if method in ["quest", "raas", "arkv", "spec_ret"]:
            budget = config["budget"]
            page_size = config["page_size"]
            out_path += f"-p{page_size}-b{budget}"
            if method == "raas":
                alpha = config["raas_alpha"]
                out_path += f"-a{alpha}"
            else:
                GQA_policy = config["GQA_policy"]
                out_path += f"-{GQA_policy}"
                if method == "spec_ret":
                    spec_ret_steps = config["spec_ret_steps"]
                    llb = config["llb"]
                    correct_sim = config["correct_sim"]
                    corr_group = config["corr_group"]
                    out_path += f"-pQ{spec_ret_steps}-llb{llb}"
                    if correct_sim is not None:
                        out_path += f"-pQcs{correct_sim}"
                    if corr_group != "avg":
                        out_path += f"-cog{corr_group}"
        else:
            assert False and "Not covered method"
    else:
        out_path = f"{out_dir}/{dataset}"
    if args.skip_layer != 0:
        out_path += f"-skl{args.skip_layer}"
    if args.temperature != 0.0:
        out_path += f"-t{args.temperature}"
    if args.top_p != 1.0:
        out_path += f"-topp{args.top_p}"
    if args.max_gen != 8192:
        out_path += f"-Mg{args.max_gen}"
    if args.seed != 42:
        out_path += f"-seed{args.seed}"
    if args.data_from is not None:
        out_path += f"-from{args.data_from}"
    if args.data_idx_to is not None:
        out_path += f"-to{args.data_idx_to}"

    out_path += ".jsonl"
    print("Output to:", out_path)
    return out_path


def build_chat(tokenizer, prompt, model_name, to_token=True):
    if "ds-r1" in model_name or "skywork-or1" in model_name:
        return prompt + "<think>\n"
    elif "llama" in model_name:
        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        if to_token:
            chat_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda" if not use_npu else "npu")
        else:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=to_token, add_generation_prompt=True, return_tensors="pt"
            )
    elif "qwq" in model_name or "qwen" in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "QwQ" in model_name:
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return chat_prompt
