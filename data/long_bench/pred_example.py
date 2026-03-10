import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
from argparse import ArgumentParser
from torch import distributed as dist
import datetime

# attach shadowkv package
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)


def parse_args():
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str,
                   default="gradientai/Llama-3-8B-Instruct-Gradient-1048k")
    p.add_argument("--dataset_name", type=str_to_list,
                   default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=128 *
                   1024, help="The length of the context.")
    p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--minference", action='store_true', default=False)
    p.add_argument("-e", action='store_true', default=False)

    return p.parse_args()


class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process


def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=60*90))
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        master_process = (
            rank == 0
        )
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(f"[Dist init] world_size={world_size}", 'cyan')

    return DistConfig(is_distributed, rank, world_size, device, master_process)
# This is the customized building prompt for chat models


def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    return response


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
):
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            print(f"Truncate prompt to fit max_length={max_length}")
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=False)

        with torch.no_grad():
            pred = model.generate(input_ids.to(
                model.device), gen_len=max_gen, verbose=False, top_p=1.0, temperature=0.0)[0]

        pred = post_process(pred, model_name)
        print(f"Prediction: {pred}")
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name, dist_config, dtype, args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )

    eos_token_ids = tokenizer.eos_token_id

    # load LLM implementation
    from models import choose_model_class
    LLM = choose_model_class(model_name)
    print("This evaluator only support batch_size=1 for now !!!")
    model = LLM(model_name=model_name, batch_size=1, device=dist_config.device, max_length=args.datalen+2048, attn_mode=args.method,
                dtype=dtype, sparse_budget=args.sparse_budget, rank=args.rank, chunk_size=args.chunk_size, minference=args.minference)

    return model, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("data/long_bench/config/model2path.json", "r"))
    model2maxlen = json.load(
        open("data/long_bench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model_name
    # distributed config
    dist_config = init_dist()
    # define your model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model_name, dist_config, torch.bfloat16, args)

    max_length = model2maxlen[model_name.split('/')[-1]]
    if args.datalen > 0:
        max_length = min(args.datalen, max_length)
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [ds.split('/')[-1] for ds in args.dataset_name]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(
        open("data/long_bench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(
        open("data/long_bench/config/dataset2maxlen.json", "r"))
    for dataset in datasets:
        data = load_dataset("THUDM/LongBench", dataset, split="test")
        out_path = f"archive/{args.model_name.split('/')[-1]}/{dataset}_{args.datalen}_{args.method}_{args.sparse_budget}_{args.rank}_{args.chunk_size}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")


def long_bench_score(task, predictions, answers, all_classes):
    dataset2metric = json.load(
        open("data/long_bench/dataset2metric.json", "r"))
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        prediction = (
            prediction.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        if task in ["multifieldqa_zh", "dureader"]:
            prediction = prediction.split("问题：")[0].strip()
        if task in ["lsht"]:
            prediction = prediction.split("新闻内容：")[0].strip()
        if task in ["passage_retrieval_zh"]:
            prediction = prediction.split("请问")[0].split("提示")[0].strip()
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[task](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)
