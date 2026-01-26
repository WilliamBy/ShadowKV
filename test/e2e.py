################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from argparse import ArgumentParser, Namespace
from termcolor import colored
import time
import gc
import torch
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from models import choose_model_class
from data.dataset import Dataset

os.chdir(root_dir)


dataset_name = "ruler/qa_2"

configs = {
    # 1.7% sparsity
    "gradientai/Llama-3-8B-Instruct-Gradient-1048k": {
        "12k": {
            "sparse_budget": 256,
            "min_prompt_len": 1024*12,
            "baseline_bsz": 1,
            "shadowkv_bsz": 1,
        },
        "28k": {
            "sparse_budget": 512,
            "min_prompt_len": 1024*28,
            "baseline_bsz": 1,
            "shadowkv_bsz": 1,
        },
        "44k": {
            "sparse_budget": 768,
            "min_prompt_len": 1024*44,
            "baseline_bsz": 1,
            "shadowkv_bsz": 1,
        },
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 1,
            "shadowkv_bsz": 1,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "01-ai/Yi-9B-200K": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 10,
            "shadowkv_bsz": 42,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 5,
            "shadowkv_bsz": 21,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 10,
        }
    },
    "THUDM/glm-4-9b-chat-1m": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 12,
            "shadowkv_bsz": 50,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 6,
            "shadowkv_bsz": 25,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 3,
            "shadowkv_bsz": 12,
        }
    }
}


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", choices=[
                   "gradientai/Llama-3-8B-Instruct-Gradient-1048k", "meta-llama/Meta-Llama-3.1-8B-Instruct", "01-ai/Yi-9B-200K", "THUDM/glm-4-9b-chat-1m"])
    p.add_argument("--datalen", type=str, default="122k",
                   choices=["12k", "44k", "28k", "60k", "122k", "244k"])
    p.add_argument("--device", type=int, default="0", help="gpu device to use")

    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    model_name = args.model_name
    length = args.datalen
    device = args.device

    min_prompt_len = configs[model_name][length]["min_prompt_len"]
    temperature = 0.6
    baseline_bsz = configs[model_name][length]["baseline_bsz"]
    shadowkv_bsz = configs[model_name][length]["shadowkv_bsz"]
    sparse_budget = configs[model_name][length]["sparse_budget"]

    ##################### Baseline #####################
    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_name, device=f'cuda:{device}',
              max_length=min_prompt_len, attn_mode='full', sparse_budget=sparse_budget)
    torch.cuda.synchronize(llm.device)
    mem_after_load_baseline = torch.cuda.memory_allocated(
        llm.device) / (1024 ** 3)
    dataset = Dataset(dataset_name, llm.tokenizer, 256*1024, 20)

    input_ids = torch.cat([dataset[i][0][:, :min_prompt_len]
                          for i in range(llm.batch_size)], dim=0)

    assert input_ids.shape[-1] == min_prompt_len

    # Reset CUDA peak memory stats before baseline generation
    torch.cuda.reset_peak_memory_stats(llm.device)
    start_time_baseline = time.time()
    _, throughput_baseline, generated_tokens_baseline = llm.batch_generate(
        input_ids.to(llm.device), gen_len=100, benchmark=True, temperature=temperature)
    end_time_baseline = time.time()
    torch.cuda.synchronize(llm.device)
    peak_mem_baseline = torch.cuda.max_memory_allocated(
        llm.device) / (1024 ** 3)
    mem_increment_baseline = peak_mem_baseline - mem_after_load_baseline
    input_tokens_baseline = input_ids.shape[0] * input_ids.shape[1]
    total_tokens_baseline = input_tokens_baseline + generated_tokens_baseline
    mem_per_batch_baseline = mem_increment_baseline / baseline_bsz
    mem_per_token_baseline = mem_increment_baseline / total_tokens_baseline
    print(colored(
        f"[Baseline] Memory after model loading on {llm.device}: {mem_after_load_baseline:.2f} GB", 'cyan'))
    print(
        colored(f"[Baseline] Throughput: {throughput_baseline} tokens/s", 'red'))
    print(colored(
        f"[Baseline] Peak CUDA memory on {llm.device}: {peak_mem_baseline:.2f} GB", 'yellow'))
    print(colored(
        f"[Baseline] Memory increment: {mem_increment_baseline:.2f} GB", 'yellow'))
    print(colored(
        f"[Baseline] Memory per batch: {mem_per_batch_baseline:.2f} GB", 'yellow'))
    print(colored(
        f"[Baseline] Memory per token: {mem_per_token_baseline:.4f} GB", 'yellow'))
    print(colored(
        f"[Baseline] Elapsed time: {end_time_baseline - start_time_baseline:.3f} s", 'yellow'))

    del llm.kv_cache
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    ##################### ShadowKV #####################

    llm = LLM(model_name=model_name, device=f'cuda:{device}',  batch_size=shadowkv_bsz,
              max_length=min_prompt_len, attn_mode='shadowkv_cpu', sparse_budget=sparse_budget)
    torch.cuda.synchronize(llm.device)
    mem_after_load_shadowkv = torch.cuda.memory_allocated(
        llm.device) / (1024 ** 3)
    dataset = Dataset(dataset_name, llm.tokenizer, 256*1024, 100)

    input_ids = torch.cat([dataset[i][0][:, :min_prompt_len]
                          for i in range(llm.batch_size)], dim=0)
    # Reset CUDA peak memory stats before shadowkv generation
    torch.cuda.reset_peak_memory_stats(llm.device)
    start_time_shadowkv = time.time()
    _, throughput_shadowkv, generated_tokens_shadowkv = llm.batch_generate(
        input_ids.to(llm.device), gen_len=100, benchmark=True, temperature=temperature)
    end_time_shadowkv = time.time()
    torch.cuda.synchronize(llm.device)
    peak_mem_shadowkv = torch.cuda.max_memory_allocated(
        llm.device) / (1024 ** 3)
    mem_increment_shadowkv = peak_mem_shadowkv - mem_after_load_shadowkv
    input_tokens_shadowkv = input_ids.shape[0] * input_ids.shape[1]
    total_tokens_shadowkv = input_tokens_shadowkv + generated_tokens_shadowkv
    mem_per_batch_shadowkv = mem_increment_shadowkv / shadowkv_bsz
    mem_per_token_shadowkv = mem_increment_shadowkv / total_tokens_shadowkv
    print(colored(
        f"[ShadowKV] Memory after model loading on {llm.device}: {mem_after_load_shadowkv:.2f} GB", 'cyan'))
    print(
        colored(f"[ShadowKV] Throughput: {throughput_shadowkv} tokens/s", 'red'))
    print(colored(
        f"[ShadowKV] Peak CUDA memory on {llm.device}: {peak_mem_shadowkv:.2f} GB", 'yellow'))
    print(colored(
        f"[ShadowKV] Memory increment: {mem_increment_shadowkv:.2f} GB", 'yellow'))
    print(colored(
        f"[ShadowKV] Memory per batch: {mem_per_batch_shadowkv:.2f} GB", 'yellow'))
    print(colored(
        f"[ShadowKV] Memory per token: {mem_per_token_shadowkv:.4f} GB", 'yellow'))
    print(colored(
        f"[ShadowKV] Elapsed time: {end_time_shadowkv - start_time_shadowkv:.3f} s", 'yellow'))

    print(
        colored(f"Speedup: {throughput_shadowkv / throughput_baseline:.2f}x", 'red'))
