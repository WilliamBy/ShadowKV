import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from data.dataset import Dataset
import torch
from models.base import LLM
from models import choose_model_class

model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"

LLM = choose_model_class(model_name)

datalen = 131072

llm = LLM(model_name=model_name, batch_size=1, device="cuda:0", max_length=datalen, attn_mode="shadowkv", dtype=torch.bfloat16, sparse_budget=2048, rank=160, chunk_size=8, minference=False)

dataset = Dataset("long_bench/trec", llm.tokenizer, datalen, -1, 0, 1)
print(f"in: {dataset.tokenized_prompts[0].shape}")

tokenized_prompt = llm.encode(dataset.tokenizer.decode(dataset.tokenized_prompts[0].squeeze(0)), template="ctx", truncation=False)
print(f"out: {tokenized_prompt.shape}")
