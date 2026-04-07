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

# Base LLM class

import torch
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm

from .tensor_op import sample_token, layer_norm
from .kvcache import *
from .attention import Cache2Attn
from utils.logger import get_logger

logger = get_logger(__name__)

class LLM:

    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"

    # NOTE: register your kvcache method here
    def init_kv_cache(self, sparse_budget: int, rank: int, chunk_size: int, config, **kwargs):
        if self.attn_mode == 'full':
            self.kv_cache = FullKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        elif self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'shadowkv_cpu':
            self.kv_cache = ShadowKVCache_CPU(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'experimental':
            self.kv_cache = ExperimentalKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'optimized':
            self.kv_cache = OptKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'streaming':
            self.kv_cache = StreamingKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sink_size=4, sparse_budget=sparse_budget)
        elif self.attn_mode.lower() == 'quest':
            self.kv_cache = QuestCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'tova':
            self.kv_cache = TOVACache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget)
        elif self.attn_mode.lower() == 'local_div':
            dynamic_ratio = kwargs.get("dynamic_ratio", 0)
            self.kv_cache = LocalDivCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, dynamic_ratio=dynamic_ratio)
        elif self.attn_mode.lower() == 'random_outlier':
            dynamic_ratio = kwargs.get("dynamic_ratio", 0)
            self.kv_cache = RandomOutlierCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, dynamic_ratio=dynamic_ratio)
        elif self.attn_mode.lower() == 'oracle_topk':
            self.kv_cache = OracleTopkCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget)
        elif self.attn_mode.lower() == 'sparq':
            self.kv_cache = SparQCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, r=rank, k=sparse_budget)
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        cache_type = type(self.kv_cache)
        self.attn_hdlr = Cache2Attn[cache_type](self.kv_cache, self.num_key_value_groups, self.head_dim, self.hidden_size, self.apply_rotary_pos_emb, self.apply_rotary_pos_emb_single, self.cos_sin_cache)

        logger.info(f"Using KVCache Type: {self.kv_cache.__class__.__name__}\n Using Attention Type: {self.attn_hdlr.__class__.__name__}")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        # clear kvcache and re-prefill
        self.kv_cache.clear()
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        # prefill based on previous kvcache
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation, add_special_tokens=False).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        attn_hdlr = self.attn_hdlr 

        if self.minference:
            attn_hdlr.minference = self.minference_prefill_kernel
            attn_hdlr.minference_parttern = self.minference_parttern

        if q_len > 1: # prefill
            hidden_states = attn_hdlr.prefill(query_states, key_states, value_states, position_ids, layer_idx)
        else: # decoding
            hidden_states = attn_hdlr.decode(query_states, key_states, value_states, position_ids, layer_idx)

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        if bsz*q_len > 64*1024: # [bsz, seq, 128]
            output = torch.empty_like(hidden_states)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
            
            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """accuracy eval usage, not for throughput eval"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())
        
        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8
        else:
            T = 4
        # for bsz in range(0, batch_size, T):
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):

        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        for _ in range(100):
            torch.bmm(a, b)
        del a, b

        print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """throughput eval usage"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            logits = self.prefill_cont(input_ids)
        
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        generated_ids = []
        generated_ids.append(next_token[:, -1].tolist())
        
        self.kv_cache.H2D()
        self.warmup()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())

        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        if benchmark == True:
            total_generated_tokens = sum(len(seq) for seq in generated_ids)
            return self.decode(generated_ids, skip_special_tokens=True), total_generated_tokens / (end - start), total_generated_tokens

        return self.decode(generated_ids, skip_special_tokens=True)