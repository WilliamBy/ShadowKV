import math
import torch
import torch.nn as nn

import torch
from utils import gc_and_sync

from models.tensor_op import repeat_kv
from models.kvcache.kv_base import KVCacheBase


class LokiCache(KVCacheBase):
    """Use Max and Min landmarks for retrieval"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        rank: int = 160
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)

        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device="cpu",
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device="cpu",
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0


    def print_stats(self):
        print(f"LokiCache | sparse budget {self.sparse_budget} | cached {self.kv_offset}")

    def prefill_kv_cache(self,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        
        incoming = new_k_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        
        self.v_cache[layer_idx][:, :, :incoming] = new_v_cache.clone()
        self.k_cache[layer_idx][:, :, :incoming] = new_k_cache.clone()

        self.chunks = incoming // self.chunk_size - 32 // self.chunk_size
        self.select_sets = self.sparse_budget // self.chunk_size
        
        self.chunk_end = self.chunks * self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        key_states_roped_ctx = new_k_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        
        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def collect_kv(self, layer_idx, query_states):
        A = self.A[layer_idx] # torch.Size([8, seq, 16])
        B = self.B[layer_idx] # torch.Size([8, 16, 128])
        self.incoming_q_len = query_states.shape[-2]
        recon_k = torch.einsum('bsd,bdD->bsD', A, B).unsqueeze(0) # low rank recon key  # [1, 8, seq, 128]
        
        attn_weights = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), recon_k.transpose(2, 3)) / math.sqrt(128)
        attn_weights = nn.functional.softmax(attn_weights.squeeze(2), dim=-1, dtype=torch.float32).to(self.dtype)

        attn_weights = attn_weights.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            attn_weights, _ = torch.max(attn_weights, dim=-2)
        merged_results = torch.topk(attn_weights, k=self.select_sets, dim=-1).indices

        position_ids = merged_results.unsqueeze(-1).view(1, self.num_key_value_heads, -1)  # [bsz, 8, select_sets * chunk_size]
        value_ = self.v_cache_backup[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        
        if layer_idx == self.num_layers - 1:
            cur_len = self.kv_offset
        else:
            cur_len = self.kv_offset + self.incoming_q_len

        recent_budget = cur_len - self.prefill
        ret_v = torch.cat([value_, self.v_cache_backup[layer_idx][:,:,self.prefill:cur_len]], dim = 2)

        # gather ret_k
        self.k_cache_gen[layer_idx][:,:,:self.sparse_budget].copy_(self.k_cache_backup[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)))
        ret_k = self.k_cache_gen[layer_idx][:, :, :self.sparse_budget + recent_budget].clone()

        return ret_v, ret_k
        
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.k_cache[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_landmark_max = []
        self.k_landmark_min = []

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        gc_and_sync()

    def get_kv_len(self):
        return self.kv_offset
    
    def H2D(self):
        gc_and_sync()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)
