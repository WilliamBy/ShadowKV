import torch
import gc

from .kv_base import KVCacheBase


class FullKVCache(KVCacheBase):
    """Full Attention"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:

        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

        # batch prefill record
        self.prefilled_batch = 0
        self.batch_size = batch_size

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]

        if bsz == self.batch_size:
            self.prefilled_batch = 0

        self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        key = self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]
        value = self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]

        if incoming > 1: # prefill
            key = key.to(self.device)
            value = value.to(self.device)

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        return key.to(self.device), value.to(self.device)
    
    def print_stats(self):
        print(f"FullKVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset