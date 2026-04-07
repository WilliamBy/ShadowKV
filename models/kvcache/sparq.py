import torch
from utils import gc_and_sync
from models.kvcache.base import KVCacheBase

class SparQCache(KVCacheBase):
    """
    SparQ Cache Implementation: 
    Retains all KV pairs, but maintains a running mean of Value vectors (v_mean) 
    for the Mean Reallocation step in SparQ Attention.
    """
    def __init__(self, 
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = 'cuda:0',
        dtype = torch.bfloat16,
        r: int = 32,
        k: int = 128,
    ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_attention_heads

        # SparQ Hyperparameters
        self.r = int(r)
        self.k = int(k)
        self.num_layers = config.num_hidden_layers
        self.rank_ratio = int(r) / self.head_dim

        # Buffer for K and V cache
        self.k_cache = torch.zeros(
            self.num_layers, batch_size, self.num_key_value_heads, self.max_length + 4096, self.head_dim,
            device=self.device, dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            self.num_layers, batch_size, self.num_key_value_heads, self.max_length + 4096, self.head_dim,
            device=self.device, dtype=self.dtype
        )

        # Running mean of V vectors for Mean Reallocation step
        self.v_mean = torch.zeros(
            self.num_layers, batch_size, self.num_key_value_heads, self.head_dim,
            device=self.device, dtype=self.dtype
        )

        self.layer_seq_len = [0] * self.num_layers

    def print_stats(self):
        print(f"SparQCache | rank_ratio {self.rank_ratio}({self.r}/{self.head_dim}) | sparse_budget {self.k} | maxlen {self.max_length} |  ")

    def prefill_kv_cache(self, new_k_cache: torch.Tensor, new_v_cache: torch.Tensor, layer_idx: int):
        incoming = new_k_cache.shape[-2]
        
        self.k_cache[layer_idx][:, :, :incoming] = new_k_cache.to(self.device)
        self.v_cache[layer_idx][:, :, :incoming] = new_v_cache.to(self.device)
        
        # Calculate V mean across the prefilled sequence
        self.v_mean[layer_idx] = new_v_cache.to(self.device).mean(dim=-2)
        
        self.layer_seq_len[layer_idx] = incoming

    def collect_kv(self, layer_idx: int):
        """Returns the full currently active K and V cache."""
        current_len = self.layer_seq_len[layer_idx]
        ret_k = self.k_cache[layer_idx][:, :, :current_len]
        ret_v = self.v_cache[layer_idx][:, :, :current_len]
        return ret_k, ret_v
        
    def update_kv_cache(self, new_k_cache: torch.Tensor, new_v_cache: torch.Tensor, layer_idx: int):
        incoming = new_k_cache.shape[-2]
        current_len = self.layer_seq_len[layer_idx]
        
        self.k_cache[layer_idx][:, :, current_len:current_len + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][:, :, current_len:current_len + incoming].copy_(new_v_cache)
        
        # Iteratively update the running mean of V
        old_mean = self.v_mean[layer_idx]
        new_v_sum = new_v_cache.sum(dim=-2)
        self.v_mean[layer_idx] = (old_mean * current_len + new_v_sum) / (current_len + incoming)
        
        self.layer_seq_len[layer_idx] += incoming

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.v_mean.zero_()
        self.layer_seq_len = [0] * self.num_layers
        gc_and_sync()

    def get_kv_len(self):
        return max(self.layer_seq_len)
    
    def H2D(self):
        pass