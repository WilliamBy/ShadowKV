import torch
from models.kvcache.base import KVCacheBase
from utils.logger import get_logger

logger = get_logger(__name__)

class StreamingKVCache(KVCacheBase):
    """StreamingKVCache: GPU-only version keeping only attention sinks + local window tokens."""

    def __init__(
        self,
        config,
        batch_size=1,
        max_length=32*1024,
        device='cuda:0',
        dtype=torch.bfloat16,
        sink_size=4,
        sparse_budget=512,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sink_size = sink_size
        self.local_window_size = sparse_budget

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.sparse_budget = sparse_budget

    def print_stats(self):
        print(
            f"StreamingKVCache | sparse budget {self.sparse_budget} | max_len {self.max_length} | sink_size {self.sink_size} | local_window_size {self.local_window_size} | kv_offset {self.kv_offset}"
        )

    def prefill_kv_cache(self, new_v_cache: torch.Tensor, layer_idx: int, new_rope_key_states: torch.Tensor):
        incoming = new_v_cache.shape[-2]
        self.k_cache_buffer[layer_idx][:, :, :incoming].copy_(new_rope_key_states)
        self.v_cache_buffer[layer_idx][:, :, :incoming].copy_(new_v_cache)
        self.kv_offset = incoming

    def update_kv_cache(self, new_k_cache: torch.Tensor, new_v_cache: torch.Tensor, layer_idx: int):
        incoming = new_k_cache.size(-2)
        if self.kv_offset + incoming > self.max_length:
            raise ValueError("No Sufficient KVCache Buffer to Allocate")

        self.k_cache_buffer[layer_idx][:, :, self.kv_offset:self.kv_offset+incoming].copy_(new_k_cache)
        self.k_cache_buffer[layer_idx][:, :, self.kv_offset:self.kv_offset+incoming].copy_(new_v_cache)

        self.kv_offset += incoming

    def get_value_cache(self, layer_idx):
        if self.kv_offset > self.sparse_budget:
            return torch.cat([
                self.v_cache_buffer[layer_idx][:, :, :self.sink_size],
                self.v_cache_buffer[layer_idx][:, :, -self.local_window_size:]
            ])

        return self.v_cache_buffer[layer_idx][:, :, :self.kv_offset]

    def get_key_cache(self, layer_idx):
        if self.kv_offset > self.sparse_budget:
            return torch.cat([
                self.k_cache_buffer[layer_idx][:, :, :self.sink_size],
                self.k_cache_buffer[layer_idx][:, :, -self.local_window_size:]
            ])

        return self.k_cache_buffer[layer_idx][:, :, :self.kv_offset]

    def clear(self):
        self.kv_offset = 0
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()

    def get_kv_len(self):
        return self.kv_offset

    def H2D(self):
        pass
