import torch
from utils import gc_and_sync
from models.kvcache.base import KVCacheBase

class TOVACache(KVCacheBase):
    """
    TOVA Cache Implementation: 
    Retains the Top-K KV pairs based on average attention scores across heads.
    """
    def __init__(self, 
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
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

        self.sparse_budget = int(sparse_budget)
        self.num_layers = config.num_hidden_layers

        # TOVA drops tokens permanently, no need for a CPU buffer. 
        # Both K and V are placed directly on the device.
        self.k_cache = torch.zeros(
            self.num_layers,
            batch_size,
            self.num_key_value_heads,
            self.max_length+4096,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            self.num_layers,
            batch_size,
            self.num_key_value_heads,
            self.max_length+4096,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        # Because reduce() dynamically changes sequence length per layer during generation,
        # it is safer to track the sequence length for each layer independently.
        self.layer_seq_len = [0] * self.num_layers

    def print_stats(self):
        print(f"TOVACache | sparse budget {self.sparse_budget} | maxlen {self.max_length}")

    def prefill_kv_cache(self,
            new_k_cache: torch.Tensor,
            new_v_cache: torch.Tensor,
            layer_idx: int,
            ):
        
        incoming = new_k_cache.shape[-2] # [bsz, num_kv_heads, seq_len, head_dim]
        
        # Prefill directly to the front of the cache
        self.k_cache[layer_idx][:, :, :incoming] = new_k_cache.to(self.device)
        self.v_cache[layer_idx][:, :, :incoming] = new_v_cache.to(self.device)
        
        self.layer_seq_len[layer_idx] = incoming

    def collect_kv(self, layer_idx: int, query_states: torch.Tensor):
        """
        Unlike Quest, TOVA doesn't perform pre-attention heuristic retrieval. 
        It returns all CURRENTLY valid cached KV pairs for precise attention calculation.
        """
        current_len = self.layer_seq_len[layer_idx]
        
        ret_k = self.k_cache[layer_idx][:, :, :current_len]
        ret_v = self.v_cache[layer_idx][:, :, :current_len]
        
        return ret_k, ret_v
        
    def update_kv_cache(self, 
            new_k_cache: torch.Tensor,
            new_v_cache: torch.Tensor,
            layer_idx: int,
            ):
        """
        Append the newly generated token's KV to the active boundary.
        """
        incoming = new_k_cache.shape[-2]
        current_len = self.layer_seq_len[layer_idx]
        
        self.k_cache[layer_idx][:, :, current_len:current_len + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][:, :, current_len:current_len + incoming].copy_(new_v_cache)
        
        self.layer_seq_len[layer_idx] += incoming

    def reduce(self,
            layer_idx: int,
            attn_weights: torch.Tensor,
            ):
        """
        The core TOVA compression logic. Must be called AFTER attention weight calculation.
        attn_weights shape expected: [bsz, num_heads, q_len, kv_len]
        """
        current_len = self.layer_seq_len[layer_idx]
        
        # If cache hasn't exceeded the budget, skip reduction
        if current_len <= self.sparse_budget:
            return

        bsz, num_heads, _, num_keys = attn_weights.size()
        
        # Calculate mean attention weights across all heads for the LAST query token
        mean_attn_weights = torch.mean(attn_weights[:, :, -1, :], dim=1).clone().detach()

        # Get the indices of the Top-K elements
        vals, ind = torch.topk(mean_attn_weights, k=self.sparse_budget, dim=-1)
        
        # Sort indices to maintain the original relative temporal order of tokens
        ind = torch.sort(ind).values 
        
        # Expand indices for gather operation: [bsz, kv_heads, sparse_budget, head_dim]
        expand_ind = ind.unsqueeze(1).unsqueeze(-1).expand(bsz, self.num_key_value_heads, self.sparse_budget, self.head_dim)

        # In-place gather to compress the cache, keeping only the important tokens at the front of the buffer
        active_k = self.k_cache[layer_idx][:, :, :current_len]
        active_v = self.v_cache[layer_idx][:, :, :current_len]
        
        compacted_k = torch.gather(active_k, dim=2, index=expand_ind)
        compacted_v = torch.gather(active_v, dim=2, index=expand_ind)

        # Overwrite the front of the cache with compacted KV
        self.k_cache[layer_idx][:, :, :self.sparse_budget] = compacted_k
        self.v_cache[layer_idx][:, :, :self.sparse_budget] = compacted_v

        # Update the active length
        self.layer_seq_len[layer_idx] = self.sparse_budget

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.layer_seq_len = [0] * self.num_layers
        gc_and_sync()

    def get_kv_len(self):
        # Can return the maximum active length across all layers as an indicator
        return max(self.layer_seq_len)
    
    def H2D(self):
        pass