import torch
from utils import gc_and_sync

from models.tensor_op import repeat_kv
from models.kvcache.kv_base import KVCacheBase


class QuestCache(KVCacheBase):
    """Use Max and Min landmarks for retrieval"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
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
        self.chunk_size = chunk_size

        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark_max = [] # [(bsz, kv_head, chunks, dim)]
        self.k_landmark_min = []

    def print_stats(self):
        print(f"QuestCache | sparse budget {self.sparse_budget} | maxlen {self.max_length} | chunk size {self.chunk_size} | cached {self.kv_offset}")

    def register_k_landmark(self, k_landmark_max, k_landmark_min):
        self.k_landmark_max.append(k_landmark_max.clone())
        self.k_landmark_min.append(k_landmark_min.clone())

    def prefill_kv_cache(self,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        
        incoming = new_k_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()
        self.k_cache[layer_idx][:, :, :incoming] = new_k_cache.clone()

        self.chunks = incoming // self.chunk_size - 32 // self.chunk_size
        self.select_sets = self.sparse_budget // self.chunk_size
        
        self.chunk_end = self.chunks * self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        key_states_roped_ctx = new_k_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        
        k_landmark_max = key_states_roped_ctx.min(dim=-2).values
        k_landmark_min = key_states_roped_ctx.max(dim=-2).values

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(k_landmark_max, k_landmark_min)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def collect_kv(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2] # 1
        min_cache = repeat_kv(self.k_landmark_min[layer_idx], self.num_key_value_groups)
        max_cache = repeat_kv(self.k_landmark_max[layer_idx], self.num_key_value_groups)
        min_value = min_cache * query_states
        max_value = max_cache * query_states

        heuristic = torch.max(min_value, max_value)
        heuristic = heuristic.sum(dim=-1)
        
        heuristic = heuristic.reshape(1, self.num_key_value_heads, self.num_key_value_groups, -1)
        heuristic = heuristic.sum(dim=-2, keepdim=True)
        
        topk_chunk = heuristic.topk(k=self.select_sets, dim=-1).indices

        position_ids = (topk_chunk.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=topk_chunk.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(1, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        key_ = self.k_cache[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).to('cpu'))

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        ret_k = torch.cat([key_, self.k_cache[layer_idx][:,:,self.chunk_end:self.prefill+gen_offset]], dim = 2).to(self.device)
        ret_v = torch.cat([value_, self.v_cache_cpu[layer_idx][:,:,self.chunk_end:self.prefill+gen_offset]], dim = 2).to(self.device)

        return ret_k, ret_v
        
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.k_cache[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache.zero_()
        self.v_cache_cpu.zero_()
        self.k_landmark_max = []
        self.k_landmark_min = []

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        gc_and_sync()

    def get_kv_len(self):
        return self.kv_offset
    
    def H2D(self):
        pass