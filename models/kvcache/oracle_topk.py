################################################################################
#
# Oracle TopK KV Cache Implementation
#
# This implementation computes exact attention scores at each decode step,
# then selects top-k tokens per KV head using the maximum attention weight
# across all queries in the same KV group.
#
################################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import KVCacheBase


class OracleTopkCache(KVCacheBase):
    """
    Oracle TopK Cache: 在解码阶段的每个时间步计算当前query与所有历史KV的
    注意力分数，动态选择top-k位置参与注意力计算。
    
    Key features:
    - 在解码阶段的每个时间步精确计算注意力分数
    - 对于每个KV head，使用其group中所有queries的最大注意力权重作为代理分数
    - 每个KV head独立选择top-k tokens
    - 为同一KV group中的所有queries使用相同的位置子集（如ShadowKV）
    """
    
    def __init__(self, 
        config: object,
        batch_size: int = 1,
        max_length: int = 32*1024, 
        device: str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        ) -> None:
        
        super().__init__()
        
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
        
        assert self.batch_size == 1, "OracleTopkCache only supports batch_size=1"
        
        # 存储完整的 key 和 value cache（用于计算注意力）
        # [num_layers, bsz, num_kv_heads, max_length, head_dim]
        self.k_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
            device=device,
            dtype=dtype
        )
        
        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
            device=device,
            dtype=dtype
        )
        
        # GPU buffer for selected KV cache
        # Layout: [top-k sparse tokens | recent generation tokens]
        buffer_size = self.sparse_budget + 4096
        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            buffer_size,
            self.head_dim,
            device=device,
            dtype=dtype
        )
        
        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            buffer_size,
            self.head_dim,
            device=device,
            dtype=dtype
        )
        
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.gen_offset = 0
        self.sparse_start = 0
        self.sparse_end = self.sparse_budget
        
    def print_stats(self):
        print(f"OracleTopkCache Stats:")
        print(f"  Sparse budget: {self.sparse_budget}")
        print(f"  KV length: {self.kv_offset}")
        print(f"  Gen offset: {self.gen_offset}")
        print(f"  Sparse range: [{self.sparse_start}, {self.sparse_end})")
    
    def prefill_kv_cache(self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query_states: torch.Tensor = None):
        """
        Prefill阶段：只存储完整的KV cache，不计算注意力分数
        
        Args:
            new_v_cache: [bsz, num_kv_heads, prefill_len, head_dim]
            layer_idx: int
            key_states_roped: [bsz, num_kv_heads, prefill_len, head_dim]
            query_states: [bsz, num_q_heads, prefill_len, head_dim] - 在prefill阶段不使用
        """
        bsz, num_kv_heads, prefill_len, head_dim = new_v_cache.shape
        
        # 存储完整的key和value cache到CPU
        self.k_cache_cpu[layer_idx][:, :, :prefill_len].copy_(key_states_roped)
        self.v_cache_cpu[layer_idx][:, :, :prefill_len].copy_(new_v_cache)
        
        if layer_idx == self.num_layers - 1:
            self.kv_offset += prefill_len
    
    def get_retrieval_position_ids(self, layer_idx: int, query_states: torch.Tensor):
        """
        解码阶段：精确计算当前query与所有历史key的注意力分数，选择top-k位置
        
        这是Oracle TopK的核心方法，在解码阶段的每个时间步调用：
        1. 计算当前query与所有历史key [1, 2, ..., t-1] 的注意力分数
        2. 对每个KV head group，使用所有query heads中最大的权重作为代理分数
        3. 根据代理分数选择top-k位置
        
        Args:
            query_states: [bsz, num_q_heads, q_len, head_dim]
        
        Returns:
            selected_positions: [bsz, num_kv_heads, sparse_budget]
        """
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        num_kv_heads = self.num_key_value_heads
        num_groups = self.num_key_value_groups
        
        # 只使用有效长度的cache
        valid_len = self.kv_offset
        
        # 获取有效的key cache: [bsz, num_kv_heads, valid_len, head_dim]
        valid_k_cache = self.k_cache_cpu[layer_idx][:, :, :valid_len, :]
        
        # 重排query以匹配KV heads: [bsz, num_kv_heads, num_groups, q_len, head_dim]
        query_grouped = query_states.view(bsz, num_kv_heads, num_groups, q_len, head_dim)
        
        # 精确计算注意力分数: [bsz, num_kv_heads, num_groups, q_len, valid_len]
        attn_scores = torch.einsum('bhgqd,bhkd->bhgqk', query_grouped, valid_k_cache) / math.sqrt(head_dim)
        
        # 转换为注意力权重（使用softmax）
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(self.dtype)
        # attn_weights: [bsz, num_kv_heads, num_groups, q_len, valid_len]
        
        # 对同一KV group，使用所有query heads中最大的权重作为代理分数
        # [bsz, num_kv_heads, num_groups, q_len, valid_len] -> [bsz, num_kv_heads, valid_len]
        proxy_scores, _ = torch.max(attn_weights, dim=2)  # max over num_groups
        proxy_scores, _ = torch.max(proxy_scores, dim=2)  # max over q_len
        
        # 对每个KV head独立选择TopK
        # [bsz, num_kv_heads, valid_len] -> [bsz, num_kv_heads, sparse_budget]
        topk_indices = torch.topk(proxy_scores, k=min(self.sparse_budget, valid_len), dim=-1).indices
        
        # 排序选中的索引以获得更好的内存局部性
        topk_indices = torch.sort(topk_indices, dim=-1).values
        
        return topk_indices
    
    def get_value_cache(self, layer_idx: int, position_ids: torch.Tensor):
        """
        根据位置ID从CPU缓存中收集value cache
        
        Args:
            position_ids: [bsz, num_kv_heads, sparse_budget]
        
        Returns:
            value_cache: [bsz, num_kv_heads, sparse_budget + gen_offset, head_dim]
        """
        # 从CPU cache收集value
        # [bsz, num_kv_heads, sparse_budget, head_dim]
        value_gathered = self.v_cache_cpu[layer_idx].gather(
            dim=-2,
            index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        
        # 复制到GPU buffer的sparse区域
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(
            value_gathered, non_blocking=True
        )
        
        # 包含新生成的tokens
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else 0
        total_len = self.sparse_end + gen_offset
        
        return self.v_cache_buffer[layer_idx][:, :, :total_len]
    
    def get_key_cache(self, layer_idx: int, position_ids: torch.Tensor):
        """
        根据位置ID从CPU缓存中收集key cache
        
        Args:
            position_ids: [bsz, num_kv_heads, sparse_budget]
        
        Returns:
            key_cache: [bsz, num_kv_heads, sparse_budget + gen_offset, head_dim]
        """
        # 从CPU cache收集key
        # [bsz, num_kv_heads, sparse_budget, head_dim]
        key_gathered = self.k_cache_cpu[layer_idx].gather(
            dim=-2,
            index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        
        # 复制到GPU buffer的sparse区域
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(
            key_gathered, non_blocking=True
        )
        
        # 包含新生成的tokens
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else 0
        total_len = self.sparse_end + gen_offset
        
        return self.k_cache_buffer[layer_idx][:, :, :total_len]
    
    def update_kv_cache(self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int):
        """
        更新生成的token的KV cache
        
        Args:
            new_k_cache: [bsz, num_kv_heads, incoming, head_dim]
            new_v_cache: [bsz, num_kv_heads, incoming, head_dim]
            layer_idx: int
        """
        incoming = new_k_cache.shape[-2]
        
        # 添加到CPU cache
        start = self.kv_offset
        end = start + incoming
        self.k_cache_cpu[layer_idx][:, :, start:end].copy_(new_k_cache)
        self.v_cache_cpu[layer_idx][:, :, start:end].copy_(new_v_cache)
        
        # 复制到GPU buffer（在sparse区域之后）
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(
            new_v_cache, non_blocking=True
        )
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(
            new_k_cache, non_blocking=True
        )
        
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming
    
    def clear(self):
        """清除所有缓存"""
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_cache_cpu.zero_()
        self.v_cache_cpu.zero_()
        
        self.kv_offset = 0
        self.gen_offset = 0
    
    def H2D(self):
        """Host to Device transfer (no-op for OracleTopk)"""
        pass
    
    def get_kv_len(self):
        """获取当前KV cache长度"""
        return self.kv_offset
