import torch
from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase
from models.kvcache.oracle_topk import OracleTopkCache


class OracleTopKAttention(AttentionBase):
    """
    Oracle TopK Attention: 在解码阶段的每个时间步精确计算注意力分数，
    动态选择top-k个KV cache位置参与注意力计算。
    
    核心特点：
    1. Prefill阶段：存储完整的KV cache
    2. Decode阶段：对每个新生成的token，计算它与所有历史key的注意力分数
    3. 对每个KV head group，使用所有query heads中最大的权重作为代理分数
    4. 根据代理分数选择top-k位置，为同一group的所有queries使用相同的位置子集
    """
    
    def __init__(self, kv_cache, num_key_value_groups, head_dim, hidden_size, rope_f=None, rope_f_single=None, cos_sin_cache=None):
        super().__init__(kv_cache, num_key_value_groups, head_dim, hidden_size, rope_f, rope_f_single, cos_sin_cache)
        # Create copy stream for async transfers
        self.copy_stream = torch.cuda.Stream()
    
    def prefill(self, query_states, key_states, value_states, position_ids, layer_idx):
        """
        Prefill阶段：存储完整的KV cache（不计算注意力分数）
        
        Args:
            query_states: [bsz, num_q_heads, prefill_len, head_dim]
            key_states: [bsz, num_kv_heads, prefill_len, head_dim]
            value_states: [bsz, num_kv_heads, prefill_len, head_dim]
            position_ids: [bsz, prefill_len]
            layer_idx: int
        
        Returns:
            hidden_states: [bsz, prefill_len, hidden_size]
        """
        kv_cache = self.kv_cache
        
        # 应用RoPE旋转位置编码
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        
        # 存储完整的KV cache（不计算注意力分数）
        kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states)
        
        # 使用标准flash attention进行prefill（完整上下文）
        if self.minference:
            hidden_states = self.minference(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                minference_parttern=self.minference_parttern[layer_idx]
            )
        else:
            hidden_states = flash_attn_with_kvcache(
                q=query_states.transpose(1, 2),
                k_cache=key_states.transpose(1, 2),
                v_cache=value_states.transpose(1, 2),
                causal=True
            )
        
        return hidden_states
    
    def decode(self, query_states, key_states, value_states, position_ids, layer_idx):
        """
        Decode阶段：精确计算当前query与所有历史key的注意力分数，选择top-k位置
        
        流程：
        1. 应用RoPE
        2. 精确计算当前query与所有历史key的注意力分数
        3. 选择top-k位置
        4. 检索对应的KV cache
        5. 使用稀疏KV cache进行精确的注意力计算
        
        Args:
            query_states: [bsz, num_q_heads, q_len, head_dim]
            key_states: [bsz, num_kv_heads, q_len, head_dim]
            value_states: [bsz, num_kv_heads, q_len, head_dim]
            position_ids: [bsz, q_len]
            layer_idx: int
        
        Returns:
            hidden_states: [bsz, q_len, hidden_size]
        """
        kv_cache = self.kv_cache
        
        # 应用RoPE旋转位置编码
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        
        # 更新生成的token到CPU cache
        kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        
        # ========================================
        # 核心步骤：计算当前query与所有历史key的注意力分数，选择top-k位置
        # ========================================
        retrieval_position_ids = kv_cache.get_retrieval_position_ids(
            layer_idx=layer_idx, 
            query_states=query_states
        )
        # retrieval_position_ids: [bsz, num_kv_heads, sparse_budget]
        
        # 多流传输：异步获取value cache
        curr_stream = torch.cuda.current_stream()
        get_value_stream = self.copy_stream
        
        with torch.cuda.stream(get_value_stream):
            get_value_stream.wait_stream(curr_stream)
            value_states_sparse = kv_cache.get_value_cache(layer_idx, retrieval_position_ids)
        
        # 获取key cache
        key_states_sparse = kv_cache.get_key_cache(layer_idx, retrieval_position_ids)
        # key_states_sparse: [bsz, num_kv_heads, sparse_len, head_dim]
        # value_states_sparse: [bsz, num_kv_heads, sparse_len, head_dim]
        
        # 等待value cache传输完成
        curr_stream.wait_stream(get_value_stream)
        
        # 使用稀疏KV cache进行注意力计算
        if self.minference:
            hidden_states = self.minference(
                query_states=query_states,
                key_states=key_states_sparse,
                value_states=value_states_sparse,
                minference_parttern=self.minference_parttern[layer_idx]
            )
        else:
            hidden_states = flash_attn_with_kvcache(
                q=query_states.transpose(1, 2),
                k_cache=key_states_sparse.transpose(1, 2),
                v_cache=value_states_sparse.transpose(1, 2),
                causal=True
            )
        
        return hidden_states


