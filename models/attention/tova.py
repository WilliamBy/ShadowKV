import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase
from models.kvcache.tova import TOVACache
from models.tensor_op import repeat_kv
from transformers.models.llama.modeling_llama import LlamaSdpaAttention

class TovaAttention(AttentionBase):
    def prefill(self, llm, query_states, key_states, value_states, position_ids, layer_idx, kv_cache: TOVACache, minference=None, minference_parttern=None):
        query_states, key_states = llm.apply_rotary_pos_emb(query_states, key_states, position_ids)
        kv_cache.prefill_kv_cache(key_states, value_states, layer_idx)
        hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
        return hidden_states

    def decode(self, llm, query_states, key_states, value_states, position_ids, layer_idx, kv_cache: TOVACache, minference=None, minference_parttern=None):
        query_states, key_states = llm.apply_rotary_pos_emb(query_states, key_states, position_ids)
        kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        key_states, value_states = kv_cache.collect_kv(layer_idx, query_states)
        # Expand for heads
        key_states = repeat_kv(key_states, llm.num_key_value_groups)
        value_states = repeat_kv(value_states, llm.num_key_value_groups)
        # Compute attn
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / (llm.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, value_states)
        # Reshape
        bsz, q_len, _ = query_states.size()
        hidden_states = hidden_states.view(bsz, q_len, llm.hidden_size)
        # Reduce
        kv_cache.reduce(layer_idx, attn_weights)
        return hidden_states