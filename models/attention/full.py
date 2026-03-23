import torch
from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase

class FullAttention(AttentionBase):
    def prefill(self, llm, query_states, key_states, value_states, position_ids, layer_idx, kv_cache, minference=None, minference_parttern=None):
        query_states, key_states = llm.apply_rotary_pos_emb(query_states, key_states, position_ids)
        key_states, value_states = kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        
        if minference == True and query_states.shape[2] > 1:
            hidden_states = llm.minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=minference_parttern[layer_idx])
        else:
            hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
        
        return hidden_states

    def decode(self, llm, query_states, key_states, value_states, position_ids, layer_idx, kv_cache, minference=None, minference_parttern=None):
        return self.prefill(llm, query_states, key_states, value_states, position_ids, layer_idx, kv_cache, minference, minference_parttern)