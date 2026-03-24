from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase

class FullAttention(AttentionBase):
    def prefill(self, query_states, key_states, value_states, position_ids, layer_idx):
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        
        if self.minference is not None and query_states.shape[2] > 1:
            hidden_states = self.minference(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
        else:
            hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
        
        return hidden_states

    def decode(self, query_states, key_states, value_states, position_ids, layer_idx):
        return self.prefill(query_states, key_states, value_states, position_ids, layer_idx)