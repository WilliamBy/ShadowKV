from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase
from ..kvcache import StreamingKVCache

class StreamingAttention(AttentionBase):
    def prefill(self, query_states, key_states, value_states, position_ids, layer_idx):
        assert isinstance(StreamingKVCache)
        kv_cache = self.kv_cache
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        kv_cache.prefill_kv_cache(value_states, layer_idx, key_states)

        if self.minference:
            hidden_states = self.minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
        else:
            hidden_states = flash_attn_with_kvcache(
                q=query_states.transpose(1, 2),
                k_cache=key_states.transpose(1, 2),
                v_cache=value_states.transpose(1, 2),
                causal=True,
            )

        return hidden_states

    def decode(self, query_states, key_states, value_states, position_ids, layer_idx):
        assert isinstance(StreamingKVCache)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

        kv_cache = self.kv_cache
        kv_cache.update_kv_cache(key_states, value_states, layer_idx)

        value_states = kv_cache.get_value_cache(layer_idx)
        key_states = kv_cache.get_key_cache(layer_idx)

        hidden_states = flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=key_states.transpose(1, 2),
            v_cache=value_states.transpose(1, 2),
            causal=True,
        )

        return hidden_states
