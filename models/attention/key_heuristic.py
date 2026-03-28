"""Context key heuristic search"""
import torch
from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase

class LocalDivAttn(AttentionBase):

    def prefill(self, query_states, key_states, value_states, position_ids, layer_idx):
        kv_cache = self.kv_cache
        # svd unrope key and save
        kv_cache.get_svd(key_states, layer_idx=layer_idx)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
        
        if self.minference:
            hidden_states = self.minference(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
        else:
            hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
        
        return hidden_states

    def decode(self, query_states, key_states, value_states, position_ids, layer_idx):
        kv_cache = self.kv_cache 
            
        # rope query and key
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

        # update kv cache to buffer
        kv_cache.update_kv_cache(key_states, value_states, layer_idx)

        # get retrieval idx
        position_ids = kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)

        # multi-stream
        curr_stream = torch.cuda.current_stream()
        get_value_stream = kv_cache.copy_stream

        with torch.cuda.stream(get_value_stream):
            get_value_stream.wait_stream(curr_stream)
            value_states = kv_cache.get_value_cache(layer_idx, position_ids)

        # gather key cache from GPU and RoPE it (should be hide by CPU offloading time)
        key_states = kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)

        curr_stream.wait_stream(get_value_stream)

        # flash attention
        hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        return hidden_states