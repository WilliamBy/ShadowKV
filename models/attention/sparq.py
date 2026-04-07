import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
from .base import AttentionBase
from models.kvcache.sparq import SparQCache
from models.tensor_op import repeat_kv

class SparQAttention(AttentionBase):

    def prefill(self, query_states, key_states, value_states, position_ids, layer_idx):
        assert isinstance(self.kv_cache, SparQCache)

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        self.kv_cache.prefill_kv_cache(key_states, value_states, layer_idx)
        
        # Full flash attention during prefill
        hidden_states = flash_attn_with_kvcache(
            q=query_states.transpose(1, 2), 
            k_cache=key_states.transpose(1, 2), 
            v_cache=value_states.transpose(1, 2), 
            causal=True
        )
        return hidden_states

    def decode(self, query_states, key_states, value_states, position_ids, layer_idx):
        assert isinstance(self.kv_cache, SparQCache)
        kv_cache = self.kv_cache
        r = kv_cache.r
        k = kv_cache.k

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        
        # Get full K, V, and v_mean up to current sequence length
        K, V = kv_cache.collect_kv(layer_idx)
        V_mean = kv_cache.v_mean[layer_idx]
        
        bsz, num_heads, q_len, head_dim = query_states.size()
        self.num_attention_heads = num_heads
        num_kv_heads = self.num_attention_heads // self.num_key_value_groups
        g = self.num_key_value_groups
        seq_len = K.size(2)
        
        # Fallback to standard full attention if seq_len is smaller than budget k
        if seq_len <= k:
            K_rep = repeat_kv(K, g)
            V_rep = repeat_kv(V, g)
            attn_weights = torch.matmul(query_states, K_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            hidden_states = torch.matmul(attn_weights, V_rep)
            return hidden_states.view(bsz, q_len, self.hidden_size)

        # Reshape Q for Group Query Attention mapping
        Q = query_states.view(bsz, num_kv_heads, g, head_dim)
        
        # --- Step 1: Approximate attention scores ---
        # Find top r indices along the embedding dimension based on query magnitude
        Q_mag = torch.abs(Q).sum(dim=2, keepdim=True) # [bsz, num_kv_heads, 1, head_dim]
        i1 = torch.topk(Q_mag, r, dim=-1).indices # [bsz, num_kv_heads, 1, r]

        # Gather subset of Q and K components
        Q_hat = torch.gather(Q, dim=-1, index=i1.expand(bsz, num_kv_heads, g, r))
        K_hat = torch.gather(K, dim=-1, index=i1.expand(bsz, num_kv_heads, seq_len, r))
        
        # Adjusted L1 ratio based scaling (with epsilon)
        q_hat_sum = torch.abs(Q_hat).sum(dim=-1, keepdim=True)
        q_sum = torch.abs(Q).sum(dim=-1, keepdim=True) + 1e-6
        scale = (self.head_dim ** 0.5) * (q_hat_sum / q_sum)
        
        # Approximate attention scores
        s_hat = torch.matmul(Q_hat, K_hat.transpose(-1, -2)) / scale
        s_hat = F.softmax(s_hat, dim=-1) # [bsz, num_kv_heads, g, seq_len]

        # --- Step 2: Extract top-k KV pairs ---
        # Sum approximate scores over the GQA group to find top-k shared KV pairs
        s_hat_group = s_hat.sum(dim=2, keepdim=True) # [bsz, num_kv_heads, 1, seq_len]
        i2 = torch.topk(s_hat_group, k, dim=-1).indices # [bsz, num_kv_heads, 1, k]

        # Gather top-k full K and V vectors
        i2_expand = i2.transpose(-1, -2).expand(bsz, num_kv_heads, k, head_dim)
        K_topk = torch.gather(K, dim=-2, index=i2_expand) # [bsz, num_kv_heads, k, head_dim]
        V_topk = torch.gather(V, dim=-2, index=i2_expand) # [bsz, num_kv_heads, k, head_dim]

        # Standard precise attention over the selected k pairs
        K_topk_flat = repeat_kv(K_topk, g) # [bsz, num_heads, k, head_dim]
        V_topk_flat = repeat_kv(V_topk, g) # [bsz, num_heads, k, head_dim]

        attn_weights = torch.matmul(query_states, K_topk_flat.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1) # [bsz, num_heads, 1, k]
        
        y_ = torch.matmul(attn_weights, V_topk_flat) # [bsz, num_heads, 1, head_dim]

        # --- Step 3: Mean Reallocation ---
        # Calculate alpha weight: the total sum of approximate scores for the chosen top-k pairs
        alpha = torch.gather(s_hat, dim=-1, index=i2.expand(bsz, num_kv_heads, g, k)).sum(dim=-1, keepdim=True) 
        alpha = alpha.view(bsz, num_heads, 1, 1)

        V_mean_rep = repeat_kv(V_mean.unsqueeze(2), g) # [bsz, num_heads, 1, head_dim]
        
        # Interpolate between calculated output and missing background V-mean
        y = alpha * y_ + (1 - alpha) * V_mean_rep

        return y.view(bsz, q_len, self.hidden_size)