import math
import torch
import torch.nn as nn


def collect_kv(self, layer_idx, query_states, func):
    A = self.A[layer_idx] # torch.Size([8, seq, 16])
    B = self.B[layer_idx] # torch.Size([8, 16, 128])
    self.incoming_q_len = query_states.shape[-2]
    recon_k = torch.einsum('bsd,bdD->bsD', A, B).unsqueeze(0) # low rank recon key  # [1, 8, seq, 128]
    
    attn_weights = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), recon_k.transpose(2, 3)) / math.sqrt(128)
    attn_weights = nn.functional.softmax(attn_weights.squeeze(2), dim=-1, dtype=torch.float32).to(self.dtype)

    attn_weights = attn_weights.sum(dim = -2) # [bsz, 8, 4, chunks]
    if self.num_key_value_groups > 1:
        attn_weights, _ = torch.max(attn_weights, dim=-2)
    merged_results = torch.topk(attn_weights, k=self.select_sets, dim=-1).indices

    position_ids = merged_results.unsqueeze(-1).view(1, self.num_key_value_heads, -1)  # [bsz, 8, select_sets * chunk_size]
    value_ = self.v_cache_backup[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
    
    if layer_idx == self.num_layers - 1:
        cur_len = self.kv_offset
    else:
        cur_len = self.kv_offset + self.incoming_q_len

    recent_budget = cur_len - self.prefill
    ret_v = torch.cat([value_, self.v_cache_backup[layer_idx][:,:,self.prefill:cur_len]], dim = 2)

    # gather ret_k
    self.k_cache_gen[layer_idx][:,:,:self.sparse_budget].copy_(self.k_cache_backup[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)))
    ret_k = self.k_cache_gen[layer_idx][:, :, :self.sparse_budget + recent_budget].clone()

    return ret_v, ret_k