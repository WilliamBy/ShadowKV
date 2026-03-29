import torch
import gc
from models.kvcache.base import KVCacheBase


class RandomOutlierKVCache(KVCacheBase):
    """Random Outlier KVCache: Randomly selects n tokens during prefill and maintains consistency during generation"""
    
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        ) -> None:

        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Number of tokens to randomly select and keep
        self.sparse_budget = int(sparse_budget)
        
        # Full cache stored on CPU (for reference)
        self.k_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            self.head_dim,
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            self.head_dim,
            device='cpu',
            dtype=self.dtype
        )

        # GPU buffer for selected tokens only
        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        # Track selected indices for consistency
        self.selected_indices = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.sparse_budget,
            device=self.device,
            dtype=torch.long
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefilled_batch = 0
        self.is_prefill = True

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):
        
        bsz, _, incoming, _ = new_v_cache.shape  # [bsz, num_kv_heads, incoming, head_dim]

        if bsz == self.batch_size:
            self.prefilled_batch = 0

        # Copy to CPU cache
        self.k_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache.cpu())
        self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache.cpu())

        if incoming > 1:  # Prefill phase
            # Randomly select sparse_budget tokens from the prefill sequence
            total_tokens = self.kv_offset + incoming
            
            if total_tokens <= self.sparse_budget:
                # If total tokens less than budget, keep all
                selected_idx = torch.arange(total_tokens, device=self.device)
                # Pad with zeros if needed
                if selected_idx.shape[0] < self.sparse_budget:
                    padding = torch.zeros(self.sparse_budget - selected_idx.shape[0], 
                                        device=self.device, dtype=torch.long)
                    selected_idx = torch.cat([selected_idx, padding])
            else:
                # Randomly select sparse_budget tokens
                selected_idx = torch.randperm(total_tokens, device=self.device)[:self.sparse_budget]
            
            # Store selected indices for this layer
            self.selected_indices[layer_idx][:bsz, :self.sparse_budget] = selected_idx.unsqueeze(0).expand(bsz, -1)
            
            # Gather selected key and value states
            k_cache_full = self.k_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :total_tokens].to(self.device)
            v_cache_full = self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :total_tokens].to(self.device)
            
            # Gather selected tokens
            key_states = torch.gather(k_cache_full, dim=2, 
                                    index=selected_idx[:self.sparse_budget].view(1, 1, -1, 1).expand(bsz, self.num_key_value_heads, -1, self.head_dim))
            value_states = torch.gather(v_cache_full, dim=2, 
                                      index=selected_idx[:self.sparse_budget].view(1, 1, -1, 1).expand(bsz, self.num_key_value_heads, -1, self.head_dim))
            
            # Store in GPU buffer
            self.k_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget].copy_(key_states)
            self.v_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget].copy_(value_states)
            
            self.is_prefill = False
            
        else:  # Decode phase - append new token
            # In decode phase, we need to maintain the selected tokens from prefill
            # and append the new token
            current_kv_len = self.kv_offset
            
            # Get the selected indices from prefill
            selected_idx = self.selected_indices[layer_idx][:bsz, :self.sparse_budget]
            
            # Gather selected tokens from CPU cache
            k_cache_full = self.k_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :current_kv_len].to(self.device)
            v_cache_full = self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :current_kv_len].to(self.device)
            
            # Gather previously selected tokens (excluding the last position which will be new token)
            key_states = torch.gather(k_cache_full, dim=2, 
                                    index=selected_idx[:, :self.sparse_budget-1].view(bsz, 1, -1, 1).expand(-1, self.num_key_value_heads, -1, self.head_dim))
            value_states = torch.gather(v_cache_full, dim=2, 
                                      index=selected_idx[:, :self.sparse_budget-1].view(bsz, 1, -1, 1).expand(-1, self.num_key_value_heads, -1, self.head_dim))
            
            # Append the new token
            key_states = torch.cat([key_states, new_k_cache], dim=2)
            value_states = torch.cat([value_states, new_v_cache], dim=2)
            
            # Update buffer
            self.k_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget].copy_(key_states)
            self.v_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget].copy_(value_states)
            
            # Update selected indices to include the new token position
            self.selected_indices[layer_idx][:bsz, self.sparse_budget-1] = current_kv_len

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        # Return the cached states from GPU buffer
        key_states = self.k_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget]
        value_states = self.v_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget]
        
        return key_states, value_states
    
    def print_stats(self):
        print(f"RandomOutlierKVCache | max_length {self.max_length} | sparse_budget {self.sparse_budget} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache_cpu = self.k_cache_cpu.to(self.device)
        self.v_cache_cpu = self.v_cache_cpu.to(self.device)

    def clear(self):
        self.kv_offset = 0
        self.prefilled_batch = 0
        self.is_prefill = True
        self.selected_indices.zero_()

    def get_kv_len(self):
        return self.kv_offset
    
    def get_svd(self, new_k_cache, layer_idx):
        """Dummy method for compatibility with ShadowAttention - not used in RandomOutlier"""
        pass
    
    def prefill_kv_cache(self, new_v_cache, layer_idx, key_states_roped, query=None):
        """Prefill KV cache with random outlier selection"""
        bsz, _, incoming, _ = new_v_cache.shape
        
        # Randomly select sparse_budget tokens from the prefill sequence
        total_tokens = self.kv_offset + incoming
        
        if total_tokens <= self.sparse_budget:
            # If total tokens less than budget, keep all
            selected_idx = torch.arange(total_tokens, device=self.device)
            # Pad with zeros if needed
            if selected_idx.shape[0] < self.sparse_budget:
                padding = torch.zeros(self.sparse_budget - selected_idx.shape[0], 
                                    device=self.device, dtype=torch.long)
                selected_idx = torch.cat([selected_idx, padding])
        else:
            # Randomly select sparse_budget tokens
            selected_idx = torch.randperm(total_tokens, device=self.device)[:self.sparse_budget]
        
        # Store selected indices for this layer
        self.selected_indices[layer_idx][:bsz, :self.sparse_budget] = selected_idx.unsqueeze(0).expand(bsz, -1)
        
        # Copy to CPU cache
        self.k_cache_cpu[layer_idx][:bsz, :, :total_tokens].copy_(key_states_roped.cpu())
        self.v_cache_cpu[layer_idx][:bsz, :, :total_tokens].copy_(new_v_cache.cpu())
        
        # Gather selected key and value states
        k_cache_full = self.k_cache_cpu[layer_idx][:bsz, :, :total_tokens].to(self.device)
        v_cache_full = self.v_cache_cpu[layer_idx][:bsz, :, :total_tokens].to(self.device)
        
        # Gather selected tokens
        key_states = torch.gather(k_cache_full, dim=2, 
                                index=selected_idx[:self.sparse_budget].view(1, 1, -1, 1).expand(bsz, self.num_key_value_heads, -1, self.head_dim))
        value_states = torch.gather(v_cache_full, dim=2, 
                                  index=selected_idx[:self.sparse_budget].view(1, 1, -1, 1).expand(bsz, self.num_key_value_heads, -1, self.head_dim))
        
        # Store in GPU buffer
        self.k_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget].copy_(key_states)
        self.v_cache_buffer[layer_idx][:bsz, :, :self.sparse_budget].copy_(value_states)
        
        self.is_prefill = False
        
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
        
        return key_states, value_states
    
    def get_retrieval_position_ids(self, layer_idx, query_states):
        """Get position IDs for retrieval - returns selected indices"""
        bsz = query_states.shape[0]
        # Return the selected indices (excluding the last position which is for new token)
        return self.selected_indices[layer_idx][:bsz, :self.sparse_budget-1]
    
    def get_value_cache(self, layer_idx, position_ids):
        """Get value cache at specified positions"""
        bsz = position_ids.shape[0]
        # Gather from CPU cache using position_ids
        max_pos = position_ids.max().item() + 1 if position_ids.numel() > 0 else 0
        v_cache_full = self.v_cache_cpu[layer_idx][:bsz, :, :max_pos].to(self.device)
        
        # Expand position_ids for all heads
        position_ids_expanded = position_ids.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.num_key_value_heads, -1, self.head_dim)
        
        # Gather values
        value_states = torch.gather(v_cache_full, dim=2, index=position_ids_expanded)
        return value_states
    
    def get_key_cache(self, layer_idx, position_ids, rope_func=None, cos_sin_cache=None):
        """Get key cache at specified positions with optional RoPE"""
        bsz = position_ids.shape[0]
        # Gather from CPU cache using position_ids
        max_pos = position_ids.max().item() + 1 if position_ids.numel() > 0 else 0
        k_cache_full = self.k_cache_cpu[layer_idx][:bsz, :, :max_pos].to(self.device)
        
        # Expand position_ids for all heads
        position_ids_expanded = position_ids.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.num_key_value_heads, -1, self.head_dim)
        
        # Gather keys
        key_states = torch.gather(k_cache_full, dim=2, index=position_ids_expanded)
        
        # Apply RoPE if provided
        if rope_func is not None and cos_sin_cache is not None:
            # Create position IDs for RoPE
            rope_pos_ids = position_ids.unsqueeze(-1)
            key_states = rope_func(key_states, key_states, rope_pos_ids)[0]
        
        return key_states
