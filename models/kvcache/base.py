from abc import ABC, abstractmethod

import torch


class KVCacheBase(ABC):
    """Base KVCache"""

    @abstractmethod
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):
        """Append new kvcache to pool"""
        pass

    @abstractmethod
    def H2D(self):
        pass
    
    @abstractmethod
    def print_stats(self):
        """Print current stats of this kvcache pool"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all kvcache in the pool"""
        pass

    @abstractmethod
    def get_kv_len(self):
        """Get current kvcache pool length (context length)"""
        pass