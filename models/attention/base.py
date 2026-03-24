import abc

import torch

class AttentionBase(abc.ABC):

    def __init__(self, kv_cache, num_key_value_groups, head_dim, hidden_size, rope_f=None, rope_f_single=None, cos_sin_cache=None):
        self.apply_rotary_pos_emb = rope_f
        self.apply_rotary_pos_emb_single = rope_f_single
        self.cos_sin_cache = cos_sin_cache
        self.kv_cache = kv_cache
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.minference = None
        self.minference_parttern = None

    @abc.abstractmethod
    def prefill(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, position_ids, layer_idx):
        pass

    @abc.abstractmethod
    def decode(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, position_ids, layer_idx):
        pass