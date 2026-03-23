from models.base import LLM
import abc

import torch

class AttentionBase(abc.ABC):
    @abc.abstractmethod
    def prefill(self, llm: LLM, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, position_ids, layer_idx, kv_cache, minference=None, minference_parttern=None):
        pass

    @abc.abstractmethod
    def decode(self, llm: LLM, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, position_ids, layer_idx, kv_cache, minference=None, minference_parttern=None):
        pass