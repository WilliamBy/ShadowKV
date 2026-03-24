from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch

class Landmarker(ABC):
    def __init__(self, keys_for_chunk, chunk_size, chunks):
        if keys_for_chunk.ndim != 4:
            raise ValueError("Input tensor shape dimension number should be 4")
        self.bsz, self.kv_head, self.origin_length, self.head_dim = keys_for_chunk.size()
        self.csz, self.chunks = chunk_size, chunks
        self.length = chunk_size * chunks
        if self.length > self.origin_length:
            raise ValueError("Chunks size too big")
        self.rest_length = self.origin_length - self.length
        self.keys = keys_for_chunk[:, :, :self.length, :]
        self.rest_keys = keys_for_chunk[:, :, -self.rest_length:, :]
        self.chunk_keys = self.keys.view(self.bsz, self.kv_head, self.chunks, self.csz, self.head_dim)

    @abstractmethod
    def landmark_candidates(self):
        pass

    @abstractmethod
    def outlier_idx(self, n_outliers, landmarks):
        pass

class ShadowLandmarker(Landmarker):

    def landmark_candidates(self):
        # (bsz, kv_head, chunks, head_dim)
        landmarks = torch.mean(self.chunk_keys, dim=-2)
        self.landmark_shape = landmarks.size()
        return landmarks

    def outlier_idx(self, n_outliers, landmarks):
        chunk_keys = self.chunk_keys
        assert self.landmark_shape == landmarks.shape
        
        # Compute cosine similarity between landmark candidates and original key cache
        # Expand landmark_candidates to match key_states_roped_ctx for comparison
        cos_sim = F.cosine_similarity(
            landmarks.unsqueeze(3).expand(-1, -1, -1, self.csz, -1),
            chunk_keys,
            dim=-1
        )  # [bsz, kv_heads, chunks, chunk_size]
        
        # Get outlier_chunk idx for each head by finding minimum cosine similarity
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(n_outliers, largest=False).indices
        
        # [bsz, kv_heads, outlier_chunk]
        return outlier_chunk_idx

    @staticmethod
    def shrink_landmarks(outlier_chunk_idx, landmarks):
        return landmarks.gather(outlier_chunk_idx)

    def gather_outlier_keys(self, outlier_chunk_idx):
        return self.chunk_keys.gather(outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.csz, self.head_dim), dim=2)
