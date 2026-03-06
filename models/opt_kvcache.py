import torch
import math
from torch import nn
from utils import get_logger

import torch.nn.functional as F

logger = get_logger(__name__)

class OptKVCache:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=4,
        rank=160,
        ) -> None:

        logger.info("initializing OptKVCache")
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank
        self.local_chunk = 8
        self.outlier_chunk = 48

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        # self.selected_chunk_idx = None
        # self.v_cache_cpu = None
        # self.k_cache_buffer = None
        # self.v_cache_buffer = None

        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.copy_stream = torch.cuda.Stream()
        self.sink_size = 4

    def print_stats(self):
        print(f"OptimizedKVCache | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk} | sink_size {self.sink_size}")

    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
        self.SV[layer_idx].copy_(torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)) # [bsz, 8, 160, 128]
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def square_root_js_divergence(self, p: torch.Tensor, q: torch.Tensor):
        m = (p + q) / 2
        return torch.sqrt(
            0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
        )

    # def prefill_kv_cache_ours(self,
    #         new_v_cache :torch.Tensor,
    #         layer_idx :int,
    #         key_states_roped: torch.Tensor,
    #         query: torch.Tensor=None
    #         ):
        
    #     incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
    #     self.prefill = incoming
    #     self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

    #     # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
    #     self.chunks = incoming // self.chunk_size - self.local_chunk 
    #     self.select_sets = self.sparse_budget // self.chunk_size
        
    #     assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
    #     # store Post-RoPE k cache <prefill_local> to the cache
    #     self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
    #     self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
    #     self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

    #     key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
    #     landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]

    #     # x_norm = F.normalize(landmark_candidates, dim=-1)  # shape [N, D]

    #     # 步骤2：矩阵乘法计算点积（即余弦相似度）
    #     # cos_sim = torch.bmm(x_norm, x_norm.t())  # shape [N, N]
    #     # cos_sim = torch.einsum('bhid,bhjd->bhij', landmark_candidates, key_states_roped[:,:, -self.chunks:])

        
    #     # cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates, landmark_candidates.transpose(-1, -2), dim=-1) # [bsz, kv_heads, chunks, chunk_size]

    #     # outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
        
    #     j1 = F.softmax(landmark_candidates, dim=-1)
    #     j2 = F.softmax(key_states_roped[:,:, -self.chunks:], dim=-1)
    #     cos_sim = self.square_root_js_divergence(j1, j2)
       
    #     outlier_chunk_idx = cos_sim.topk(self.outlier_chunk, largest=False).indices
        
        
    #     # print(cos_sim.shape)
    #     # print(outlier_chunk_idx.shape)
        
    #     # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
    #     outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
    #     outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

    #     self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
    #     self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
    #     # store outlier_chunk to the cache
    #     self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
    #     self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

    #     # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
    #     # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
    #     # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
    #     all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
    #     mask = torch.ones_like(all_idx, dtype=torch.bool)
    #     mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
    #     rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

    #     # register rest_idxed landmarks to k_landmark
    #     self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

    #     if layer_idx == self.num_layers - 1:
    #         assert self.sparse_budget < incoming
    #         self.kv_offset += incoming

    def prefill_kv_cache(self,
                         new_v_cache: torch.Tensor,
                         layer_idx: int,
                         key_states_roped: torch.Tensor,
                         query: torch.Tensor = None):
        
        bsz, kv_heads, incoming, head_dim = new_v_cache.shape
        device = key_states_roped.device
        
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        # ==========================================
        # 1. 内存布局参数计算 (引入 Attention Sinks)
        # ==========================================
        sink_size = self.sink_size # 强制保留前 4 个 Token 稳定注意力
        seq_for_chunks = incoming - sink_size
        
        self.chunks = seq_for_chunks // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        # local chunks + 无法整除的余数
        self.prefill_local = seq_for_chunks - self.chunks * self.chunk_size 
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, \
            f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        # 划分 Buffer 边界
        sink_end = sink_size
        local_end = sink_end + self.prefill_local
        
        # Outlier 的存储起点和终点
        self.sparse_start = local_end + self.outlier_chunk * self.chunk_size
        # 动态检索的终点
        self.sparse_end = self.sparse_start + self.sparse_budget

        # ==========================================
        # 2. 写入静态 Cache (Sinks & Local)
        # ==========================================
        # 存储 Sinks (0 到 sink_size)
        self.k_cache_buffer[layer_idx][:, :, :sink_end].copy_(key_states_roped[:, :, :sink_size])
        self.v_cache_buffer[layer_idx][:, :, :sink_end].copy_(new_v_cache[:, :, :sink_size])
        
        # 存储 Local Cache (最新的 prefill_local 个 Token)
        self.k_cache_buffer[layer_idx][:, :, sink_end:local_end].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, sink_end:local_end].copy_(new_v_cache[:, :, -self.prefill_local:])

        # ==========================================
        # 3. 提取 Context Chunks 并计算 Outlier Score
        # ==========================================
        chunk_start_idx = sink_size
        chunk_end_idx = chunk_start_idx + self.chunks * self.chunk_size
        
        key_states_ctx = key_states_roped[:, :, chunk_start_idx:chunk_end_idx].view(
            bsz, kv_heads, self.chunks, self.chunk_size, head_dim
        )
        v_states_ctx = new_v_cache[:, :, chunk_start_idx:chunk_end_idx].view(
            bsz, kv_heads, self.chunks, self.chunk_size, head_dim
        )

        # 改进：抗 RoPE 干扰的内部方差评估
        chunk_mean = key_states_ctx.mean(dim=-2, keepdim=True)
        chunk_variance = torch.sum((key_states_ctx - chunk_mean) ** 2, dim=(-1, -2)) # [bsz, kv_heads, chunks]
        
        # 改进：跨头投票 (Cross-Head Voting)，统一 Outlier 索引
        unified_chunk_score = chunk_variance.mean(dim=1) # [bsz, chunks]
        outlier_chunk_idx = unified_chunk_score.topk(self.outlier_chunk, dim=-1, largest=True).indices
        outlier_chunk_idx, _ = outlier_chunk_idx.sort(dim=-1) # 维持物理时间顺序

        # 构造共享 Gather 索引
        gather_idx = outlier_chunk_idx.view(bsz, 1, self.outlier_chunk, 1, 1).expand(
            -1, kv_heads, -1, self.chunk_size, head_dim
        )
        
        outlier_shape = (bsz, kv_heads, self.outlier_chunk * self.chunk_size, head_dim)
        outlier_chunk_k = key_states_ctx.gather(dim=2, index=gather_idx).view(outlier_shape)
        outlier_chunk_v = v_states_ctx.gather(dim=2, index=gather_idx).view(outlier_shape)

        # ==========================================
        # 4. 存储 Outlier Chunk 并注册剩余 Landmark
        # ==========================================
        self.k_cache_buffer[layer_idx][:, :, local_end:self.sparse_start].copy_(outlier_chunk_k)
        self.v_cache_buffer[layer_idx][:, :, local_end:self.sparse_start].copy_(outlier_chunk_v)

        # 获取被抛弃的 Chunks 用于生成 Landmark
        all_idx = torch.arange(self.chunks, device=device).expand(bsz, -1)
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(bsz, -1)

        rest_gather_idx = rest_idx.view(bsz, 1, -1, 1, 1).expand(-1, kv_heads, -1, self.chunk_size, head_dim)
        rest_k_landmarks = key_states_ctx.gather(dim=2, index=rest_gather_idx).mean(dim=-2)
        
        # 注册 Landmark
        self.register_k_landmark(rest_k_landmarks, rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2] # 1
        
        # 1. 整理形状并计算基础 Attention
        query_states_view = query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim)
        k_landmark_transposed = self.k_landmark[layer_idx].transpose(2, 3)
        
        # Einsum 计算得到 [bsz, heads, groups, q_len, chunks]
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states_view, k_landmark_transposed).squeeze(2) / math.sqrt(self.head_dim)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) 
        
        # 2. 消除 Query Length 维度: [bsz, heads, groups, chunks]
        chunk_attn = chunk_attn.sum(dim=-2) 
        
        # 3. 修复点：消除 Groups 维度 (如果使用了 GQA)
        # 取每个 Group 中的最大注意力得分，形状变为 [bsz, heads, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) 
            
        # 4. 跨头投票 (Cross-Head Voting)
        # 沿着 heads 维度取平均，得到所有头统一的 chunk 得分: [bsz, chunks]
        unified_chunk_attn = chunk_attn.mean(dim=1) 
        
        # 5. 获取 Top-K 索引并按时间顺序重排
        merged_results = torch.topk(unified_chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, select_sets]
        merged_results, _ = merged_results.sort(dim=-1) # 重排，保证显存访问连续与 RoPE 相位顺序
        
        # 6. 扩展回多头形状以兼容 gather 操作
        # 此时 merged_results 是 [bsz, select_sets]，unsqueeze 之后是 [bsz, 1, select_sets]
        # expand 之后变为 [bsz, num_key_value_heads, select_sets] -> 完美契合 3 维参数
        merged_results = merged_results.unsqueeze(1).expand(-1, self.num_key_value_heads, -1)

        # 7. 提取真实 Landmark 物理索引并记录
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results)
        self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

        # 8. 补偿 Attention Sinks 造成的物理位置偏移
        sink_size = self.sink_size # 必须与 Prefill 阶段强行保留的 Token 数量完全一致
        
        # 映射回 Token 级的物理 Position IDs
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + \
                        torch.arange(self.chunk_size, device=chunk_attn.device).view(1, 1, 1, -1))
        
        position_ids = position_ids.view(self.batch_size, self.num_key_value_heads, -1)
        
        # 加上偏移量
        position_ids = position_ids + sink_size 

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        # 此处 position_ids 已包含 sink_size 偏移，从 CPU 拉取完全精准
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        
        # 填入专门的动态检索 Buffer 区
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        u = self.U[layer_idx] 
        sv = self.SV[layer_idx] 

        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1)) 
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1) 
        U_head = torch.gather(u_expand, 2, index_expanded)

        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # 这里的 RoPE 会根据补偿后的 position_ids 进行正确的相位注入
        result = rope_func(result, position_ids)

        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def update_kv_cache(self, 
                        new_k_cache: torch.Tensor,
                        new_v_cache: torch.Tensor,
                        layer_idx: int):

        incoming = new_k_cache.shape[-2]
        start_idx = self.sparse_end + self.gen_offset
        end_idx = start_idx + incoming
        
        # 将新生成的 Token 追加到 Buffer 末尾
        self.v_cache_buffer[layer_idx][:, :, start_idx:end_idx].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, start_idx:end_idx].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming






            


    # def prefill_kv_cache(self,
    #         new_v_cache :torch.Tensor,
    #         layer_idx :int,
    #         key_states_roped: torch.Tensor,
    #         query: torch.Tensor=None
    #         ):
        
    #     incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
    #     self.prefill = incoming
    #     self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

    #     # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
    #     self.chunks = incoming // self.chunk_size - self.local_chunk 
    #     self.select_sets = self.sparse_budget // self.chunk_size
        
    #     assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
    #     # store Post-RoPE k cache <prefill_local> to the cache
    #     self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
    #     self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
    #     self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

    #     key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
    #     landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]



    #     # value_states_roped_ctx = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
    #     # landmark_candidates_v = value_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]

    #     # compute the cos similarity between it and the original key cache
    #     cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]

        
    #     # # # compute the cos similarity between it and the original key cache
    #     # cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), value_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
    #     # # cos_sim2 = torch.nn.functional.cosine_similarity(landmark_candidates_v.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), value_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
    #     # # outlier_chunk_idx2 = cos_sim2.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices

    #     #avg_pooling

    #     # key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
    #     # landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]



    #     # value_states_roped_ctx = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
    #     # landmark_candidates_v = value_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]

    #     # compute the cos similarity between it and the original key cache
    #     # cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), value_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]

        









    #     # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
    #     outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices

    #     # print(cos_sim.shape)
    #     # print(outlier_chunk_idx.shape)
    #     # print(outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim).shape)
    #     # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
    #     outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
    #     outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        
    #     self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
    #     self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
    #     # print("shape::::", outlier_chunk_k_cache.shape)
    #     # print("shape::::", outlier_chunk_v_cache.shape, self.prefill_local - self.sparse_start)

    #     # store outlier_chunk to the cache
    #     self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
    #     self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

    #     # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
    #     # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
    #     # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
    #     all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
    #     mask = torch.ones_like(all_idx, dtype=torch.bool)
    #     mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
    #     rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)
       
    #     # register rest_idxed landmarks to k_landmark
    #     self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

    #     if layer_idx == self.num_layers - 1:
    #         assert self.sparse_budget < incoming
    #         self.kv_offset += incoming

    # def get_retrieval_position_ids(self, layer_idx, query_states):
    #     # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
    #     # chunk_attn: [bsz, 32, window_size, chunks]
    #     self.incoming_q_len = query_states.shape[-2] # 1
    #     # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
    #     # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
    #     chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), self.k_landmark[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(128)
    #     chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
    #     chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
    #     if self.num_key_value_groups > 1:
    #         chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
    #     merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]

    #     # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
    #     selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

    #     # this is chunk idx, which can be used to offload value cache and decide if the cache hits
    #     self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

    #     position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

    #     return position_ids
        
    # def get_value_cache(self, layer_idx, position_ids):
    #     # gather value cache
    #     value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
    #     # print(value_.shape)
    #     # print(self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].shape)
    #     self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
    #     gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

    #     return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    # def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
    #     # gather key cache and rope them
    #     u = self.U[layer_idx] # [bsz, 128k, rank]
    #     sv = self.SV[layer_idx] # [bsz, 8, rank, 128]

    #     # indexing, [bsz, 8, sparse_budget, rank]
    #     index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1)) # [bsz, 8, sparse_budget, rank]
    #     u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1) # [bsz, 8, 128k, rank]
    #     U_head = torch.gather(u_expand, 2, index_expanded)

    #     # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
    #     result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

    #     # rope the key cache
    #     result = rope_func(result, position_ids)

    #     # send to buffer
    #     self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
    #     gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

    #     return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    # def update_kv_cache(self, 
    #         new_k_cache :torch.Tensor,
    #         new_v_cache :torch.Tensor,
    #         layer_idx :int,
    #         ):

    #     incoming = new_k_cache.shape[-2]
    #     self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
    #     self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

    #     if layer_idx == self.num_layers - 1:
    #         self.kv_offset += incoming
    #         self.gen_offset += incoming


    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.selected_chunk_idx.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset

