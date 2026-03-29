from typing import Dict

from .full import FullAttention
from .shadow import ShadowAttention
from .streaming import StreamingAttention
from .quest import QuestAttention
from .tova import TovaAttention
from .base import AttentionBase

from ..kvcache import FullKVCache, ShadowKVCache, ShadowKVCache_CPU, ExperimentalKVCache, OptKVCache, QuestCache, TOVACache, KVCacheBase, StreamingKVCache, LocalDivCache, RandomOutlierCache

# Attention method mapping
Cache2Attn: Dict[KVCacheBase, AttentionBase] = {
    FullKVCache: FullAttention,
    ShadowKVCache: ShadowAttention,
    ShadowKVCache_CPU: ShadowAttention,
    OptKVCache: ShadowAttention,
    ExperimentalKVCache: ShadowAttention,
    QuestCache: QuestAttention,
    TOVACache: TovaAttention,
    StreamingKVCache: StreamingAttention,
    LocalDivCache: ShadowAttention,
    RandomOutlierCache: ShadowAttention
}