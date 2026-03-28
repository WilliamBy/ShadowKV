from .base import KVCacheBase

from .shadow import ShadowKVCache, ShadowKVCache_CPU
from .exp import ExperimentalKVCache, ExperimentalKVCache_CPU
from .opt import OptKVCache
from .streaming import StreamingKVCache
from .quest import QuestCache
from .full import FullKVCache
from .loki import LokiCache
from .tova import TOVACache
from .key_heuristic import LocalDivCache

method2kvcache = {
    "full": FullKVCache,
    "shadowkv": ShadowKVCache,
    "shadowkv_cpu": ShadowKVCache_CPU,
    "experimental": ExperimentalKVCache,
    "experimental_cpu": ExperimentalKVCache_CPU,
    "optimized": OptKVCache,
    "streaming": StreamingKVCache,
    "quest": QuestCache,
    "loki": LokiCache,
    "tova": TOVACache,
    "local_div": LocalDivCache
}