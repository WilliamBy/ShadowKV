from .kv_base import KVCacheBase

from .shadowkv import ShadowKVCache, ShadowKVCache_CPU
from .exp import ExperimentalKVCache, ExperimentalKVCache_CPU
from .opt import OptKVCache
from .quest import QuestCache
from .full import FullKVCache
from .loki import LokiCache

method2kvcache = {
    "full": FullKVCache,
    "shadowkv": ShadowKVCache,
    "shadowkv_cpu": ShadowKVCache_CPU,
    "experimental": ExperimentalKVCache,
    "experimental_cpu": ExperimentalKVCache_CPU,
    "optimized": OptKVCache,
    "quest": QuestCache,
    "loki": LokiCache
}