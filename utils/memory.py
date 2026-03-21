"""GPU & CPU memory management tools"""
import gc
import torch


def gc_and_sync():
    """collect memory and sync cuda steam"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_mem_stat():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    max_allocated = torch.cuda.max_memory_allocated()
    print("+++ GPU Memory Statistics +++")
    print(f"current memory used: {allocated}/{reserved} GB")
    print(f"max memory allocated: {max_allocated} GB")
