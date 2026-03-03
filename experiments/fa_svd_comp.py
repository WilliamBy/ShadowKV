import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
import numpy as np

# 尝试导入 Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    FLASH_ATTENTION_AVAILABLE = True
    print("✓ Flash Attention 可用")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("✗ Flash Attention 不可用，将跳过相关测试")

def generate_random_matrix(seq_len: int, hidden_dim: int, device: str = 'cuda') -> torch.Tensor:
    """生成随机矩阵用于测试"""
    return torch.randn(seq_len, hidden_dim, device=device, dtype=torch.float16)

def benchmark_torch_svd(matrix: torch.Tensor, warmup: int = 5, repeats: int = 20) -> Dict:
    """测试 torch.linalg.svd 的性能"""
    print(f"\n测试 torch.linalg.svd...")
    print(f"  矩阵形状: {matrix.shape}")
    print(f"  数据类型: {matrix.dtype}")
    
    # 转换为 float32（SVD 通常使用 float32）
    matrix_f32 = matrix.float()
    
    def svd_func(mat):
        U, S, V = torch.linalg.svd(mat, full_matrices=True)
        return U, S, V
    
    # Warmup
    for _ in range(warmup):
        _ = svd_func(matrix_f32)
    
    if matrix.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    memory_allocated = []
    
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() if matrix.is_cuda else 0
        
        start = time.perf_counter()
        U, S, V = svd_func(matrix_f32)
        if matrix.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        peak_mem = torch.cuda.max_memory_allocated() if matrix.is_cuda else 0
        memory_allocated.append((peak_mem - start_mem) / 1024**2)  # MB
        times.append((end - start) * 1000)  # ms
    
    times = torch.tensor(times)
    memory_allocated = torch.tensor(memory_allocated)
    
    return {
        'name': 'torch.linalg.svd',
        'mean_time_ms': times.mean().item(),
        'std_time_ms': times.std().item(),
        'min_time_ms': times.min().item(),
        'max_time_ms': times.max().item(),
        'mean_memory_mb': memory_allocated.mean().item(),
        'peak_memory_mb': memory_allocated.max().item(),
        'output_shape': (U.shape, S.shape, V.shape),
    }

def benchmark_flash_attention(matrix: torch.Tensor, warmup: int = 5, repeats: int = 20) -> Dict:
    """测试 Flash Attention 的性能"""
    if not FLASH_ATTENTION_AVAILABLE:
        return None
    
    print(f"\n测试 Flash Attention...")
    print(f"  矩阵形状: {matrix.shape}")
    print(f"  数据类型: {matrix.dtype}")
    
    # Flash Attention 需要 (batch, seqlen, hidden_dim) 格式
    # 我们将矩阵作为 Q, K, V
    batch_size = 1
    seq_len = matrix.shape[0]
    hidden_dim = matrix.shape[1]
    
    # 创建 Q, K, V（使用相同的矩阵）
    q = matrix.unsqueeze(0)  # (1, seq_len, hidden_dim)
    k = matrix.unsqueeze(0)
    v = matrix.unsqueeze(0)
    
    def flash_attn_func_wrapper(q, k, v):
        return flash_attn_func(q, k, v)
    
    # Warmup
    for _ in range(warmup):
        _ = flash_attn_func_wrapper(q, k, v)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    memory_allocated = []
    
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        start = time.perf_counter()
        output = flash_attn_func_wrapper(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        peak_mem = torch.cuda.max_memory_allocated()
        memory_allocated.append((peak_mem - start_mem) / 1024**2)  # MB
        times.append((end - start) * 1000)  # ms
    
    times = torch.tensor(times)
    memory_allocated = torch.tensor(memory_allocated)
    
    return {
        'name': 'Flash Attention',
        'mean_time_ms': times.mean().item(),
        'std_time_ms': times.std().item(),
        'min_time_ms': times.min().item(),
        'max_time_ms': times.max().item(),
        'mean_memory_mb': memory_allocated.mean().item(),
        'peak_memory_mb': memory_allocated.max().item(),
        'output_shape': output.shape,
    }

def print_comparison_results(torch_result: Dict, fa_result: Dict = None):
    """打印比较结果"""
    print(f"\n{'='*80}")
    print(f"性能比较结果")
    print(f"{'='*80}")
    
    print(f"\n{'方法':<25} {'平均时间(ms)':<15} {'标准差(ms)':<15} {'峰值内存(MB)':<15}")
    print(f"{'-'*80}")
    
    print(f"{torch_result['name']:<25} {torch_result['mean_time_ms']:<15.4f} "
          f"{torch_result['std_time_ms']:<15.4f} {torch_result['peak_memory_mb']:<15.2f}")
    
    if fa_result:
        print(f"{fa_result['name']:<25} {fa_result['mean_time_ms']:<15.4f} "
              f"{fa_result['std_time_ms']:<15.4f} {fa_result['peak_memory_mb']:<15.2f}")
        
        # 计算加速比
        speedup = torch_result['mean_time_ms'] / fa_result['mean_time_ms']
        memory_ratio = fa_result['peak_memory_mb'] / torch_result['peak_memory_mb']
        
        print(f"\n{'='*80}")
        print(f"对比分析:")
        print(f"{'='*80}")
        print(f"Flash Attention 速度比 SVD 快 {speedup:.2f}x")
        print(f"Flash Attention 内存使用是 SVD 的 {memory_ratio:.2f}x")
        
        if memory_ratio < 1:
            print(f"Flash Attention 节省 {(1 - memory_ratio) * 100:.1f}% 内存")
        else:
            print(f"Flash Attention 多使用 {(memory_ratio - 1) * 100:.1f}% 内存")
    
    print(f"\n{'='*80}")
    print(f"输出形状:")
    print(f"{'='*80}")
    print(f"torch.linalg.svd: U={torch_result['output_shape'][0]}, "
          f"S={torch_result['output_shape'][1]}, V={torch_result['output_shape'][2]}")
    if fa_result:
        print(f"Flash Attention: {fa_result['output_shape']}")

def run_comparison(seq_len: int = 32000, hidden_dim: int = 1024):
    """运行比较测试"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"{'='*80}")
    print(f"Flash Attention vs torch.linalg.svd 性能比较")
    print(f"{'='*80}")
    print(f"\n配置:")
    print(f"  设备: {device}")
    print(f"  序列长度: {seq_len}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  矩阵大小: {seq_len} x {hidden_dim}")
    
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 生成测试矩阵
    print(f"\n生成随机矩阵...")
    matrix = generate_random_matrix(seq_len, hidden_dim, device)
    
    # 测试 torch.linalg.svd
    torch_result = benchmark_torch_svd(matrix)
    
    # 测试 Flash Attention
    fa_result = None
    if FLASH_ATTENTION_AVAILABLE:
        fa_result = benchmark_flash_attention(matrix)
    
    # 打印结果
    print_comparison_results(torch_result, fa_result)
    
    # 额外的分析
    print(f"\n{'='*80}")
    print(f"算法特性对比:")
    print(f"{'='*80}")
    print(f"""
torch.linalg.svd:
  - 用途: 矩阵分解，用于降维、特征提取等
  - 输出: U, S, V 三个矩阵
  - 计算复杂度: O(min(mn², m²n))
  - 内存需求: 较高（需要存储三个分解矩阵）
  - 适用场景: 数据压缩、特征分析、矩阵求逆等

Flash Attention:
  - 用途: 高效的注意力机制计算
  - 输出: 注意力输出 (batch, seq_len, hidden_dim)
  - 计算复杂度: O(seq_len² * hidden_dim)
  - 内存需求: 优化的注意力计算，使用分块技术
  - 适用场景: Transformer 模型、长序列建模等

注意: 这两个操作解决的是不同的问题，不能直接替代使用。
此比较仅用于展示它们在处理相同大小矩阵时的性能特征。
    """)
    
    return torch_result, fa_result

def run_multiple_sizes():
    """测试多个矩阵大小"""
    test_cases = [
        (8192, 1024, "8K"),
        (16000, 1024, "16K"),
        (32000, 1024, "32K"),
        (64000, 1024, "64K"),
    ]
    
    print(f"\n{'#'*80}")
    print(f"多尺寸性能测试")
    print(f"{'#'*80}")
    
    results = []
    
    for seq_len, hidden_dim, name in test_cases:
        print(f"\n{'='*80}")
        print(f"测试: {name} 序列长度 ({seq_len} x {hidden_dim})")
        print(f"{'='*80}")
        
        try:
            torch_result, fa_result = run_comparison(seq_len, hidden_dim)
            results.append({
                'name': name,
                'seq_len': seq_len,
                'torch_result': torch_result,
                'fa_result': fa_result
            })
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠ 显存不足，跳过 {name} 测试")
                break
            else:
                raise
    
    # 总结
    print(f"\n{'#'*80}")
    print(f"测试总结")
    print(f"{'#'*80}")
    
    print(f"\n{'序列长度':<15} {'SVD时间(ms)':<15} {'FA时间(ms)':<15} {'加速比':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        seq_len = r['seq_len']
        torch_time = r['torch_result']['mean_time_ms']
        fa_time = r['fa_result']['mean_time_ms'] if r['fa_result'] else float('inf')
        speedup = torch_time / fa_time if r['fa_result'] else 'N/A'
        
        print(f"{seq_len:<15} {torch_time:<15.2f} {fa_time:<15.2f} {speedup:<10}")

if __name__ == "__main__":
    # 单次测试
    print("运行单次测试 (32K x 1024)...")
    torch_result, fa_result = run_comparison(32000, 1024)
    
    # 多尺寸测试
    print("\n\n" + "="*80)
    print("是否运行多尺寸测试？(y/n)")
    print("注意: 这将需要更多时间和显存")
    print("="*80)
    # 取消注释下面的注释来运行多尺寸测试
    # run_multiple_sizes()
