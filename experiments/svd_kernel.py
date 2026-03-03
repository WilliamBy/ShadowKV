import torch
import torch.nn.functional as F
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def generate_random_matrix(rows: int, cols: int, device: str = 'cuda') -> torch.Tensor:
    """生成随机矩阵用于测试"""
    return torch.randn(rows, cols, device=device, dtype=torch.float32)

def benchmark_svd(matrix: torch.Tensor, svd_func, name: str, warmup: int = 5, repeats: int = 20) -> Dict:
    """测试单个 SVD 实现的性能"""
    # Warmup
    for _ in range(warmup):
        _ = svd_func(matrix)
    
    # Synchronize CUDA
    if matrix.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = svd_func(matrix)
        end = time.perf_counter()
        times.append(end - start)
        if matrix.is_cuda:
            torch.cuda.synchronize()
    
    times = torch.tensor(times)
    return {
        'name': name,
        'mean': times.mean().item() * 1000,  # ms
        'std': times.std().item() * 1000,    # ms
        'min': times.min().item() * 1000,    # ms
        'max': times.max().item() * 1000,    # ms
    }

def test_svd_implementations(matrix: torch.Tensor) -> List[Dict]:
    """测试所有 SVD 实现"""

    results = []
    
    def torch_svd(mat):
        U, S, V = torch.svd(mat, )
        return U, S, V
    
    result = benchmark_svd(matrix, torch_svd, "torch.svd")
    results.append(result)
    
    def linalg_svd(mat):
        U, S, V = torch.linalg.svd(mat)
        return U, S, V
    
    result = benchmark_svd(matrix, linalg_svd, "linalg.svd")
    results.append(result)

    def gesvdj(mat):
        U, S, V = torch.linalg.svd(mat, driver='gesvdj')
        return U, S, V
    
    result = benchmark_svd(matrix, gesvdj, "linalg.svd(driver='gesvdj')")
    results.append(result)
    
    def gesvd(mat):
        U, S, V = torch.linalg.svd(mat, driver='gesvd')
        return U, S, V
    
    result = benchmark_svd(matrix, gesvd, "linalg.svd(driver='gesvd')")
    results.append(result)
    
    def gesvda(mat):
        U, S, V = torch.linalg.svd(mat, driver='gesvda')
        return U, S, V
    
    result = benchmark_svd(matrix, gesvda, "linalg.gesvda")
    results.append(result)

    return results

def print_results(results: List[Dict], matrix_shape: tuple):
    """打印结果"""
    print(f"\n{'='*80}")
    print(f"SVD 性能测试 - 矩阵形状: {matrix_shape}")
    print(f"{'='*80}")
    print(f"{'方法':<30} {'平均时间(ms)':<15} {'标准差(ms)':<15} {'最小值(ms)':<15} {'最大值(ms)':<15}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['name']:<30} {r['mean']:<15.4f} {r['std']:<15.4f} {r['min']:<15.4f} {r['max']:<15.4f}")
    
    # 找出最快的方法
    fastest = min(results, key=lambda x: x['mean'])
    speedup = [r['mean'] / fastest['mean'] for r in results]
    
    print(f"\n{'='*80}")
    print(f"性能对比 (相对于 {fastest['name']}):")
    print(f"{'='*80}")
    for i, r in enumerate(results):
        print(f"{r['name']:<30} {speedup[i]:.2f}x")

def run_comprehensive_test():
    """运行全面的测试"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试不同的矩阵大小
    test_cases = [(int(x), 1024) for x in np.linspace(4096, 32768, 5)]
    
    
    all_results = {}
    
    for rows, cols in test_cases:
        print(f"\n{'#'*80}")
        print(f"测试矩阵: {rows} x {cols}")
        print(f"{'#'*80}")
        
        matrix = generate_random_matrix(rows, cols, device)
        results = test_svd_implementations(matrix)
        print_results(results, (rows, cols))
        
        all_results[(rows, cols)] = results
    
    return all_results

def plot_results(all_results: Dict):
    """可视化结果"""
    if not all_results:
        return
    
    fig, axes = plt.subplots(len(all_results), 1, figsize=(12, 4 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, ((rows, cols), results) in enumerate(all_results.items()):
        ax = axes[idx]
        names = [r['name'] for r in results]
        means = [r['mean'] for r in results]
        stds = [r['std'] for r in results]
        
        x_pos = np.arange(len(names))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xlabel('SVD method')
        ax.set_ylabel('elipse')
        ax.set_title(f'matrix size: {rows} x {cols}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/data/bwl/ShadowKV/experiments/svd_benchmark_results.png', dpi=150)
    print(f"\n图表已保存到: /data/bwl/ShadowKV/experiments/svd_benchmark_results.png")

if __name__ == "__main__":
    # 运行测试
    all_results = run_comprehensive_test()
    
    # 绘制结果
    try:
        plot_results(all_results)
    except Exception as e:
        print(f"绘图失败: {e}")

