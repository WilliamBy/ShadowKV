import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def demonstrate_warmup_effect():
    """演示预热对性能测试的影响"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建测试矩阵
    matrix = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    
    # 测试 SVD 函数
    def svd_func(mat):
        U, S, V = torch.linalg.svd(mat)
        return U, S, V
    
    # 记录每次运行的时间
    times = []
    
    print("\n运行 30 次 SVD，记录每次的时间...")
    print("-" * 60)
    
    for i in range(30):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        U, S, V = svd_func(matrix)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # 转换为毫秒
        times.append(elapsed)
        
        if i < 5:
            print(f"第 {i+1:2d} 次 (预热): {elapsed:8.4f} ms")
        elif i == 5:
            print(f"第 {i+1:2d} 次 (正式): {elapsed:8.4f} ms  <-- 预热结束")
        elif i < 10:
            print(f"第 {i+1:2d} 次 (正式): {elapsed:8.4f} ms")
    
    times = np.array(times)
    
    # 分析结果
    print("\n" + "=" * 60)
    print("统计分析:")
    print("=" * 60)
    
    warmup_times = times[:5]
    stable_times = times[5:]
    
    print(f"\n预热阶段 (前5次):")
    print(f"  平均时间: {warmup_times.mean():.4f} ms")
    print(f"  标准差:   {warmup_times.std():.4f} ms")
    print(f"  最大值:   {warmup_times.max():.4f} ms")
    print(f"  最小值:   {warmup_times.min():.4f} ms")
    
    print(f"\n稳定阶段 (后25次):")
    print(f"  平均时间: {stable_times.mean():.4f} ms")
    print(f"  标准差:   {stable_times.std():.4f} ms")
    print(f"  最大值:   {stable_times.max():.4f} ms")
    print(f"  最小值:   {stable_times.min():.4f} ms")
    
    print(f"\n预热影响:")
    print(f"  预热阶段比稳定阶段慢 {(warmup_times.mean() / stable_times.mean() - 1) * 100:.2f}%")
    print(f"  预热阶段的标准差是稳定阶段的 {warmup_times.std() / stable_times.std():.2f}x")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: 所有运行的时间序列
    ax1.plot(times, marker='o', linewidth=2, markersize=6)
    ax1.axvline(x=4.5, color='r', linestyle='--', linewidth=2, label='预热结束')
    ax1.fill_between(range(5), 0, times.max(), alpha=0.3, color='red', label='预热区域')
    ax1.set_xlabel('运行次数', fontsize=12)
    ax1.set_ylabel('时间 (ms)', fontsize=12)
    ax1.set_title('SVD 性能测试 - 预热效果演示', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 预热阶段 vs 稳定阶段的对比
    positions = [0, 1]
    data = [warmup_times, stable_times]
    colors = ['#ff7f0e', '#2ca02c']
    labels = ['预热阶段\n(前5次)', '稳定阶段\n(后25次)']
    
    bp = ax2.boxplot(data, positions=positions, labels=labels, patch_artist=True,
                     medianprops=dict(linewidth=2, color='black'),
                     boxprops=dict(linewidth=2))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('时间 (ms)', fontsize=12)
    ax2.set_title('预热阶段 vs 稳定阶段', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for i, (d, pos) in enumerate(zip(data, positions)):
        ax2.text(pos, d.mean(), f'{d.mean():.2f}ms', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/data/bwl/ShadowKV/experiments/warmup_effect.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: /data/bwl/ShadowKV/experiments/warmup_effect.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print("""
1. 第一次运行通常最慢（GPU kernel 初始化、内存分配）
2. 前几次运行时间波动较大（缓存预热、编译优化）
3. 预热后性能趋于稳定，标准差显著降低
4. 不进行预热会导致性能测试结果不准确

建议：
- 预热次数：3-10 次（取决于操作复杂度）
- 测试次数：20-100 次（获得稳定的统计数据）
- 忽略预热阶段的数据，只统计稳定阶段
    """)

if __name__ == "__main__":
    demonstrate_warmup_effect()
