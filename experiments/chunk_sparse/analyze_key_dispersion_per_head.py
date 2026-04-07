#!/usr/bin/env python3
"""
分析每个Attention Head的Key离散程度

目标：对不同attention head的key分别进行离散程度分析
预期发现：
- 不同head可能有不同的离散程度模式
- 某些head可能更"专注"（低离散度）
- 某些head可能更"发散"（高离散度）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import argparse
import sys
import os
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# 设置seaborn样式
sns.set_style("whitegrid")

class PerHeadKeyDispersionAnalyzer:
    """收集并分析每个attention head的key离散程度"""
    
    def __init__(self, model, tokenizer, chunk_size=128, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.device = device
        
        # 存储数据：{layer_idx: {head_idx: [keys]}}
        self.per_head_keys = defaultdict(lambda: defaultdict(list))
        
        # 注册hook
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册forward hook来收集每个head的key"""
        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                # Llama-3的attention输出格式: (hidden_states, attention_weights, cache)
                if isinstance(output, tuple) and len(output) >= 3:
                    cache = output[2]
                    if hasattr(cache, 'key_cache') and len(cache.key_cache) > layer_idx:
                        key = cache.key_cache[layer_idx]  # [batch_size, num_heads, seq_len, head_dim]
                        
                        # 只收集第一个样本
                        key = key[0].detach().cpu()  # [num_heads, seq_len, head_dim]
                        
                        # 分别存储每个head的key
                        num_heads, seq_len, head_dim = key.shape
                        for head_idx in range(num_heads):
                            head_key = key[head_idx]  # [seq_len, head_dim]
                            self.per_head_keys[layer_idx][head_idx].append(head_key)
            return hook
        
        # 为每个attention层注册hook
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_dispersion(self, keys):
        """
        计算key的块内离散程度
        
        参数:
            keys: [seq_len, head_dim]
        
        返回:
            chunk_stats: 每个块的统计信息
        """
        seq_len, head_dim = keys.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        chunk_centroids = []
        chunk_dispersions = []
        chunk_sizes = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            chunk_keys = keys[start_idx:end_idx]  # [chunk_size, head_dim]
            chunk_sizes.append(chunk_keys.shape[0])
            
            # 计算质心
            centroid = chunk_keys.mean(dim=0, keepdim=True)  # [1, head_dim]
            chunk_centroids.append(centroid)
            
            # 计算离散程度（平均欧氏距离）
            distances = torch.norm(chunk_keys - centroid, dim=1)  # [chunk_size]
            dispersion = distances.mean().item()
            chunk_dispersions.append(dispersion)
        
        return {
            'centroids': torch.cat(chunk_centroids, dim=0) if chunk_centroids else torch.tensor([]),
            'dispersions': torch.tensor(chunk_dispersions),
            'sizes': torch.tensor(chunk_sizes),
        }
    
    @torch.no_grad()
    def analyze_generation(self, input_ids, max_new_tokens=64):
        """
        分析生成过程中的key离散程度（per-head）
        
        参数:
            input_ids: [seq_len] 或 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
        """
        # 确保有batch维度
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        
        input_ids = input_ids.to(self.device)
        
        # 清空之前的收集
        self.per_head_keys.clear()
        
        # 逐token生成并收集key
        current_ids = input_ids.clone()
        generated_ids = []
        
        print(f"开始生成，目标生成 {max_new_tokens} 个tokens...")
        
        for step in tqdm(range(max_new_tokens), desc="生成进度"):
            # 前向传播
            outputs = self.model(input_ids=current_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # 贪婪解码
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            generated_ids.append(next_token.item())
            
            # 添加到序列
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # 检查是否达到EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                print(f"在步骤 {step} 遇到EOS token，停止生成")
                break
        
        print(f"实际生成了 {len(generated_ids)} 个tokens")
        
        # 收集完成后的分析
        print("分析收集到的key（per-head）...")
        results = self._analyze_collected_keys()
        
        return results, current_ids[0]
    
    def _analyze_collected_keys(self):
        """分析收集到的所有key（per-head）"""
        all_results = {}
        
        for layer_idx in sorted(self.per_head_keys.keys()):
            layer_heads = self.per_head_keys[layer_idx]
            all_results[layer_idx] = {}
            
            for head_idx in sorted(layer_heads.keys()):
                head_keys = layer_heads[head_idx]  # list of [seq_len, head_dim]
                
                if len(head_keys) > 0:
                    # 取最后一个完整步骤的key（包含完整序列）
                    final_keys = head_keys[-1]  # [seq_len, head_dim]
                    
                    # 计算离散程度
                    stats = self.compute_dispersion(final_keys)
                    all_results[layer_idx][head_idx] = stats
            
            # 打印统计信息
            if layer_idx in all_results and all_results[layer_idx]:
                dispersions = [stats['dispersions'].mean().item() 
                              for stats in all_results[layer_idx].values()]
                print(f"Layer {layer_idx}: Head数={len(all_results[layer_idx])}, "
                      f"平均离散度={np.mean(dispersions):.4f} "
                      f"(min={np.min(dispersions):.4f}, max={np.max(dispersions):.4f})")
        
        return all_results


def plot_per_head_heatmap(results, output_dir):
    """
    绘制Layer-Head的离散程度Heatmap
    
    参数:
        results: {layer_idx: {head_idx: {'dispersions': tensor}}}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    num_layers = len(results)
    num_heads = len(next(iter(results.values())))  # 获取第一层的head数
    
    # 创建矩阵：[num_layers, num_heads]
    mean_dispersion_matrix = np.zeros((num_layers, num_heads))
    max_dispersion_matrix = np.zeros((num_layers, num_heads))
    
    for layer_idx, heads in results.items():
        for head_idx, stats in heads.items():
            dispersions = stats['dispersions'].numpy()
            mean_dispersion_matrix[layer_idx, head_idx] = dispersions.mean()
            max_dispersion_matrix[layer_idx, head_idx] = dispersions.max()
    
    # 创建图表 - 增加尺寸并调整布局
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 平均离散程度heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(mean_dispersion_matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_xlabel('Head Index', fontsize=12)
    ax1.set_ylabel('Layer Index', fontsize=12)
    ax1.set_title('Mean Dispersion per Layer-Head', fontsize=14, pad=10)
    ax1.set_xticks(range(num_heads))
    ax1.set_yticks(range(num_layers))
    
    # 只在格子足够大时添加数值标签
    min_font_size = min(6, 200 // max(num_layers, num_heads))
    if num_heads <= 12 and num_layers <= 20:
        for i in range(num_layers):
            for j in range(num_heads):
                text = ax1.text(j, i, f'{mean_dispersion_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", 
                              fontsize=min_font_size, weight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='Mean Dispersion', fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)
    
    # 最大离散程度heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(max_dispersion_matrix, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Head Index', fontsize=12)
    ax2.set_ylabel('Layer Index', fontsize=12)
    ax2.set_title('Max Dispersion per Layer-Head', fontsize=14, pad=10)
    ax2.set_xticks(range(num_heads))
    ax2.set_yticks(range(num_layers))
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='Max Dispersion', fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'per_head_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("已保存 Per-Head Heatmap")


def plot_head_comparison(results, layer_idx, output_dir):
    """
    绘制指定层的所有head的对比图
    
    参数:
        results: {layer_idx: {head_idx: {'dispersions': tensor}}}
        layer_idx: 要分析的层索引
        output_dir: 输出目录
    """
    if layer_idx not in results:
        return
    
    layer_data = results[layer_idx]
    num_heads = len(layer_data)
    
    # 准备数据
    head_dispersions = []
    head_labels = []
    
    for head_idx, stats in sorted(layer_data.items()):
        dispersions = stats['dispersions'].numpy()
        head_dispersions.append(dispersions)
        head_labels.append(f'Head {head_idx}')
    
    # 创建图表 - 增加尺寸并调整布局间距
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    # 1. 箱线图对比所有head
    ax1 = fig.add_subplot(gs[0, :])
    bp = ax1.boxplot(head_dispersions, tick_labels=head_labels, patch_artist=True)
    
    # 美化箱线图
    colors = plt.cm.viridis(np.linspace(0, 1, num_heads))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Head Index', fontsize=12)
    ax1.set_ylabel('Dispersion', fontsize=12)
    ax1.set_title(f'Layer {layer_idx}: Dispersion Distribution Across Heads', fontsize=14, pad=10)
    ax1.grid(True, alpha=0.3, axis='y')
    # 调整x轴标签，避免重叠
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    # 2. 平均离散程度条形图
    ax2 = fig.add_subplot(gs[1, 0])
    mean_dispersions = [d.mean() for d in head_dispersions]
    bars = ax2.bar(range(num_heads), mean_dispersions, color=colors, alpha=0.7)
    ax2.set_xlabel('Head Index', fontsize=11)
    ax2.set_ylabel('Mean Dispersion', fontsize=11)
    ax2.set_title(f'Layer {layer_idx}: Mean Dispersion per Head', fontsize=12, pad=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(num_heads))
    ax2.set_xticklabels(head_labels, rotation=30, ha='right', fontsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    
    # 智能添加数值标签（只在有足够空间时）
    y_range = max(mean_dispersions) - min(mean_dispersions)
    for i, (bar, val) in enumerate(zip(bars, mean_dispersions)):
        if y_range / num_heads > 2:  # 只有空间足够时才显示
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_range * 0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 最大离散程度条形图
    ax3 = fig.add_subplot(gs[1, 1])
    max_dispersions = [d.max() for d in head_dispersions]
    bars = ax3.bar(range(num_heads), max_dispersions, color=colors, alpha=0.7)
    ax3.set_xlabel('Head Index', fontsize=11)
    ax3.set_ylabel('Max Dispersion', fontsize=11)
    ax3.set_title(f'Layer {layer_idx}: Max Dispersion per Head', fontsize=12, pad=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(range(num_heads))
    ax3.set_xticklabels(head_labels, rotation=30, ha='right', fontsize=9)
    ax3.tick_params(axis='y', labelsize=9)
    
    # 智能添加数值标签
    y_range = max(max_dispersions) - min(max_dispersions)
    for i, (bar, val) in enumerate(zip(bars, max_dispersions)):
        if y_range / num_heads > 2:  # 只有空间足够时才显示
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_range * 0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 排序后的离散程度对比（选择几个代表性head）
    ax4 = fig.add_subplot(gs[2, 0])
    representative_heads = [0, num_heads//4, num_heads//2, 3*num_heads//4, num_heads-1]
    
    for head_idx in representative_heads:
        if head_idx < num_heads:
            sorted_disp = np.sort(head_dispersions[head_idx])[::-1]
            ax4.plot(sorted_disp, marker='o', markersize=3, alpha=0.7,
                    label=f'Head {head_idx}', linewidth=2)
    
    ax4.set_xlabel('Chunk Rank (sorted)', fontsize=11)
    ax4.set_ylabel('Dispersion', fontsize=11)
    ax4.set_title(f'Layer {layer_idx}: Sorted Dispersion Comparison', fontsize=12, pad=8)
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', labelsize=9)
    
    # 5. 统计汇总表
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('tight')
    ax5.axis('off')
    
    stats_data = []
    for head_idx in range(num_heads):
        disp = head_dispersions[head_idx]
        stats_data.append([
            f'H{head_idx}',
            f'{disp.mean():.2f}',
            f'{disp.std():.2f}',
            f'{disp.min():.2f}',
            f'{disp.max():.2f}',
            f'{np.median(disp):.2f}'
        ])
    
    table = ax5.table(cellText=stats_data,
                     colLabels=['Head', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.3, 1.8)
    
    # 设置表头样式（明确指定列数）
    num_cols = 6  # Head, Mean, Std, Min, Max, Median
    for i in range(num_cols):
        cell = table.get_celld()[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # 交替行颜色
    for i in range(1, num_heads + 1):
        for j in range(6):
            cell = table.get_celld()[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
    
    # 添加标题
    ax5.set_title(f'Layer {layer_idx}: Statistics Summary', fontsize=12, pad=15)
    
    plt.savefig(os.path.join(output_dir, f'layer_{layer_idx}_head_comparison.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"已保存 Layer {layer_idx} 的Head对比图")


def plot_individual_head_analysis(results, layer_idx, head_idx, output_dir):
    """
    绘制单个head的详细分析图（与主脚本类似）
    
    参数:
        results: {layer_idx: {head_idx: {'dispersions': tensor}}}
        layer_idx: 层索引
        head_idx: head索引
        output_dir: 输出目录
    """
    if layer_idx not in results or head_idx not in results[layer_idx]:
        return
    
    stats = results[layer_idx][head_idx]
    dispersions = stats['dispersions'].numpy()
    
    # 增加图表尺寸并调整布局
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. 离散程度 vs 块位置
    ax1 = axes[0]
    ax1.plot(range(len(dispersions)), dispersions, marker='o', markersize=4, alpha=0.7)
    ax1.set_xlabel('Chunk Index', fontsize=11)
    ax1.set_ylabel('Dispersion (Euclidean Distance)', fontsize=11)
    ax1.set_title(f'Layer {layer_idx} Head {head_idx}: Dispersion vs Position', fontsize=12, pad=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=10)
    
    # 2. 排序后的离散程度分布
    ax2 = axes[1]
    sorted_dispersions = np.sort(dispersions)[::-1]
    ax2.bar(range(len(sorted_dispersions)), sorted_dispersions, alpha=0.7)
    ax2.set_xlabel('Chunk Rank (sorted by dispersion)', fontsize=11)
    ax2.set_ylabel('Dispersion', fontsize=11)
    ax2.set_title(f'Layer {layer_idx} Head {head_idx}: Sorted Dispersion', fontsize=12, pad=8)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=10)
    
    # 优化统计信息文本框
    stats_text = (f'Max: {sorted_dispersions[0]:.3f}\n'
                 f'Min: {sorted_dispersions[-1]:.3f}\n'
                 f'Mean: {sorted_dispersions.mean():.3f}\n'
                 f'Median: {np.median(sorted_dispersions):.3f}')
    ax2.text(0.97, 0.96, stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7, edgecolor='gray'),
            fontsize=9,
            family='monospace')
    
    # 3. 离散程度的直方图
    ax3 = axes[2]
    ax3.hist(dispersions, bins=min(20, len(dispersions)//2), alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Dispersion', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Layer {layer_idx} Head {head_idx}: Dispersion Histogram', fontsize=12, pad=8)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.tick_params(axis='both', labelsize=10)
    
    # 添加统计线
    ax3.axvline(dispersions.mean(), color='red', linestyle='--',
               label=f'Mean: {dispersions.mean():.3f}', linewidth=2)
    ax3.axvline(np.median(dispersions), color='green', linestyle='--',
               label=f'Median: {np.median(dispersions):.3f}', linewidth=2)
    ax3.legend(fontsize=9, framealpha=0.9)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, f'layer_{layer_idx}_head_{head_idx}_analysis.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"已保存 Layer {layer_idx} Head {head_idx} 的详细分析图")


def main():
    parser = argparse.ArgumentParser(description="分析每个Attention Head的Key离散程度")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                       help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to process")
    parser.add_argument("--chunk_size", type=int, default=128,
                       help="Chunk size for key analysis")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="Maximum new tokens to generate")
    parser.add_argument("--output_dir", type=str, default="experiments/chunk_sparse/results_per_head",
                       help="Output directory for results")
    parser.add_argument("--analyze_layers", type=str, default="auto",
                       help="Which layers to analyze (comma-separated indices, 'all', or 'auto' for representative)")
    parser.add_argument("--analyze_heads", type=str, default="representative",
                       help="Which heads to analyze individually ('all', 'representative', or comma-separated indices)")
    parser.add_argument("--ruler", action="store_true",
                       help="Use RULER dataset instead of LongBench")
    parser.add_argument("--ruler_data_path", type=str,
                       default="data/ruler/data/llama-3/131072/qa_1/validation.jsonl",
                       help="Path to RULER dataset jsonl file")
    parser.add_argument("--max_input_length", type=int, default=131072,
                       help="Maximum input length for ruler dataset")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Per-Head Key离散程度分析实验")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"块大小: {args.chunk_size}")
    print(f"最大生成tokens: {args.max_new_tokens}")
    print(f"样本数量: {args.num_samples}")
    print("=" * 60)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    num_layers = len(model.model.layers)
    num_heads = model.model.layers[0].self_attn.num_heads
    print(f"模型已加载，层数: {num_layers}, Heads数: {num_heads}")
    
    # 加载数据
    print("\n加载数据...")
    
    # 检查是否使用ruler数据集
    use_ruler = '--ruler' in sys.argv or os.environ.get('USE_RULER', '').lower() == 'true'
    
    if use_ruler:
        # 加载ruler数据集
        ruler_data_path = os.environ.get('RULER_DATA_PATH', 
                                         'data/ruler/data/llama-3/131072/qa_1/validation.jsonl')
        
        try:
            import json
            dataset = []
            with open(ruler_data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= args.num_samples:
                        break
                    item = json.loads(line)
                    dataset.append({
                        'context': item.get('context', ''),
                        'input': item.get('input', ''),
                        'answers': item.get('outputs', []),
                        'all_classes': [],
                        'length': item.get('length', 0)
                    })
            
            print(f"Ruler数据集已加载，样本数量: {len(dataset)}")
            if len(dataset) > 0:
                print(f"第一个样本的input长度: {len(dataset[0]['input'])} 字符")
                print(f"第一个样本的length: {dataset[0]['length']}")
        except Exception as e:
            print(f"加载Ruler数据集失败: {e}")
            print("使用示例数据...")
            example_text = "Please analyze the following research paper and answer questions about it. " * 50
            dataset = [{"context": example_text, "input": "What is the main contribution?",
                       "answers": ["This is a sample answer."], "all_classes": [], "length": 0}]
    else:
        # 使用LongBench数据集
        try:
            from datasets import load_dataset
            raw_dataset = load_dataset("THUDM/LongBench", "2wikimqa", split="test", trust_remote_code=True)
            print(f"LongBench数据集已加载，样本数量: {len(raw_dataset)}")
            
            dataset = []
            for item in raw_dataset:
                dataset.append({
                    'context': item['context'],
                    'input': item.get('input', ''),
                    'answers': item['answers'],
                    'all_classes': item.get('all_classes', [])
                })
        except Exception as e:
            print(f"加载数据失败: {e}")
            example_text = "Please analyze the following research paper and answer questions about it. " * 50
            dataset = [{"context": example_text, "input": "What is the main contribution?",
                       "answers": ["This is a sample answer."], "all_classes": []}]
    
    # 创建分析器
    analyzer = PerHeadKeyDispersionAnalyzer(model, tokenizer,
                                            chunk_size=args.chunk_size,
                                            device=device)
    
    # 处理每个样本
    all_results = {}
    for sample_idx in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*60}")
        print(f"处理样本 {sample_idx + 1}/{min(args.num_samples, len(dataset))}")
        print(f"{'='*60}")
        
        sample = dataset[sample_idx]
        context = sample['context']
        input_question = sample.get('input', '')
        
        prompt = f"Context: {context}\n\nQuestion: {input_question}\n\nAnswer:" if input_question else context
        
        print(f"上下文长度: {len(context)} 字符")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs['input_ids'][0]
        
        print(f"输入tokens: {len(input_ids)}")
        
        # 分析生成过程
        results, generated_ids = analyzer.analyze_generation(
            input_ids,
            max_new_tokens=args.max_new_tokens
        )
        
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n生成的文本（前500字符）:\n{generated_text[:500]}...")
        
        all_results[sample_idx] = results
        
        # 绘制图表
        output_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制per-head heatmap
        print("\n生成Per-Head Heatmap...")
        plot_per_head_heatmap(results, output_dir)
        
        # 2. 确定要分析的层
        if args.analyze_layers == "all":
            layers_to_analyze = sorted(results.keys())
        elif args.analyze_layers == "auto":
            layers_to_analyze = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
        else:
            layers_to_analyze = [int(x.strip()) for x in args.analyze_layers.split(',')]
        
        # 3. 绘制层级的head对比
        print(f"\n生成 {len(layers_to_analyze)} 个层的Head对比图...")
        for layer_idx in layers_to_analyze:
            if layer_idx in results:
                plot_head_comparison(results, layer_idx, output_dir)
        
        # 4. 确定要分析的head
        if args.analyze_heads == "all":
            heads_to_analyze = range(num_heads)
        elif args.analyze_heads == "representative":
            heads_to_analyze = [0, num_heads//4, num_heads//2, 3*num_heads//4, num_heads-1]
        else:
            heads_to_analyze = [int(x.strip()) for x in args.analyze_heads.split(',')]
        
        # 5. 绘制单个head的详细分析
        print(f"\n生成 {len(layers_to_analyze * len(heads_to_analyze))} 个head的详细分析图...")
        for layer_idx in layers_to_analyze:
            for head_idx in heads_to_analyze:
                if layer_idx in results and head_idx in results[layer_idx]:
                    plot_individual_head_analysis(results, layer_idx, head_idx, output_dir)
    
    # 清理
    analyzer.remove_hooks()
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("=" * 60)
    print("\n生成的图表包括：")
    print("  1. per_head_heatmap.png - 所有层和head的离散程度热力图")
    print("  2. layer_*_head_comparison.png - 每层的head对比分析")
    print("  3. layer_*_head_*_analysis.png - 单个head的详细分析")


if __name__ == "__main__":
    main()
