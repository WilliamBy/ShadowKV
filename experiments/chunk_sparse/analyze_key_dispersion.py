#!/usr/bin/env python3
"""
分析Key在不同块内的离散程度

目标：观察按位置分块的key在不同块内的离散程度（通过各key到块内质心的欧氏距离度量）
预期：高离散的块数量较少
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import argparse
import sys
import os
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

class KeyDispersionAnalyzer:
    """收集并分析key的离散程度"""
    
    def __init__(self, model, tokenizer, chunk_size=128, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.device = device
        
        # 存储数据
        self.all_keys = defaultdict(list)  # {layer_idx: [keys]}
        self.all_positions = []
        
        # 注册hook
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册forward hook来收集key"""
        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                # Llama-3的attention输出格式: (hidden_states, attention_weights, cache)
                # cache是DynamicCache对象，包含key_cache和value_cache
                if isinstance(output, tuple) and len(output) >= 3:
                    cache = output[2]
                    if hasattr(cache, 'key_cache') and len(cache.key_cache) > layer_idx:
                        key = cache.key_cache[layer_idx]  # [batch_size, num_heads, seq_len, head_dim]
                        
                        # 只收集第一个样本
                        key = key[0].detach().cpu()  # [num_heads, seq_len, head_dim]
                        
                        # 合并所有head: [seq_len, num_heads * head_dim]
                        num_heads, seq_len, head_dim = key.shape
                        key = key.permute(1, 0, 2).reshape(seq_len, -1)
                        
                        self.all_keys[layer_idx].append(key)
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
            keys: [seq_len, hidden_dim]
        
        返回:
            chunk_stats: 每个块的统计信息
        """
        seq_len, hidden_dim = keys.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        chunk_centroids = []
        chunk_dispersions = []
        chunk_sizes = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            chunk_keys = keys[start_idx:end_idx]  # [chunk_size, hidden_dim]
            chunk_sizes.append(chunk_keys.shape[0])
            
            # 计算质心
            centroid = chunk_keys.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            chunk_centroids.append(centroid)
            
            # 计算离散程度（平均欧氏距离）
            distances = torch.norm(chunk_keys - centroid, dim=1)  # [chunk_size]
            dispersion = distances.mean().item()
            chunk_dispersions.append(dispersion)
        
        return {
            'centroids': torch.cat(chunk_centroids, dim=0),  # [num_chunks, hidden_dim]
            'dispersions': torch.tensor(chunk_dispersions),  # [num_chunks]
            'sizes': torch.tensor(chunk_sizes),  # [num_chunks]
        }
    
    @torch.no_grad()
    def analyze_generation(self, input_ids, max_new_tokens=64):
        """
        分析生成过程中的key离散程度
        
        参数:
            input_ids: [seq_len] 或 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
        """
        # 确保有batch维度
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        
        input_ids = input_ids.to(self.device)
        
        # 清空之前的收集
        self.all_keys.clear()
        self.all_positions = []
        
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
        print("分析收集到的key...")
        results = self._analyze_collected_keys()
        
        return results, current_ids[0]
    
    def _analyze_collected_keys(self):
        """分析收集到的所有key"""
        all_results = {}
        
        for layer_idx in sorted(self.all_keys.keys()):
            layer_keys = self.all_keys[layer_idx]  # list of [seq_len, hidden_dim]
            
            # 合并所有步骤的key
            # 由于每一步的key只包含该步骤的新增key，我们需要按位置拼接
            if len(layer_keys) > 0:
                # 取最后一个完整步骤的key（包含完整序列）
                final_keys = layer_keys[-1]  # [total_seq_len, hidden_dim]
                
                # 计算离散程度
                stats = self.compute_dispersion(final_keys)
                all_results[layer_idx] = stats
                
                print(f"Layer {layer_idx}: 序列长度={final_keys.shape[0]}, "
                      f"块数={len(stats['dispersions'])}, "
                      f"平均离散度={stats['dispersions'].mean():.4f}")
        
        return all_results


def plot_dispersion_analysis(results, output_dir):
    """
    绘制离散程度分析图表
    
    参数:
        results: {layer_idx: {'dispersions': tensor, 'sizes': tensor}}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择几个代表性的层进行可视化
    representative_layers = [0, len(results)//4, len(results)//2, 3*len(results)//4, len(results)-1]
    
    for layer_idx in representative_layers:
        if layer_idx not in results:
            continue
            
        stats = results[layer_idx]
        dispersions = stats['dispersions'].numpy()
        sizes = stats['sizes'].numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 离散程度 vs 块位置
        ax1 = axes[0]
        ax1.plot(range(len(dispersions)), dispersions, marker='o', markersize=4, alpha=0.7)
        ax1.set_xlabel('Chunk Index', fontsize=12)
        ax1.set_ylabel('Dispersion (Euclidean Distance)', fontsize=12)
        ax1.set_title(f'Layer {layer_idx}: Dispersion vs Chunk Position', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. 排序后的离散程度分布
        ax2 = axes[1]
        sorted_dispersions = np.sort(dispersions)[::-1]  # 降序
        ax2.bar(range(len(sorted_dispersions)), sorted_dispersions, alpha=0.7)
        ax2.set_xlabel('Chunk Rank (sorted by dispersion)', fontsize=12)
        ax2.set_ylabel('Dispersion', fontsize=12)
        ax2.set_title(f'Layer {layer_idx}: Sorted Dispersion Distribution', fontsize=14)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 添加统计信息
        ax2.text(0.98, 0.97, 
                f'Max: {sorted_dispersions[0]:.3f}\n'
                f'Min: {sorted_dispersions[-1]:.3f}\n'
                f'Mean: {sorted_dispersions.mean():.3f}\n'
                f'Median: {np.median(sorted_dispersions):.3f}',
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        # 3. 离散程度的直方图分布
        ax3 = axes[2]
        ax3.hist(dispersions, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Dispersion', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title(f'Layer {layer_idx}: Dispersion Histogram', fontsize=14)
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 添加统计线
        ax3.axvline(dispersions.mean(), color='red', linestyle='--', 
                   label=f'Mean: {dispersions.mean():.3f}', linewidth=2)
        ax3.axvline(np.median(dispersions), color='green', linestyle='--', 
                   label=f'Median: {np.median(dispersions):.3f}', linewidth=2)
        ax3.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layer_{layer_idx}_dispersion_analysis.png'), dpi=300)
        plt.close()
        
        print(f"已保存 Layer {layer_idx} 的图表")
    
    # 创建汇总图表：所有层的平均离散程度
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    layer_means = []
    layer_maxs = []
    layer_mins = []
    layer_nums = []
    
    for layer_idx in sorted(results.keys()):
        stats = results[layer_idx]
        dispersions = stats['dispersions'].numpy()
        layer_nums.append(layer_idx)
        layer_means.append(dispersions.mean())
        layer_maxs.append(dispersions.max())
        layer_mins.append(dispersions.min())
    
    ax.plot(layer_nums, layer_means, 'o-', label='Mean', linewidth=2, markersize=6)
    ax.fill_between(layer_nums, layer_mins, layer_maxs, alpha=0.3, label='Min-Max Range')
    
    ax.set_xlabel('Layer Index', fontsize=14)
    ax.set_ylabel('Dispersion', fontsize=14)
    ax.set_title('Dispersion Statistics Across All Layers', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_layers_summary.png'), dpi=300)
    plt.close()
    
    print(f"已保存所有层的汇总图表")


def main():
    parser = argparse.ArgumentParser(description="分析Key的块内离散程度")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                       help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to process")
    parser.add_argument("--chunk_size", type=int, default=128,
                       help="Chunk size for key analysis")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="Maximum new tokens to generate")
    parser.add_argument("--output_dir", type=str, default="experiments/chunk_sparse/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Key离散程度分析实验")
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
    
    print(f"模型已加载，层数: {len(model.model.layers)}")
    
    # 加载数据
    print("\n加载数据...")
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("THUDM/LongBench", "qasper", split="test", trust_remote_code=True)
        print(f"数据集已加载，样本数量: {len(raw_dataset)}")
        
        # 将数据转换为需要的格式
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
        print("使用示例数据...")
        # 使用示例数据
        example_text = "Please analyze the following research paper and answer questions about it. " * 50
        dataset = [{"context": example_text, "input": "What is the main contribution?", "answers": ["This is a sample answer."], "all_classes": []}]
    
    # 创建分析器
    analyzer = KeyDispersionAnalyzer(model, tokenizer, 
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
        
        # 构建完整的输入
        if input_question:
            prompt = f"Context: {context}\n\nQuestion: {input_question}\n\nAnswer:"
        else:
            prompt = context
        
        print(f"上下文长度: {len(context)} 字符")
        print(f"总输入长度: {len(prompt)} 字符")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs['input_ids'][0]
        
        print(f"输入tokens: {len(input_ids)}")
        
        # 分析生成过程
        results, generated_ids = analyzer.analyze_generation(
            input_ids, 
            max_new_tokens=args.max_new_tokens
        )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n生成的文本:\n{generated_text}")
        
        all_results[sample_idx] = results
        
        # 绘制图表
        output_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        plot_dispersion_analysis(results, output_dir)
    
    # 清理
    analyzer.remove_hooks()
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
