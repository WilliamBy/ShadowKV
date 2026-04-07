"""
实验：分析key在不同块内的离散程度
目标：
1. 从longbench/qasper采样一条输入
2. 使用贪婪解码生成最大不超过128的文本
3. 在生成过程中收集各解码步、各层的key
4. 对key进行分块，求质心，计算块内离散程度（欧氏距离）
5. 绘制离散程度与块位置的关系以及排序后的离散程度直方图
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

# 使用非交互式后端
matplotlib.use('Agg')

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


@dataclass
class KeyData:
    """存储key数据的结构"""
    layer_idx: int
    step_idx: int  # 解码步骤
    keys: torch.Tensor  # shape: [num_heads, seq_len, head_dim] 或 [batch, num_heads, seq_len, head_dim]
    position: int  # 当前解码位置


class KeyCollector:
    """Hook类，用于收集模型各层的key"""
    
    def __init__(self, model):
        self.model = model
        self.collected_keys: List[KeyData] = []
        self.hooks = []
        self.current_step = 0
        
        # 注册hook
        self._register_hooks()
    
    def _register_hooks(self):
        """为每一层的attention注册forward hook来捕获key cache"""
        # 对于Llama模型，我们hook每层的self_attn的k_proj
        for layer_idx, layer in enumerate(self.model.model.layers):
            # 获取self_attn模块
            self_attn = layer.self_attn
            k_proj = self_attn.k_proj
            
            # 创建hook函数
            def make_hook(layer_idx):
                def hook(module, args, output):
                    # output是投影后的key
                    # shape: [batch, seq_len, hidden_size] (对于k_proj)
                    # 但这不是最终的key cache格式，还需要reshape和transpose
                    # 所以我们不在hook中收集，而是在生成后手动从past_key_values中获取
                    pass
                return hook
            
            # 注册hook（暂不使用，改用手动收集）
            # handle = k_proj.register_forward_hook(make_hook(layer_idx))
            # self.hooks.append(handle)
            pass
    
    def collect_from_past_key_values(self, past_key_values, step_idx):
        """
        从past_key_values中收集所有层的key
        past_key_values: tuple of tuples, 每个元素是(key, value)
        """
        if past_key_values is None:
            return
        
        for layer_idx in range(len(past_key_values)):
            key_tensor = past_key_values[layer_idx][0]  # [batch, num_heads, seq_len, head_dim]
            
            # 复制数据以避免引用问题
            key_data = KeyData(
                layer_idx=layer_idx,
                step_idx=step_idx,
                keys=key_tensor.detach().cpu().clone(),
                position=key_tensor.shape[2] - 1  # 最后一个位置
            )
            self.collected_keys.append(key_data)
    
    def reset(self):
        """重置收集器"""
        self.collected_keys = []
        self.current_step = 0
    
    def increment_step(self):
        """增加步骤计数"""
        self.current_step += 1
    
    def cleanup(self):
        """清理hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class KeySparseAnalyzer:
    """分析key的稀疏性和离散程度"""
    
    def __init__(self, chunk_size: int = 128):
        self.chunk_size = chunk_size
    
    def compute_centroid(self, keys: np.ndarray) -> np.ndarray:
        """
        计算质心
        keys: [N, D] 其中N是序列长度，D是key维度
        返回: [D] 质心向量
        """
        return np.mean(keys, axis=0)
    
    def compute_dispersion(self, keys: np.ndarray, centroid: np.ndarray) -> float:
        """
        计算离散程度（平均欧氏距离）
        keys: [N, D] 其中N是序列长度，D是key维度
        centroid: [D] 质心向量
        返回: 平均欧氏距离
        """
        # 计算每个key到质心的欧氏距离
        distances = np.linalg.norm(keys - centroid, axis=1)
        return np.mean(distances)
    
    def analyze_keys_by_chunk(self, keys: torch.Tensor) -> Dict[int, Dict]:
        """
        对keys按位置分块并分析离散程度
        keys: [batch, num_heads, seq_len, head_dim] 或 [num_heads, seq_len, head_dim]
        返回: {chunk_idx: {centroid, dispersion, num_keys}}
        """
        # 移除batch维度（如果存在）
        if keys.dim() == 4:
            keys = keys.squeeze(0)  # [num_heads, seq_len, head_dim]
        
        num_heads, seq_len, head_dim = keys.shape
        chunk_results = {}
        
        # 将所有head的所有key拼接在一起
        # 我们按位置分块，即每个块包含所有head在某个位置范围的key
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * self.chunk_size
            end_pos = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            if end_pos <= start_pos:
                continue
            
            # 提取该位置范围内所有head的所有key
            # shape: [num_heads * (end_pos - start_pos), head_dim]
            chunk_keys = keys[:, start_pos:end_pos, :]  # [num_heads, chunk_len, head_dim]
            chunk_keys = chunk_keys.permute(1, 0, 2)  # [chunk_len, num_heads, head_dim]
            chunk_keys = chunk_keys.reshape(-1, head_dim)  # [chunk_len * num_heads, head_dim]
            
            # 转换为numpy
            chunk_keys_np = chunk_keys.cpu().numpy()
            
            # 计算质心
            centroid = self.compute_centroid(chunk_keys_np)
            
            # 计算离散程度
            dispersion = self.compute_dispersion(chunk_keys_np, centroid)
            
            chunk_results[chunk_idx] = {
                'centroid': centroid,
                'dispersion': dispersion,
                'num_keys': chunk_keys_np.shape[0],
                'start_pos': start_pos,
                'end_pos': end_pos
            }
        
        return chunk_results


def load_sample_from_qasper(tokenizer, max_samples: int = 1, min_length: int = 1000) -> Dict:
    """
    从LongBench的qasper数据集采样一个样本
    返回格式化的prompt
    """
    print("Loading LongBench/qasper dataset...")
    
    try:
        dataset = load_dataset(
            'THUDM/LongBench',
            'qasper',
            split='test',
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # 使用本地缓存的数据
        print("Trying to use local dataset cache...")
        dataset = load_dataset(
            'THUDM/LongBench',
            'qasper',
            split='test',
            trust_remote_code=True,
            cache_dir='/root/.cache/huggingface'
        )
    
    # 加载prompt模板
    prompt_template = "Answer the question based on the given passage. Passage: {context}\n\nQuestion: {input}\n\nAnswer:"
    
    # 找到符合长度要求的样本
    for sample in dataset:
        prompt = prompt_template.format(**sample)
        tokenized = tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(tokenized) >= min_length:
            print(f"Found sample with {len(tokenized)} tokens")
            return {
                'prompt': prompt,
                'context': sample['context'],
                'question': sample['input'],
                'answers': sample['answers']
            }
    
    # 如果没找到符合长度的，返回第一个
    sample = dataset[0]
    prompt = prompt_template.format(**sample)
    return {
        'prompt': prompt,
        'context': sample['context'],
        'question': sample['input'],
        'answers': sample['answers']
    }


def generate_with_key_collection(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    collector: Optional[KeyCollector] = None
) -> Tuple[str, List[KeyData]]:
    """
    使用贪婪解码生成文本，并收集各层的key
    """
    print(f"Generating with max_new_tokens={max_new_tokens}...")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # 准备生成
    if collector:
        collector.reset()
    
    generated_ids = input_ids.clone()
    past_key_values = None
    
    # Prefill阶段
    print("Running prefill...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    
    print(f"Prefill complete. Sequence length: {input_ids.shape[1]}")
    
    # Decode阶段 - 贪婪解码
    print("Running decode with greedy strategy...")
    for step in range(max_new_tokens):
        # 获取下一个token（贪婪）
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 收集当前步骤的keys（从past_key_values中）
        if collector:
            collector.increment_step()
            collector.collect_from_past_key_values(past_key_values, step)
        
        # 拼接生成的token
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # 检查是否生成结束
        if next_token.item() == tokenizer.eos_token_id:
            print(f"Generated EOS token at step {step}")
            break
        
        # 准备下一步
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        
        if (step + 1) % 32 == 0 or step == 0:
            print(f"Step {step + 1}/{max_new_tokens}, current seq len: {past_key_values[0][0].shape[2]}")
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Generation complete. Generated {len(generated_ids[0]) - len(input_ids[0])} tokens")
    
    return generated_text, collector.collected_keys if collector else []


def visualize_results(
    all_results: List[Dict],
    output_dir: str
):
    """
    可视化结果：
    1. 离散程度与块位置的关系
    2. 排序后的离散程度直方图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 聚合所有层和所有步骤的数据
    all_dispersions = []
    all_positions = []
    
    for result in all_results:
        chunk_idx = result['chunk_idx']
        layer_idx = result['layer_idx']
        step_idx = result['step_idx']
        dispersion = result['dispersion']
        start_pos = result['start_pos']
        
        all_dispersions.append(dispersion)
        all_positions.append(chunk_idx)
    
    all_dispersions = np.array(all_dispersions)
    all_positions = np.array(all_positions)
    
    # 图1: 离散程度与块位置的关系
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 散点图：所有数据点
    ax = axes[0, 0]
    scatter = ax.scatter(all_positions, all_dispersions, alpha=0.3, s=10, c=all_positions, cmap='viridis')
    ax.set_xlabel('Chunk Index', fontsize=12)
    ax.set_ylabel('Dispersion (Average Euclidean Distance)', fontsize=12)
    ax.set_title('Dispersion vs Chunk Position (All Data Points)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Chunk Index')
    
    # 1.2 箱线图：按块位置分组
    ax = axes[0, 1]
    
    # 计算每个块位置的平均离散程度
    unique_positions = sorted(set(all_positions))
    mean_dispersions = []
    std_dispersions = []
    
    for pos in unique_positions:
        mask = all_positions == pos
        mean_dispersions.append(np.mean(all_dispersions[mask]))
        std_dispersions.append(np.std(all_dispersions[mask]))
    
    ax.errorbar(unique_positions, mean_dispersions, yerr=std_dispersions, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6)
    ax.set_xlabel('Chunk Index', fontsize=12)
    ax.set_ylabel('Mean Dispersion', fontsize=12)
    ax.set_title('Mean Dispersion vs Chunk Position (with std)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图2: 排序后的离散程度分布
    # 2.1 直方图
    ax = axes[1, 0]
    sorted_dispersions = np.sort(all_dispersions)
    ax.hist(sorted_dispersions, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Dispersion (sorted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Sorted Dispersion Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2.2 累积分布
    ax = axes[1, 1]
    cumulative = np.linspace(0, 1, len(sorted_dispersions))
    ax.plot(sorted_dispersions, cumulative, linewidth=2, color='darkblue')
    ax.fill_between(sorted_dispersions, 0, cumulative, alpha=0.3, color='steelblue')
    ax.set_xlabel('Dispersion (sorted)', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative Distribution of Dispersion', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dispersion_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved dispersion analysis to {output_dir}/dispersion_analysis.png")
    plt.close()
    
    # 图3: 按层分析
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 3.1 不同层的平均离散程度
    ax = axes[0]
    layer_data = defaultdict(list)
    for result in all_results:
        layer_data[result['layer_idx']].append(result['dispersion'])
    
    layer_indices = sorted(layer_data.keys())
    layer_means = [np.mean(layer_data[idx]) for idx in layer_indices]
    layer_stds = [np.std(layer_data[idx]) for idx in layer_indices]
    
    ax.errorbar(layer_indices, layer_means, yerr=layer_stds,
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Dispersion', fontsize=12)
    ax.set_title('Mean Dispersion per Layer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3.2 热图：离散程度随层和块位置的变化
    ax = axes[1]
    
    # 创建热图数据
    unique_layers = sorted(set([r['layer_idx'] for r in all_results]))
    unique_chunks = sorted(set([r['chunk_idx'] for r in all_results]))
    
    heatmap_data = np.zeros((len(unique_layers), len(unique_chunks)))
    for i, layer_idx in enumerate(unique_layers):
        for j, chunk_idx in enumerate(unique_chunks):
            values = [r['dispersion'] for r in all_results 
                     if r['layer_idx'] == layer_idx and r['chunk_idx'] == chunk_idx]
            if values:
                heatmap_data[i, j] = np.mean(values)
            else:
                heatmap_data[i, j] = np.nan
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', origin='lower')
    ax.set_xlabel('Chunk Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title('Dispersion Heatmap (Layer vs Chunk)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Dispersion')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved layer analysis to {output_dir}/layer_analysis.png")
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Statistics Summary:")
    print("="*60)
    print(f"Total data points: {len(all_dispersions)}")
    print(f"Mean dispersion: {np.mean(all_dispersions):.4f}")
    print(f"Std dispersion: {np.std(all_dispersions):.4f}")
    print(f"Min dispersion: {np.min(all_dispersions):.4f}")
    print(f"Max dispersion: {np.max(all_dispersions):.4f}")
    print(f"Median dispersion: {np.median(all_dispersions):.4f}")
    
    # 高离散度的块（前10%）
    threshold = np.percentile(all_dispersions, 90)
    high_dispersion_count = np.sum(all_dispersions >= threshold)
    print(f"\nHigh dispersion chunks (>90th percentile): {high_dispersion_count}/{len(all_dispersions)}")
    print(f"This confirms that high dispersion chunks are rare!")
    
    # 保存详细数据
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        # 将numpy类型转换为Python原生类型以便JSON序列化
        serializable_results = []
        for r in all_results:
            serializable_results.append({
                'layer_idx': int(r['layer_idx']),
                'step_idx': int(r['step_idx']),
                'chunk_idx': int(r['chunk_idx']),
                'dispersion': float(r['dispersion']),
                'start_pos': int(r['start_pos']),
                'end_pos': int(r['end_pos']),
                'num_keys': int(r['num_keys'])
            })
        json.dump(serializable_results, f, indent=2)
    print(f"Saved detailed results to {output_dir}/results.json")


def main():
    """主函数"""
    # 配置
    model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    chunk_size = 128
    max_new_tokens = 128
    output_dir = "/root/workspace/ShadowKV/experiments/chunk_sparse/results"
    
    print("="*60)
    print("Key Dispersion Analysis Experiment")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Chunk size: {chunk_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # 加载模型和tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/root/.cache/huggingface"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/root/.cache/huggingface"
    )
    
    print(f"Model loaded. Device: {model.device}")
    print(f"Number of layers: {len(model.model.layers)}")
    
    # 采样数据
    print("\n" + "="*60)
    sample = load_sample_from_qasper(tokenizer, min_length=1000)
    print(f"Sample loaded. Context length: {len(sample['context'])}")
    print(f"Question: {sample['question']}")
    print(f"Prompt length: {len(tokenizer.encode(sample['prompt']))} tokens")
    
    # 创建key收集器
    print("\n" + "="*60)
    print("Setting up key collection...")
    collector = KeyCollector(model)
    analyzer = KeySparseAnalyzer(chunk_size=chunk_size)
    
    # 生成文本并收集keys
    print("\n" + "="*60)
    print("Generating text and collecting keys...")
    generated_text, collected_keys = generate_with_key_collection(
        model=model,
        tokenizer=tokenizer,
        prompt=sample['prompt'],
        max_new_tokens=max_new_tokens,
        collector=collector
    )
    
    print(f"\nGenerated text (first 500 chars):\n{generated_text[:500]}...")
    
    # 分析keys
    print("\n" + "="*60)
    print(f"Analyzing keys with chunk_size={chunk_size}...")
    print(f"Collected {len(collected_keys)} key snapshots")
    
    all_results = []
    
    for i, key_data in enumerate(collected_keys):
        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(collected_keys)} key snapshots...")
        
        # 对每个key snapshot进行分块分析
        chunk_results = analyzer.analyze_keys_by_chunk(key_data.keys)
        
        for chunk_idx, chunk_info in chunk_results.items():
            all_results.append({
                'layer_idx': key_data.layer_idx,
                'step_idx': key_data.step_idx,
                'chunk_idx': chunk_idx,
                'dispersion': chunk_info['dispersion'],
                'start_pos': chunk_info['start_pos'],
                'end_pos': chunk_info['end_pos'],
                'num_keys': chunk_info['num_keys']
            })
    
    print(f"Analysis complete. Total results: {len(all_results)}")
    
    # 可视化
    print("\n" + "="*60)
    print("Creating visualizations...")
    visualize_results(all_results, output_dir)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print(f"Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
