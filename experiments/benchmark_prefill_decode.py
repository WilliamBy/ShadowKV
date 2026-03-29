"""
Benchmark script to measure prefill and decode phase latencies.
This script demonstrates that decoding dominates the total generation time.
"""

import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments


def seed_everything(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0, 0


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with Flash Attention backend."""
    print(f"Loading model: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Load model with Flash Attention
    # Flash Attention 2 is automatically used when available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            # Enable Flash Attention 2 (will fall back to SDPA if not available)
            attn_implementation="flash_attention_2",
            use_cache=True,  # Important for decode speed
        )
    except Exception as e:
        print(f"Warning: Could not load with flash_attention_2: {e}")
        print("Falling back to default attention implementation")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,
        )
    
    # Verify Flash Attention is being used
    config = model.config
    attn_impl = getattr(config, '_attn_implementation', 'default')
    print(f"Attention implementation: {attn_impl}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def load_longbench_samples(
    dataset_name: str, 
    num_samples: int = 5,
    tokenizer=None,
    prompt_format: str = None,
    maxlen: int = None
) -> List[Dict]:
    """
    Load samples from LongBench dataset.
    
    Args:
        dataset_name: Name of the LongBench dataset
        num_samples: Number of samples to load
        tokenizer: Tokenizer to calculate input lengths (required if maxlen is set)
        prompt_format: Prompt template format (required if maxlen is set)
        maxlen: Maximum input length in tokens (samples exceeding this will be skipped)
    
    Returns:
        List of sample dictionaries
    """
    print(f"Loading {num_samples} samples from {dataset_name}")
    if maxlen is not None:
        print(f"Filtering samples with input length <= {maxlen} tokens")
    
    try:
        data = load_dataset("THUDM/LongBench", dataset_name, split="test", trust_remote_code=True)
        
        # If maxlen is specified, we need to filter samples by input length
        if maxlen is not None:
            if tokenizer is None or prompt_format is None:
                raise ValueError("tokenizer and prompt_format are required when maxlen is set")
            
            # First pass: calculate token lengths for all samples
            valid_samples = []
            total_samples = len(data)
            print(f"Scanning {total_samples} samples to find ones within length limit...")
            
            for idx, sample in enumerate(data):
                # Apply prompt format
                prompt = apply_prompt_format(sample, prompt_format)
                
                # Calculate token length
                inputs = tokenizer(prompt, truncation=False, max_length=None)
                input_length = len(inputs["input_ids"])
                
                # Only keep samples within the length limit
                if input_length <= maxlen:
                    valid_samples.append((idx, sample, input_length))
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{total_samples} samples, found {len(valid_samples)} valid")
            
            print(f"Found {len(valid_samples)} samples with input length <= {maxlen} tokens")
            
            if len(valid_samples) == 0:
                print("WARNING: No samples found within the length limit!")
                return []
            
            # Sample evenly from valid samples
            num_to_sample = min(num_samples, len(valid_samples))
            sample_indices = np.linspace(0, len(valid_samples) - 1, num_to_sample, dtype=int)
            samples = [valid_samples[i][1] for i in sample_indices]
            
            # Print length distribution of selected samples
            lengths = [valid_samples[i][2] for i in sample_indices]
            print(f"Selected samples input length range: {min(lengths)} - {max(lengths)} tokens (avg: {np.mean(lengths):.0f})")
        else:
            # No length limit, sample evenly across the dataset
            total_samples = len(data)
            indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
            samples = [data[int(i)] for i in indices]
        
        print(f"Loaded {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def apply_prompt_format(sample: Dict, prompt_format: str) -> str:
    """Apply prompt format to the sample."""
    context = sample.get("context", "")
    question = sample.get("input", sample.get("question", ""))
    
    try:
        prompt = prompt_format.format(context=context, question=question)
    except:
        # Fallback for different prompt formats
        if "{context}" in prompt_format and "{input}" in prompt_format:
            prompt = prompt_format.format(context=context, input=question)
        elif "{context}" in prompt_format:
            prompt = prompt_format.format(context=context)
        else:
            prompt = prompt_format.format(question=question)
    
    return prompt


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
    max_input_length: int = None,
) -> Tuple[float, float, int, int]:
    """
    Benchmark a single generation, measuring prefill and decode phases.
    
    Returns:
        prefill_time: Time to process input prompt (prefill phase)
        decode_time: Time to generate output tokens (decode phase)
        input_length: Number of input tokens
        output_length: Number of output tokens generated
    """
    # Tokenize input with strict length limit
    if max_input_length is not None:
        print(f"  Truncating input to max {max_input_length} tokens...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
    
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]
    
    print(f"  Actual input length: {input_length} tokens")
    
    # Show memory before processing
    if device == "cuda":
        alloc_mem, reserved_mem = get_memory_info()
        print(f"  GPU memory before prefill: {alloc_mem:.2f} GB allocated, {reserved_mem:.2f} GB reserved")
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids[:, :min(100, input_length)])
    
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Clear cache before measurement
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Measure prefill phase (process input prompt)
    start_prefill = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_prefill = time.perf_counter()
    prefill_time = end_prefill - start_prefill
    
    # Show memory after prefill
    if device == "cuda":
        alloc_mem, reserved_mem = get_memory_info()
        print(f"  GPU memory after prefill:  {alloc_mem:.2f} GB allocated, {reserved_mem:.2f} GB reserved")
    
    # Clear cache before decode phase
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Measure decode phase (generate tokens)
    # We'll use generate() but measure time per token
    start_decode = time.perf_counter()
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_decode = time.perf_counter()
    decode_time = end_decode - start_decode
    
    # Show memory after decode
    if device == "cuda":
        alloc_mem, reserved_mem = get_memory_info()
        print(f"  GPU memory after decode:  {alloc_mem:.2f} GB allocated, {reserved_mem:.2f} GB reserved")
        torch.cuda.empty_cache()
    
    # Calculate output length (excluding input)
    output_length = output_ids.shape[1] - input_length
    
    return prefill_time, decode_time, input_length, output_length


def run_benchmark(
    model_name: str,
    datasets: List[str],
    num_samples: int = 5,
    max_new_tokens: int = 512,
    warmup_runs: int = 3,
    maxlen: int = None,
):
    """Run comprehensive benchmark."""
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    # Load dataset configurations
    dataset2prompt = json.load(
        open("data/long_bench/config/dataset2prompt.json", "r")
    )
    dataset2maxlen = json.load(
        open("data/long_bench/config/dataset2maxlen.json", "r")
    )
    
    results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Get prompt format for this dataset
        prompt_format = dataset2prompt.get(dataset_name, "{context}\n\nQuestion: {question}\nAnswer:")
        
        # Load samples (with length filtering if maxlen is specified)
        samples = load_longbench_samples(
            dataset_name, 
            num_samples,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            maxlen=maxlen
        )
        
        if not samples:
            print(f"Skipping {dataset_name} - no samples loaded")
            continue
        
        max_gen = min(max_new_tokens, dataset2maxlen.get(dataset_name, 512))
        
        dataset_results = []
        
        for idx, sample in enumerate(samples):
            print(f"\n--- Sample {idx + 1}/{len(samples)} ---")
            
            # Apply prompt format
            prompt = apply_prompt_format(sample, prompt_format)
            
            # Check initial prompt length before tokenization
            prompt_chars = len(prompt)
            print(f"  Prompt size: {prompt_chars} characters")
            
            # Warmup runs
            print(f"Running {warmup_runs} warmup runs...")
            try:
                for i in range(warmup_runs):
                    benchmark_generation(model, tokenizer, prompt, max_gen, device, max_input_length=maxlen)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                
                # Actual benchmark
                print("Running benchmark...")
                prefill_time, decode_time, input_length, output_length = benchmark_generation(
                    model, tokenizer, prompt, max_gen, device, max_input_length=maxlen
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ERROR: Out of memory! Try reducing --maxlen or --max_new_tokens")
                    print(f"  Current maxlen: {maxlen}, max_new_tokens: {max_gen}")
                    continue
                else:
                    raise e
            
            # Calculate metrics
            total_time = prefill_time + decode_time
            prefill_pct = (prefill_time / total_time * 100) if total_time > 0 else 0
            decode_pct = (decode_time / total_time * 100) if total_time > 0 else 0
            
            # Tokens per second
            prefill_tps = input_length / prefill_time if prefill_time > 0 else 0
            decode_tps = output_length / decode_time if decode_time > 0 else 0
            
            print(f"\nResults:")
            print(f"  Input length:  {input_length} tokens")
            print(f"  Output length: {output_length} tokens")
            print(f"  Prefill time:  {prefill_time:.4f} s ({prefill_pct:.1f}%)")
            print(f"  Decode time:   {decode_time:.4f} s ({decode_pct:.1f}%)")
            print(f"  Total time:    {total_time:.4f} s")
            print(f"  Prefill speed: {prefill_tps:.1f} tokens/s")
            print(f"  Decode speed:  {decode_tps:.1f} tokens/s")
            
            dataset_results.append({
                "dataset": dataset_name,
                "sample_idx": idx,
                "input_length": input_length,
                "output_length": output_length,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time,
                "prefill_pct": prefill_pct,
                "decode_pct": decode_pct,
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
            })
        
        results.extend(dataset_results)
    
    return results


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    # Group by dataset
    datasets = {}
    for r in results:
        dataset = r["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(r)
    
    # Per-dataset statistics
    for dataset, dataset_results in datasets.items():
        print(f"\n{dataset}:")
        print("-" * 60)
        
        prefill_times = [r["prefill_time"] for r in dataset_results]
        decode_times = [r["decode_time"] for r in dataset_results]
        total_times = [r["total_time"] for r in dataset_results]
        input_lengths = [r["input_length"] for r in dataset_results]
        output_lengths = [r["output_length"] for r in dataset_results]
        
        avg_prefill = np.mean(prefill_times)
        avg_decode = np.mean(decode_times)
        avg_total = np.mean(total_times)
        avg_input = np.mean(input_lengths)
        avg_output = np.mean(output_lengths)
        
        avg_prefill_pct = np.mean([r["prefill_pct"] for r in dataset_results])
        avg_decode_pct = np.mean([r["decode_pct"] for r in dataset_results])
        
        print(f"  Samples:              {len(dataset_results)}")
        print(f"  Avg input length:     {avg_input:.0f} tokens")
        print(f"  Avg output length:    {avg_output:.0f} tokens")
        print(f"  Avg prefill time:     {avg_prefill:.4f} s ({avg_prefill_pct:.1f}%)")
        print(f"  Avg decode time:      {avg_decode:.4f} s ({avg_decode_pct:.1f}%)")
        print(f"  Avg total time:       {avg_total:.4f} s")
        print(f"  Decode/Prefill ratio: {avg_decode/avg_prefill:.2f}x")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS:")
    print(f"{'='*80}\n")
    
    all_prefill = [r["prefill_time"] for r in results]
    all_decode = [r["decode_time"] for r in results]
    all_total = [r["total_time"] for r in results]
    all_input = [r["input_length"] for r in results]
    all_output = [r["output_length"] for r in results]
    
    total_prefill = sum(all_prefill)
    total_decode = sum(all_decode)
    total_time = sum(all_total)
    
    print(f"  Total samples:         {len(results)}")
    print(f"  Total prefill time:    {total_prefill:.4f} s ({total_prefill/total_time*100:.1f}%)")
    print(f"  Total decode time:     {total_decode:.4f} s ({total_decode/total_time*100:.1f}%)")
    print(f"  Total time:            {total_time:.4f} s")
    print(f"\n  Avg input length:      {np.mean(all_input):.0f} tokens")
    print(f"  Avg output length:     {np.mean(all_output):.0f} tokens")
    print(f"  Avg prefill time:      {np.mean(all_prefill):.4f} s")
    print(f"  Avg decode time:       {np.mean(all_decode):.4f} s")
    print(f"  Avg total time:        {np.mean(all_total):.4f} s")
    print(f"\n  Decode/Prefill ratio:  {total_decode/total_prefill:.2f}x")
    print(f"\n  {'*' * 60}")
    print(f"  CONCLUSION: Decode phase takes {total_decode/total_time*100:.1f}% of total time")
    print(f"  {'*' * 60}")


def save_results(results: List[Dict], filepath: str):
    """Save benchmark results to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    serializable_results = []
    for r in results:
        serializable_result = {}
        for key, value in r.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_result[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    # Add metadata
    output_data = {
        "metadata": {
            "total_samples": len(results),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": serializable_results
    }
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {filepath}")
    print(f"{'='*80}")


def load_results(filepath: str) -> List[Dict]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Loaded results from: {filepath}")
    print(f"Total samples: {len(data['results'])}")
    print(f"{'='*80}\n")
    
    return data['results']


def plot_results(results: List[Dict], output_path: str = None):
    """
    Plot comprehensive analysis of prefill vs decode performance.
    
    Args:
        results: List of benchmark result dictionaries
        output_path: Path to save the figure (if None, display interactively)
    """
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*80}\n")
    
    # Extract data
    datasets = list(set([r['dataset'] for r in results]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    dataset_colors = dict(zip(datasets, colors))
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)
    
    # 1. Prefill vs Decode time by input length (scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        input_lengths = [r['input_length'] for r in dataset_results]
        prefill_times = [r['prefill_time'] for r in dataset_results]
        decode_times = [r['decode_time'] for r in dataset_results]
        
        ax1.scatter(input_lengths, prefill_times, label=f'{dataset} (prefill)', 
                   alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color=dataset_colors[dataset])
        ax1.scatter(input_lengths, decode_times, label=f'{dataset} (decode)', 
                   alpha=0.6, s=80, marker='^', edgecolors='black', linewidth=0.5, color=dataset_colors[dataset])
    
    ax1.set_xlabel('Input Length (tokens)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Prefill vs Decode Time by Input Length', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time breakdown by input length (stacked bar)
    ax2 = fig.add_subplot(gs[0, 1])
    # Group by input length ranges
    input_bins = [0, 2000, 4000, 6000, 8000, 10000, 15000, 20000]
    bin_labels = ['0-2k', '2-4k', '4-6k', '6-8k', '8-10k', '10-15k', '15-20k']
    
    bin_prefill = []
    bin_decode = []
    bin_counts = []
    
    for i in range(len(input_bins) - 1):
        bin_results = [r for r in results 
                      if input_bins[i] <= r['input_length'] < input_bins[i+1]]
        if bin_results:
            bin_prefill.append(np.mean([r['prefill_time'] for r in bin_results]))
            bin_decode.append(np.mean([r['decode_time'] for r in bin_results]))
            bin_counts.append(len(bin_results))
        else:
            bin_prefill.append(0)
            bin_decode.append(0)
            bin_counts.append(0)
    
    x = np.arange(len(bin_labels))
    width = 0.6
    
    p1 = ax2.bar(x, bin_prefill, width, label='Prefill', alpha=0.8, color='#3498db', edgecolor='black')
    p2 = ax2.bar(x, bin_decode, width, bottom=bin_prefill, label='Decode', alpha=0.8, color='#e74c3c', edgecolor='black')
    
    ax2.set_xlabel('Input Length Range', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Time Breakdown by Input Length', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Decode time percentage by input length
    ax3 = fig.add_subplot(gs[0, 2])
    decode_pcts = []
    for i in range(len(input_bins) - 1):
        bin_results = [r for r in results 
                      if input_bins[i] <= r['input_length'] < input_bins[i+1]]
        if bin_results:
            avg_pct = np.mean([r['decode_pct'] for r in bin_results])
            decode_pcts.append(avg_pct)
        else:
            decode_pcts.append(0)
    
    bars = ax3.bar(x, decode_pcts, width, alpha=0.8, color='#9b59b6', edgecolor='black')
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax3.set_xlabel('Input Length Range', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Decode Time Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Decode Percentage by Input Length', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 100])
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, decode_pcts)):
        if pct > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Prefill time by output length
    ax4 = fig.add_subplot(gs[1, 0])
    output_bins = [0, 50, 100, 200, 300, 400, 500, 1000]
    output_labels = ['0-50', '50-100', '100-200', '200-300', '300-400', '400-500', '500+']
    
    output_prefill = []
    output_decode = []
    
    for i in range(len(output_bins) - 1):
        bin_results = [r for r in results 
                      if output_bins[i] <= r['output_length'] < output_bins[i+1]]
        if bin_results:
            output_prefill.append(np.mean([r['prefill_time'] for r in bin_results]))
            output_decode.append(np.mean([r['decode_time'] for r in bin_results]))
        else:
            output_prefill.append(0)
            output_decode.append(0)
    
    x_out = np.arange(len(output_labels))
    ax4.plot(x_out, output_prefill, marker='o', linewidth=2, markersize=8, label='Prefill', color='#3498db')
    ax4.plot(x_out, output_decode, marker='s', linewidth=2, markersize=8, label='Decode', color='#e74c3c')
    ax4.set_xlabel('Output Length Range', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Time by Output Length', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_out)
    ax4.set_xticklabels(output_labels, rotation=45, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Throughput (tokens/sec) by input length
    ax5 = fig.add_subplot(gs[1, 1])
    prefill_tps_bins = []
    decode_tps_bins = []
    
    for i in range(len(input_bins) - 1):
        bin_results = [r for r in results 
                      if input_bins[i] <= r['input_length'] < input_bins[i+1]]
        if bin_results:
            prefill_tps_bins.append(np.mean([r['prefill_tps'] for r in bin_results]))
            decode_tps_bins.append(np.mean([r['decode_tps'] for r in bin_results]))
        else:
            prefill_tps_bins.append(0)
            decode_tps_bins.append(0)
    
    ax5.plot(x, prefill_tps_bins, marker='o', linewidth=2, markersize=8, label='Prefill TPS', color='#2ecc71')
    ax5.plot(x, decode_tps_bins, marker='s', linewidth=2, markersize=8, label='Decode TPS', color='#f39c12')
    ax5.set_xlabel('Input Length Range', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Tokens per Second', fontsize=11, fontweight='bold')
    ax5.set_title('Throughput by Input Length', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Overall time distribution pie chart
    ax6 = fig.add_subplot(gs[1, 2])
    total_prefill = sum([r['prefill_time'] for r in results])
    total_decode = sum([r['decode_time'] for r in results])
    
    sizes = [total_prefill, total_decode]
    labels = [f'Prefill\n{total_prefill:.2f}s\n({total_prefill/(total_prefill+total_decode)*100:.1f}%)',
              f'Decode\n{total_decode:.2f}s\n({total_decode/(total_prefill+total_decode)*100:.1f}%)']
    colors = ['#3498db', '#e74c3c']
    explode = (0, 0.05)
    
    wedges, texts, autotexts = ax6.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='', shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax6.set_title('Overall Time Distribution', fontsize=12, fontweight='bold')
    
    # 7. Per-dataset comparison
    ax7 = fig.add_subplot(gs[2, :])
    x_dataset = np.arange(len(datasets))
    width = 0.35
    
    dataset_prefill = [np.mean([r['prefill_time'] for r in results if r['dataset'] == d]) for d in datasets]
    dataset_decode = [np.mean([r['decode_time'] for r in results if r['dataset'] == d]) for d in datasets]
    
    bars1 = ax7.bar(x_dataset - width/2, dataset_prefill, width, label='Prefill', 
                    alpha=0.8, color='#3498db', edgecolor='black')
    bars2 = ax7.bar(x_dataset + width/2, dataset_decode, width, label='Decode', 
                    alpha=0.8, color='#e74c3c', edgecolor='black')
    
    ax7.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Average Time (seconds)', fontsize=11, fontweight='bold')
    ax7.set_title('Average Time by Dataset', fontsize=12, fontweight='bold')
    ax7.set_xticks(x_dataset)
    ax7.set_xticklabels(datasets, rotation=45, ha='right')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('Prefill vs Decode Phase Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        print(f"{'='*80}\n")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefill and decode phases")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa"],
        help="LongBench datasets to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=3,
        help="Number of warmup runs before benchmarking"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=None,
        help="Maximum input length in tokens (filter samples with longer inputs)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON file"
    )
    parser.add_argument(
        "--fig",
        type=str,
        default=None,
        help="Path to save visualization figure. If provided with --save, will load from JSON and generate plots"
    )
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # If --fig is provided with --save, load existing results and generate plots
    if args.fig is not None and args.save is not None and os.path.exists(args.save):
        print("="*80)
        print("VISUALIZATION MODE")
        print("="*80)
        results = load_results(args.save)
        plot_results(results, args.fig)
        return
    
    # If --fig is provided without --save, error
    if args.fig is not None and args.save is None:
        print("ERROR: --fig requires --save to specify the JSON file path")
        print("Usage: --save results.json --fig output.png")
        return
    
    # Normal benchmark mode
    print("="*80)
    print("PREFILL & DECODE BENCHMARK")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Warmup runs: {args.warmup_runs}")
    if args.maxlen is not None:
        print(f"Max input length: {args.maxlen} tokens (filtering)")
    if args.save is not None:
        print(f"Results will be saved to: {args.save}")
    print("="*80)
    
    results = run_benchmark(
        model_name=args.model,
        datasets=args.datasets,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        maxlen=args.maxlen,
    )
    
    print_summary(results)
    
    # Save results if requested
    if args.save is not None:
        save_results(results, args.save)
        
        # Generate figure if also requested
        if args.fig is not None:
            plot_results(results, args.fig)


if __name__ == "__main__":
    main()
