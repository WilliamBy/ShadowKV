#!/usr/bin/env python3
"""
Analyze ShadowKV rank parameter test results.

This script reads the test results from the rank testing script and generates
comprehensive analysis including:
- Accuracy vs Rank plot
- Statistical analysis
- Recommendations for optimal rank selection
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def parse_summary_file(summary_file):
    """Parse the summary file to extract all metrics."""
    results = []
    
    with open(summary_file, 'r') as f:
        in_data = False
        for line in f:
            if '-----|----------|' in line:
                in_data = True
                continue
            if in_data and line.strip():
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        rank = int(parts[0].strip())
                        acc_str = parts[1].strip()
                        
                        if acc_str != 'FAILED' and acc_str != 'N/A':
                            # Extract numeric part
                            acc_val = float(re.findall(r'[\d.]+', acc_str)[0])
                            if acc_val > 1:
                                acc_val = acc_val / 100
                            
                            # Parse timing info if available
                            prefill_time = parts[2].strip().replace('s', '') if len(parts) > 2 else 'N/A'
                            decode_time = parts[3].strip().replace('s', '') if len(parts) > 3 else 'N/A'
                            total_time = parts[4].strip().replace('s', '') if len(parts) > 4 else 'N/A'
                            
                            # Convert to float
                            try:
                                prefill_time = float(prefill_time) if prefill_time != 'N/A' else None
                                decode_time = float(decode_time) if decode_time != 'N/A' else None
                                total_time = float(total_time) if total_time != 'N/A' else None
                            except ValueError:
                                prefill_time = decode_time = total_time = None
                            
                            results.append({
                                'rank': rank,
                                'accuracy': acc_val,
                                'prefill_time': prefill_time,
                                'decode_time': decode_time,
                                'total_time': total_time,
                                'log_file': parts[5].strip() if len(parts) > 5 else None
                            })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to parse line: {line.strip()}")
                        continue
    
    return results


def plot_analysis(results, output_dir):
    """Generate comprehensive analysis plots."""
    
    if not results:
        print("No valid data to analyze!")
        return
    
    # Extract data
    ranks = [r['rank'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Main plot: Accuracy vs Rank
    ax1 = fig.add_subplot(gs[0, :])
    
    # Sort by rank
    sorted_results = sorted(results, key=lambda x: x['rank'])
    ranks_sorted = [r['rank'] for r in sorted_results]
    accs_sorted = [r['accuracy'] for r in sorted_results]
    
    # Plot line
    ax1.plot(ranks_sorted, accs_sorted, marker='o', linewidth=2.5, 
            markersize=10, color='#2E86AB', label='Accuracy')
    
    # Highlight best point
    best_idx = accs_sorted.index(max(accs_sorted))
    ax1.scatter(ranks_sorted[best_idx], accs_sorted[best_idx], 
              s=300, color='#A23B72', zorder=5, marker='*',
              label=f"Best (rank={ranks_sorted[best_idx]}, acc={accs_sorted[best_idx]:.4f})",
              edgecolors='gold', linewidth=2)
    
    ax1.set_xlabel('Rank', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('ShadowKV: Rank vs Accuracy on ruler/qa1 (64k context, 16 samples)', 
                 fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xscale('log')
    
    # Add annotations
    for rank, acc in zip(ranks_sorted, accs_sorted):
        ax1.annotate(f'{acc:.4f}', 
                    xy=(rank, acc), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=9, alpha=0.8, ha='center')
    
    # 2. Accuracy distribution (histogram)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(accs_sorted, bins=min(10, len(accs_sorted)), 
            color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(accs_sorted), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(accs_sorted):.4f}')
    ax2.axvline(np.median(accs_sorted), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(accs_sorted):.4f}')
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Accuracy variance bar plot
    ax3 = fig.add_subplot(gs[1, 1])
    relative_acc = [(a - min(accs_sorted)) / (max(accs_sorted) - min(accs_sorted)) 
                    for a in accs_sorted]
    colors = ['#A23B72' if a == max(accs_sorted) else '#2E86AB' for a in accs_sorted]
    ax3.bar(range(len(ranks_sorted)), relative_acc, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(ranks_sorted)))
    ax3.set_xticklabels(ranks_sorted, rotation=45)
    ax3.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Accuracy by Rank', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Timing analysis (if available)
    ax4 = fig.add_subplot(gs[2, 0])
    timings = [(r['total_time'], r['rank']) for r in results if r['total_time'] is not None]
    if timings:
        times, rank_ts = zip(*timings)
        ax4.plot(rank_ts, times, marker='s', linewidth=2, markersize=8, 
                color='#F18F01', label='Total Time')
        ax4.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Total Time vs Rank', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
    else:
        ax4.text(0.5, 0.5, 'Timing data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Total Time vs Rank', fontsize=13, fontweight='bold')
    
    # 5. Efficiency plot (accuracy / time)
    ax5 = fig.add_subplot(gs[2, 1])
    efficiency = [(r['accuracy'], r['rank'], r['total_time']) 
                 for r in results if r['total_time'] is not None and r['total_time'] > 0]
    if efficiency:
        accs_eff, ranks_eff, times_eff = zip(*efficiency)
        eff_scores = [a / t for a, t in zip(accs_eff, times_eff)]
        ax5.plot(ranks_eff, eff_scores, marker='^', linewidth=2, markersize=8, 
                color='#C73E1D', label='Accuracy/Time')
        ax5.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Efficiency (acc/time)', fontsize=12, fontweight='bold')
        ax5.set_title('Efficiency vs Rank', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
    else:
        ax5.text(0.5, 0.5, 'Efficiency data not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Efficiency vs Rank', fontsize=13, fontweight='bold')
    
    # Overall title
    fig.suptitle('ShadowKV Rank Parameter Analysis', fontsize=17, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = os.path.join(output_dir, 'rank_analysis_comprehensive.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive plot saved to: {output_path}")
    
    plt.close()


def generate_statistics(results, output_dir):
    """Generate statistical analysis."""
    
    if not results:
        return
    
    ranks = [r['rank'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Find best
    best_result = max(results, key=lambda x: x['accuracy'])
    
    # Statistics
    stats = {
        'best_rank': best_result['rank'],
        'best_accuracy': best_result['accuracy'],
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'median_accuracy': float(np.median(accuracies)),
        'min_accuracy': float(min(accuracies)),
        'max_accuracy': float(max(accuracies)),
        'num_tests': len(results),
        'tested_ranks': sorted(ranks),
    }
    
    # Range analysis
    acc_range = max(accuracies) - min(accuracies)
    stats['accuracy_range'] = float(acc_range)
    
    # Coefficient of variation
    if stats['mean_accuracy'] > 0:
        stats['cv'] = float(stats['std_accuracy'] / stats['mean_accuracy'])
    else:
        stats['cv'] = 0
    
    # Generate recommendations
    recommendations = []
    
    # If accuracy varies significantly with rank
    if stats['cv'] > 0.01:
        recommendations.append("⚠️  Accuracy varies significantly with rank - careful rank selection is important")
    else:
        recommendations.append("✓ Accuracy is relatively stable across different rank values")
    
    # Optimal rank range
    if acc_range < 0.01:
        recommendations.append(f"✓ Any rank in tested range performs similarly")
    else:
        # Find ranks within 1% of best
        threshold = best_result['accuracy'] - 0.01
        good_ranks = [r['rank'] for r in results if r['accuracy'] >= threshold]
        recommendations.append(f"💡 Recommended ranks (within 1% of best): {sorted(good_ranks)}")
    
    # Memory/Performance trade-off
    if stats['best_rank'] <= stats['mean_accuracy']:
        recommendations.append(f"💡 Lower rank ({stats['best_rank']}) achieves best accuracy - good for memory efficiency")
    else:
        recommendations.append(f"💡 Best accuracy at rank={stats['best_rank']} - balance between accuracy and memory")
    
    stats['recommendations'] = recommendations
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'rank_statistics.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    print(f"Number of tests: {stats['num_tests']}")
    print(f"Tested ranks: {stats['tested_ranks']}")
    print(f"\nBest Performance:")
    print(f"  Rank: {stats['best_rank']}")
    print(f"  Accuracy: {stats['best_accuracy']:.4f}")
    print(f"\nAccuracy Statistics:")
    print(f"  Mean: {stats['mean_accuracy']:.4f}")
    print(f"  Std:  {stats['std_accuracy']:.4f}")
    print(f"  Median: {stats['median_accuracy']:.4f}")
    print(f"  Range: {stats['accuracy_range']:.4f}")
    print(f"  CV: {stats['cv']:.4f}")
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    print(f"\nStatistics saved to: {output_path}")
    
    return stats


def generate_markdown_report(results, stats, output_dir):
    """Generate a markdown report."""
    
    output_path = os.path.join(output_dir, 'RANK_TEST_REPORT.md')
    
    with open(output_path, 'w') as f:
        f.write("# ShadowKV Rank Parameter Test Report\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Dataset**: ruler/qa1\n")
        f.write("- **Data Length**: 64k tokens\n")
        f.write("- **Samples**: 16\n")
        f.write("- **Method**: ShadowKV\n")
        f.write("- **Sparse Budget**: 2048\n")
        f.write("- **Chunk Size**: 64\n\n")
        
        f.write("## Results Summary\n\n")
        f.write(f"- **Tests Run**: {stats['num_tests']}\n")
        f.write(f"- **Best Rank**: {stats['best_rank']}\n")
        f.write(f"- **Best Accuracy**: {stats['best_accuracy']:.4f}\n")
        f.write(f"- **Mean Accuracy**: {stats['mean_accuracy']:.4f}\n")
        f.write(f"- **Std Deviation**: {stats['std_accuracy']:.4f}\n")
        f.write(f"- **Accuracy Range**: {stats['accuracy_range']:.4f}\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| Rank | Accuracy | Log File |\n")
        f.write("|------|----------|----------|\n")
        for r in sorted(results, key=lambda x: x['rank']):
            log_file = r['log_file'].split('/')[-1] if r['log_file'] else 'N/A'
            f.write(f"| {r['rank']} | {r['accuracy']:.4f} | {log_file} |\n")
        
        f.write("\n## Recommendations\n\n")
        for rec in stats['recommendations']:
            f.write(f"{rec}\n")
        
        f.write("\n## Plots\n\n")
        f.write("- Comprehensive Analysis: `rank_analysis_comprehensive.png`\n")
        f.write("- Simple Rank vs Accuracy: `rank_summary_plot.png`\n\n")
    
    print(f"✓ Markdown report saved to: {output_path}")


def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_rank_results.py <summary_file> [output_dir]")
        print("\nExample:")
        print("  python analyze_rank_results.py archive/test_logs/rank_tests/rank_summary.txt")
        print("  python analyze_rank_results.py archive/test_logs/rank_tests/rank_summary.txt custom_output_dir")
        sys.exit(1)
    
    summary_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(summary_file)
    
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)
    
    print(f"Analyzing results from: {summary_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Parse results
    results = parse_summary_file(summary_file)
    
    if not results:
        print("Error: No valid results found in summary file!")
        sys.exit(1)
    
    print(f"✓ Parsed {len(results)} results")
    print(f"  Ranks: {sorted([r['rank'] for r in results])}")
    print(f"  Accuracy range: {min(r['accuracy'] for r in results):.4f} - {max(r['accuracy'] for r in results):.4f}")
    print()
    
    # Generate plots
    print("Generating analysis plots...")
    plot_analysis(results, output_dir)
    
    # Generate statistics
    print("\nGenerating statistical analysis...")
    stats = generate_statistics(results, output_dir)
    
    # Generate markdown report
    print("\nGenerating markdown report...")
    generate_markdown_report(results, stats, output_dir)
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    print(f"Outputs in: {output_dir}")
    print(f"  - rank_analysis_comprehensive.png: Multi-panel analysis")
    print(f"  - rank_statistics.json: Statistical data")
    print(f"  - RANK_TEST_REPORT.md: Human-readable report")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
