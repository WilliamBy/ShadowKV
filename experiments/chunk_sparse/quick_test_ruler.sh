#!/bin/bash
# 快速测试Ruler QA1数据集的Per-Head分析

set -e

echo "=========================================="
echo "Ruler QA1 Per-Head 快速测试"
echo "=========================================="

cd /root/workspace/ShadowKV

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ 虚拟环境已激活"
fi

# 运行快速测试（2个layer，2个head，16个生成tokens）
echo ""
echo "配置:"
echo "  数据集: Ruler QA1 (128K)"
echo "  样本数: 1"
echo "  分析层: Layer 0, 16"
echo "  分析头: Head 0, 4"
echo "  生成tokens: 16"
echo "  块大小: 128"
echo ""

python experiments/chunk_sparse/analyze_key_dispersion_per_head.py \
    --ruler \
    --ruler_data_path data/ruler/data/llama-3/131072/qa_1/validation.jsonl \
    --num_samples 1 \
    --chunk_size 8 \
    --max_new_tokens 1 \
    --analyze_layers "0,8,16" \
    --analyze_heads "0,4,7" \
    --output_dir experiments/chunk_sparse/results_per_head_ruler_quick

echo ""
echo "=========================================="
echo "✓ 测试完成！"
echo "=========================================="
echo ""
echo "结果位置:"
echo "  experiments/chunk_sparse/results_per_head_ruler_quick/"
echo ""
echo "查看图表:"
ls -lh experiments/chunk_sparse/results_per_head_ruler_quick/sample_0/*.png 2>/dev/null || echo "  （图表生成中...）"
