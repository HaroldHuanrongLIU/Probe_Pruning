#!/bin/bash
# Sweep Probe_Pruning pruning ratios and measure GPU memory
# Usage: bash sweep_memory.sh

RATIOS="0.0 0.17 0.30 0.40 0.50 0.60 0.70"
BS=8
SEQ=2048

for RATIO in $RATIOS; do
    echo "=============================="
    echo "Running ratio=$RATIO bs=$BS seq=$SEQ"
    echo "=============================="
    uv run python test_model.py \
        --control_name "wikitext-2v1_llama-2-7b_clm_${BS}_${SEQ}_${RATIO}_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default" \
        --device cuda 2>&1 | grep -E "GPU Memory Report|Memory before|Peak memory|Forward overhead|Prune ratio|Batch size|evaluation_for_full|Perplexity"
    echo ""
done
