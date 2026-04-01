# Probe Pruning Baseline Experiment Results

> LLaMA-2 7B, WikiText-2 perplexity evaluation. For RAP (Runtime Adaptive Pruning) ICML 2026 rebuttal reference.

## 1. Experiment Setup

- **Model**: LLaMA-2 7B (meta-llama/Llama-2-7b-hf, FP16)
- **Evaluation dataset**: WikiText-2 (wikitext-2-raw-v1, test split)
- **Calibration dataset**: WikiText-103 (wikitext-103-raw-v1, train split, filtered for long texts, 2000 samples)
- **Pruning method**: Probe Pruning (ppwandasp, sync mode, probe-default)
- **Probe config**: ratio 0.5+0.05 for q/k/v/gate/up, selection strategy seqrank+bszrank
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB)
- **Memory measurement**: `torch.cuda.max_memory_allocated()` during inference (after warmup)
- **Codebase**: [Probe_Pruning](https://github.com/HaroldHuanrongLIU/Probe_Pruning) (ICLR 2025)

## 2. Results Summary

### 2.1 Memory & Perplexity at Different Pruning Ratios

| Method | Prune Ratio | bs | seq_len | Peak GPU (GB) | PPL (WikiText-2) |
|--------|------------|-----|---------|--------------|-----------------|
| LLaMA-2 7B (dense) | 0.0 | 8 | 2048 | 23.70 | 5.47 |
| Probe Pruning | 0.17 | 8 | 2048 | **24.78** | 6.63 |
| Probe Pruning | 0.50 | 8 | 2048 | **24.79** | 20.88 |
| Probe Pruning | 0.17 | 4 | 1024 | - | 7.04 |
| **RAP (80% budget)** | block-level | 8 | 2048 | **18.45** | 9.02 |
| **RAP (60% budget)** | block-level | 8 | 2048 | **13.61** | 28.67 |

### 2.2 Key Finding: Probe Pruning Does Not Reduce GPU Memory

Probe Pruning's peak GPU memory is **constant at ~24.79 GB regardless of pruning ratio** (even higher than the 23.70 GB dense model). Two data points confirm this:

- ratio=0.17: peak = 24.78 GB
- ratio=0.50: peak = 24.79 GB

This is because Probe Pruning:
1. **Retains all weights on GPU** (~13.48 GB parameters are never released)
2. **Does not reduce KV cache** (all 32 MHA layers still allocate full KV cache)
3. **Adds probe overhead** (~1 GB extra for probing computations)
4. Only reduces FLOPs by selecting a subset of channels during `F.linear()`, but the full weight matrices remain allocated

In contrast, RAP's block-level pruning **physically removes** entire MHA/FFN sub-blocks:
- Releases pruned block parameters from GPU memory
- Eliminates KV cache for pruned MHA blocks
- Achieves real GPU memory savings: 18.45 GB (80% budget) and 13.61 GB (60% budget)

## 3. Detailed Memory Breakdown

### 3.1 RAP Memory Model (Verified)

Verified with `scripts/verify_memory_model.py` using `torch.cuda.max_memory_allocated()`:

| Config | Theory Peak | Actual Peak | Actual Ratio |
|--------|------------|------------|-------------|
| Full model | 22.07 GB | 23.70 GB | 1.0000 |
| RAP 80% budget | 17.36 GB | 18.45 GB | 0.7787 |
| RAP 60% budget | 13.05 GB | 13.61 GB | 0.5742 |

Theory (param + KV cache) is consistently ~1-2 GB below actual (due to activation memory overhead).

### 3.2 Probe Pruning Memory Breakdown (bs=8, seq_len=2048)

| Component | Memory |
|-----------|--------|
| Model parameters (full, not released) | 13.48 GB |
| Calibration overhead (EMA metrics, probe tensors) | ~1.84 GB |
| **Memory before inference** | **15.32 GB** |
| KV cache (32 layers, full) | ~8.59 GB |
| Activation & probe overhead | ~0.87 GB |
| **Peak memory during inference** | **24.78 GB** |

## 4. Implications for Comparison

### 4.1 Why Memory-Matched Comparison is Not Possible

Probe Pruning **cannot match RAP's GPU memory** at any pruning ratio because it is a **dynamic channel-level** method that:
- Keeps all weights on GPU for runtime channel selection
- Cannot release weights since the pruning decision changes per-batch
- Cannot reduce KV cache since attention heads are not removed, only partially selected

Therefore, a fair memory-matched comparison between RAP and Probe Pruning is not feasible under our methodology (matching actual GPU memory consumption under same bs/seq_len).

### 4.2 What Probe Pruning Does Reduce

Probe Pruning reduces **FLOPs** (not memory):

| Ratio | Pruned FLOPs (Million) | Speedup vs Dense |
|-------|----------------------|-----------------|
| 0.17 | 3,996,300,466 | ~1.2x |
| 0.50 | 2,521,328,711 | ~1.9x |

### 4.3 Fundamental Difference in Pruning Paradigm

| Aspect | RAP (Ours) | Probe Pruning |
|--------|-----------|---------------|
| **Granularity** | Block-level (entire MHA/FFN) | Channel-level (weight dimensions) |
| **Pruning type** | Static per-config (adaptive across configs) | Dynamic per-batch |
| **Weight release** | Yes (forward patching, skip blocks) | No (full weights retained) |
| **KV cache reduction** | Yes (pruned MHA = no KV) | No (all heads active) |
| **GPU memory saved** | Yes (real memory reduction) | No (FLOPs only) |
| **Runtime adaptive** | Yes (adapts to bs/seq_len/budget) | Partially (adapts per-batch, but no memory control) |

## 5. RAP Pruning Configs (for reference)

### 80% Memory Budget (bs=8, seq_len=2048)
- Pruned MHA: [18, 19, 21, 23, 24, 25, 27, 28, 30] (9 blocks)
- Pruned FFN: [13, 22, 24, 27] (4 blocks)
- Total: 13/64 blocks removed, 17.0% parameter reduction
- PPL: 9.02

### 60% Memory Budget (bs=8, seq_len=2048)
- Pruned MHA: [8, 10, 12, 13, 14, 18, 19, 21, 22, 23, 24, 25, 27, 28, 30] (15 blocks)
- Pruned FFN: [7, 8, 9, 12, 13, 18, 20, 22, 24, 26, 27] (11 blocks)
- Total: 26/64 blocks removed, 37.0% parameter reduction
- PPL: 28.67

## 6. Reproducibility

### Environment
```
torch==2.11.0+cu130
transformers==4.57.6
accelerate>=0.22.0
deepspeed>=0.12.3
GPU: NVIDIA GeForce RTX 5090, Driver 590.48.01
```

### Commands
```bash
cd baselines/Probe_Pruning/src

# ratio=0.17, bs=8, seq_len=2048
uv run python test_model.py \
  --control_name "wikitext-2v1_llama-2-7b_clm_8_2048_0.17_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default" \
  --device cuda

# ratio=0.50, bs=8, seq_len=2048
uv run python test_model.py \
  --control_name "wikitext-2v1_llama-2-7b_clm_8_2048_0.50_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default" \
  --device cuda
```

### Code Modifications
The following modifications were made to the original Probe_Pruning codebase for compatibility:
1. `model/huggingface.py`: Load LLaMA-2 from HuggingFace Hub instead of local `output/` path
2. `module/hyper.py`: Store CUDA streams outside `cfg` dict to avoid pickle errors with `datasets.map()`
3. `dataset/dataset.py`: Replace C4 local loading with WikiText-103 (filtered for long texts); replace `num_proc=1` with `num_proc=None` in `datasets.map()`; rewrite calibration `preprocess_function` to concatenate texts then slice windows (avoids infinite loop when no single text exceeds `max_seq_len`)
4. `test_model.py`: Add `torch.cuda.max_memory_allocated()` reporting after inference; temporarily remove non-picklable objects from `cfg` before `process_dataset`
