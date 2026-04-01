"""Measure actual GPU memory of Probe_Pruning at different pruning ratios.

Loads LLaMA-2 7B, wraps with LlamaPPModel, runs forward pass at bs=8 seq=2048,
measures torch.cuda.max_memory_allocated() for each pruning ratio.
"""
import gc
import sys
import torch
from transformers import AutoTokenizer

# Setup cfg before importing model code
from config import cfg, process_args

RATIOS = [0.0, 0.10, 0.17, 0.20, 0.30, 0.37, 0.40, 0.50]
BATCH_SIZE = 8
SEQ_LEN = 2048
DEVICE = "cuda"


def measure_one_ratio(ratio):
    """Load model, set pruning ratio, run forward, measure memory."""
    # Reset cfg for this ratio
    control_name = f"wikitext-2v1_llama-2-7b_clm_{BATCH_SIZE}_{SEQ_LEN}_{ratio}_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default"

    # Re-parse args
    args = {k: cfg[k] for k in cfg}
    args['control_name'] = control_name
    args['device'] = DEVICE
    process_args(args)

    from module import process_control
    process_control()

    # Load model
    from model.hf.modeling_llama import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="balanced"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    cfg['pad_token_id'] = tokenizer.pad_token_id

    # Wrap with pruning model
    from model.llama_pp import LlamaPPModel
    model = LlamaPPModel(model)

    # If ratio > 0, need calibration to set up pruning metrics
    # For memory measurement, we can skip calibration and just run forward
    # The pruning decisions are made per-batch in sync mode

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    mem_loaded = torch.cuda.memory_allocated()

    # Create input
    input_ids = torch.randint(1, 32000, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    cfg['cur_batch_index'] = 0
    cfg['pad_tokens'] = None
    cfg['nonpad_tokens_denominator'] = None
    cfg['calibration_stage'] = False

    # Forward pass
    with torch.no_grad():
        model.train(False)
        try:
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        except Exception as e:
            print(f"  Forward failed: {e}")
            del model, input_ids, attention_mask, labels
            torch.cuda.empty_cache()
            gc.collect()
            return None, None, None

    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_forward_overhead = mem_peak - mem_loaded

    del model, input_ids, attention_mask, labels, output
    torch.cuda.empty_cache()
    gc.collect()

    return mem_loaded, mem_peak, mem_forward_overhead


def main():
    print(f"Measuring Probe_Pruning GPU memory: bs={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print(f"{'Ratio':>8} {'Loaded':>12} {'Peak':>12} {'Overhead':>12}")
    print("-" * 48)

    results = []
    for ratio in RATIOS:
        print(f"\n>>> Testing ratio={ratio} ...", flush=True)
        loaded, peak, overhead = measure_one_ratio(ratio)
        if peak is not None:
            print(f"{ratio:>8.2f} {loaded/1e9:>10.2f}GB {peak/1e9:>10.2f}GB {overhead/1e9:>10.2f}GB")
            results.append({"ratio": ratio, "loaded_gb": loaded/1e9, "peak_gb": peak/1e9, "overhead_gb": overhead/1e9})
        else:
            print(f"{ratio:>8.2f}  FAILED")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ratio':>8} {'Peak GPU':>12}")
    print("-" * 24)
    for r in results:
        print(f"{r['ratio']:>8.2f} {r['peak_gb']:>10.2f}GB")

    # RAP reference
    print(f"\nRAP targets: 80% budget = 18.45 GB, 60% budget = 13.61 GB")


if __name__ == "__main__":
    main()
