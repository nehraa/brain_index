#!/usr/bin/env python3
"""
Brain Index — Full 100K Activation Collection Script

Purpose: Run 100,000 diverse prompts through Qwen3-8B, capture sparse neuron
activations at 8 layers, and write results to CSV. Designed for Kaggle T4 GPU.

Features:
- Progress bar (tqdm) for visual tracking
- Batch-level CSV writes every 5000 prompts (crash recovery)
- Resumable via checkpoint file
- Silent except every 1000 prompts + final summary
- Checkpoint: last processed prompt index, loaded on restart

Usage:
    python collect_activations.py

Expected runtime: ~33-83 minutes for 100K prompts
Expected cost: ~$3-5 on Kaggle T4

Resume after crash:
    python collect_activations.py --resume
"""

import os
import sys
import ast
import time
import json
import argparse
import traceback
from typing import List, Dict, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import our utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from activation_extractor import extract_sparse_neurons, compute_layer_statistics
from sparse_storage import encode_sparse, decode_sparse, verify_encoding
from dataset_builder import build_diverse_dataset

# ==============================================================================
# CONFIGURATION — all in one place
# ==============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS_TO_CAPTURE = [4, 9, 14, 19, 24, 29, 34, 35]
TOP_K = 300

# 100K across 7 domains (proportional to original plan)
DOMAIN_CONFIG_FULL = {
    "reasoning": 20000,
    "creative": 15000,
    "code": 15000,
    "general": 20000,
    "summarization": 10000,
    "translation": 10000,
    "agentic": 10000,
}  # Total = 100,000

OUTPUT_DIR = "brain_index/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "brain_index_activations.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
BATCH_WRITE_INTERVAL = 5000  # Write CSV every 5000 prompts

# ==============================================================================
# CHECKPOINT MANAGEMENT
# ==============================================================================

def load_checkpoint() -> Optional[int]:
    """Load checkpoint to resume from previous run. Returns prompt index or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        last_index = data.get("last_prompt_index", None)
        print(f"[CHECKPOINT] Resuming from prompt index: {last_index}")
        return last_index
    except Exception as e:
        print(f"[CHECKPOINT] WARNING: Could not load checkpoint: {e}")
        return None


def save_checkpoint(prompt_index: int):
    """Save checkpoint after each batch."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "last_prompt_index": prompt_index,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f)
    except Exception as e:
        print(f"[CHECKPOINT] WARNING: Could not save checkpoint: {e}")


# ==============================================================================
# MAIN COLLECTION FUNCTION
# ==============================================================================

def collect_activations(resume: bool = False, verify_only: bool = False):
    """Run the full activation collection pipeline."""

    print("=" * 80)
    print("[MAIN] Brain Index — Activation Collection")
    print("=" * 80)
    print(f"[MAIN] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MAIN] Resume mode: {resume}")
    print(f"[MAIN] Verify only (100 prompts): {verify_only}")

    # -------------------------------------------------------------------------
    # Step 0: Setup
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SETUP] Output directory: {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Step 1: Configuration
    # -------------------------------------------------------------------------
    print(f"\n[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] Layers: {LAYERS_TO_CAPTURE}")
    print(f"[CONFIG] Top-K: {TOP_K}")

    # Use full domain config or test config
    domain_config = DOMAIN_CONFIG_FULL.copy() if not verify_only else {
        "reasoning": 15, "creative": 10, "code": 15, "general": 25,
        "summarization": 10, "translation": 10, "agentic": 15
    }
    total_prompts = sum(domain_config.values())
    print(f"[CONFIG] Domain config: {domain_config}")
    print(f"[CONFIG] Total prompts: {total_prompts}")

    # -------------------------------------------------------------------------
    # Step 2: GPU Check
    # -------------------------------------------------------------------------
    print(f"\n[GPU] CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[GPU] ERROR: CUDA not available. This script requires GPU.")
        if verify_only:
            print("[GPU] NOTE: verify_only=True, continuing anyway...")
        else:
            raise RuntimeError("GPU required for full collection")

    # -------------------------------------------------------------------------
    # Step 3: Load Model
    # -------------------------------------------------------------------------
    print(f"\n[MODEL] Loading Qwen3-8B with Q4 quantization...")
    load_start = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="sdpa"
    )
    model.eval()

    load_elapsed = time.time() - load_start
    print(f"[MODEL] Loaded in {load_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # Step 4: Register Hooks
    # -------------------------------------------------------------------------
    print(f"\n[HOOKS] Registering {len(LAYERS_TO_CAPTURE)} hooks...")
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            captured[layer_idx] = output.detach().cpu()
        return hook

    for layer_idx in LAYERS_TO_CAPTURE:
        layer = model.model.model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    print(f"[HOOKS] Registered {len(handles)} hooks")

    # -------------------------------------------------------------------------
    # Step 5: Load Tokenizer
    # -------------------------------------------------------------------------
    print(f"\n[TOKENIZER] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[TOKENIZER] Loaded, vocab_size={tokenizer.vocab_size}")

    # -------------------------------------------------------------------------
    # Step 6: Build Dataset
    # -------------------------------------------------------------------------
    print(f"\n[DATASET] Building dataset...")
    dataset_start = time.time()
    prompts = build_diverse_dataset(domain_config)
    dataset_elapsed = time.time() - dataset_start
    print(f"[DATASET] Built {len(prompts)} prompts in {dataset_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # Step 7: Determine Start Index (for resume)
    # -------------------------------------------------------------------------
    start_index = 0
    if resume:
        checkpoint_index = load_checkpoint()
        if checkpoint_index is not None:
            start_index = checkpoint_index + 1
            print(f"[RESUME] Starting from index {start_index}")
        else:
            print(f"[RESUME] No checkpoint found, starting from 0")

    # -------------------------------------------------------------------------
    # Step 8: Run Collection Loop
    # -------------------------------------------------------------------------
    print(f"\n[RUN] Starting collection: {len(prompts)} prompts (start={start_index})")
    run_start = time.time()

    results: List[Dict] = []

    # Load existing results if resuming (for append)
    if resume and os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            results = existing_df.to_dict("records")
            print(f"[RESUME] Loaded {len(results)} existing rows from CSV")
        except Exception as e:
            print(f"[RESUME] WARNING: Could not load existing CSV: {e}")

    # Main loop with tqdm
    progress_bar = tqdm(total=len(prompts), initial=start_index, desc="Collecting", unit="prompts")

    for i in range(start_index, len(prompts)):
        prompt = prompts[i]

        # ---- Tokenize ----
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        # ---- Clear & Forward ----
        captured.clear()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=inputs.attention_mask.to(model.device))

        # ---- Extract Sparse ----
        row = {
            "prompt_id": i,
            "prompt_text": prompt[:100],
        }
        for layer_idx in LAYERS_TO_CAPTURE:
            sparse = extract_sparse_neurons(captured[layer_idx], top_k=TOP_K)
            row[f"layer_{layer_idx}"] = encode_sparse(sparse)

        results.append(row)

        # ---- Progress Update ----
        progress_bar.update(1)

        # ---- Batch Write ----
        if (i + 1) % BATCH_WRITE_INTERVAL == 0:
            # Write intermediate CSV
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_FILE, index=False)
            save_checkpoint(i)
            elapsed = time.time() - run_start
            rate = (i + 1 - start_index) / elapsed
            eta = (len(prompts) - i - 1) / rate if rate > 0 else 0
            print(f"\n[BATCH] Written {len(results)} rows to CSV at prompt {i+1}")
            print(f"[BATCH] Rate: {rate:.1f} prompts/sec, ETA: {eta/60:.1f} min")

        # ---- Silent periodic (every 1000) ----
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - run_start
            rate = (i + 1 - start_index) / elapsed
            progress_bar.set_postfix({"rate": f"{rate:.1f}/s"})

    progress_bar.close()

    run_elapsed = time.time() - run_start
    print(f"\n[RUN] Completed {len(prompts)} prompts in {run_elapsed:.1f}s")
    print(f"[RUN] Rate: {len(prompts) / run_elapsed:.2f} prompts/sec")

    # -------------------------------------------------------------------------
    # Step 9: Final Write
    # -------------------------------------------------------------------------
    print(f"\n[CSV] Writing final CSV...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[CSV] Wrote {len(df)} rows to {OUTPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"[CSV] File size: {file_size / 1e6:.2f} MB")

    # -------------------------------------------------------------------------
    # Step 10: Summary Statistics
    # -------------------------------------------------------------------------
    print(f"\n[SUMMARY] Activation Statistics:")
    for layer_idx in LAYERS_TO_CAPTURE:
        col = f"layer_{layer_idx}"
        counts = [len(ast.literal_eval(r[col])) for r in results[:100] if r[col] != "[]"]
        avg = sum(counts) / len(counts) if counts else 0
        print(f"  Layer {layer_idx}: avg={avg:.0f} neurons (sample of 100)")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    print(f"\n[CLEANUP] Removing hooks...")
    for h in handles:
        h.remove()

    # Remove checkpoint (successful completion)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n[FINISH] Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[FINISH] Output: {OUTPUT_FILE}")
    print(f"[FINISH] Rows: {len(results)}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Index 100K Activation Collection")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--verify", action="store_true", help="Run only 100 prompts (verification)")
    args = parser.parse_args()

    try:
        collect_activations(resume=args.resume, verify_only=args.verify)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Collection interrupted by user")
        print("[INTERRUPT] Progress saved. Run with --resume to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Collection failed: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        sys.exit(1)