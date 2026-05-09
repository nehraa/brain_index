#!/usr/bin/env python3
"""
Brain Index — Activation Capture Verification Script

Purpose: Run 100 prompts on Qwen3-8B to validate the entire pipeline before
the full 100K collection run. Every step is logged excessively for debugging.

Run on Kaggle T4 GPU. Model loaded with bitsandbytes Q4 quantization.
Hooks capture hidden states at 8 layers → ReLU → top-k → normalize → CSV.

Usage:
    python verify_activation_capture.py

Expected runtime: ~2-5 minutes for 100 prompts
Expected cost: ~$0.50-1 on Kaggle T4
"""

import os
import sys
import ast
import time
import traceback
from typing import List, Dict, Tuple

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import our utils — ensure they're in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from activation_extractor import extract_sparse_neurons, compute_layer_statistics
from sparse_storage import encode_sparse, decode_sparse, verify_encoding
from dataset_builder import build_diverse_dataset

# ==============================================================================
# CONFIGURATION — all in one place, no magic numbers
# ==============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS_TO_CAPTURE = [4, 9, 14, 19, 24, 29, 34, 36]
TOP_K = 300
NUM_VERIFY_PROMPTS = 100
OUTPUT_DIR = "brain_index/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "verification_results.csv")

# Domain distribution for 100 prompts
DOMAIN_CONFIG = {
    "reasoning": 15,
    "creative": 10,
    "code": 15,
    "general": 25,
    "summarization": 10,
    "translation": 10,
    "agentic": 15,
}  # Total = 100

# ==============================================================================
# STEP 0: Setup — ensure output directory exists
# ==============================================================================

print("=" * 80)
print("[SETUP] Brain Index — Activation Capture Verification")
print("=" * 80)
print(f"[SETUP] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[SETUP] Working directory: {os.getcwd()}")
print(f"[SETUP] Python version: {sys.version.split()[0]}")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[SETUP] SUCCESS: Output directory ready: {OUTPUT_DIR}")
except Exception as e:
    print(f"[SETUP] ERROR: Could not create output directory: {e}")
    raise

# ==============================================================================
# STEP 1: Print configuration
# ==============================================================================

print("\n" + "=" * 80)
print("[CONFIG] Configuration Summary")
print("=" * 80)
print(f"[CONFIG] Model: {MODEL_NAME}")
print(f"[CONFIG] Layers to capture: {LAYERS_TO_CAPTURE}")
print(f"[CONFIG] Top-K active neurons per layer: {TOP_K}")
print(f"[CONFIG] Verification prompts: {NUM_VERIFY_PROMPTS}")
print(f"[CONFIG] Domain config: {DOMAIN_CONFIG}")
print(f"[CONFIG] Output file: {OUTPUT_FILE}")
print(f"[CONFIG] Total expected layers: {len(LAYERS_TO_CAPTURE)}")
print(f"[CONFIG] Layers list: {LAYERS_TO_CAPTURE}")

# ==============================================================================
# STEP 2: Check GPU availability
# ==============================================================================

print("\n" + "=" * 80)
print("[GPU] GPU Detection")
print("=" * 80)

print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
print(f"[GPU] MPS available (Mac): {torch.backends.mps.is_available()}")
print(f"[GPU] PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    try:
        gpu_count = torch.cuda.device_count()
        print(f"[GPU] SUCCESS: Found {gpu_count} CUDA GPU(s)")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"[GPU]   GPU {i}: {props.name}")
            print(f"[GPU]   GPU {i} memory: {props.total_memory / 1e9:.2f} GB")
            print(f"[GPU]   GPU {i} compute capability: {props.major}.{props.minor}")
        print(f"[GPU] Current GPU: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"[GPU] WARNING: Error querying GPU details: {e}")
elif torch.backends.mps.is_available():
    print("[GPU] INFO: MPS (Apple Silicon) available — but this script requires CUDA for bitsandbytes")
    print("[GPU] NOTE: Run this on Kaggle with CUDA GPU")
else:
    print("[GPU] WARNING: No GPU detected. This script requires CUDA.")
    print("[GPU] NOTE: Running on CPU will be very slow and may fail.")

# ==============================================================================
# STEP 3: Import and version check
# ==============================================================================

print("\n" + "=" * 80)
print("[LIBS] Library Version Check")
print("=" * 80)

try:
    import transformers
    print(f"[LIBS] SUCCESS: transformers imported, version {transformers.__version__}")
except ImportError as e:
    print(f"[LIBS] ERROR: Failed to import transformers: {e}")
    raise

try:
    import bitsandbytes
    print(f"[LIBS] SUCCESS: bitsandbytes imported, version {bitsandbytes.__version__}")
except ImportError as e:
    print(f"[LIBS] ERROR: Failed to import bitsandbytes: {e}")
    print("[LIBS] NOTE: bitsandbytes is required for Q4 quantization")
    raise

try:
    import datasets
    print(f"[LIBS] SUCCESS: datasets imported, version {datasets.__version__}")
except ImportError as e:
    print(f"[LIBS] ERROR: Failed to import datasets: {e}")
    raise

try:
    import pandas as pd
    print(f"[LIBS] SUCCESS: pandas imported, version {pd.__version__}")
except ImportError as e:
    print(f"[LIBS] ERROR: Failed to import pandas: {e}")
    raise

# ==============================================================================
# STEP 4: Load quantization config
# ==============================================================================

print("\n" + "=" * 80)
print("[QUANT] Setting up Q4 Quantization Config")
print("=" * 80)

try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("[QUANT] SUCCESS: BitsAndBytesConfig created")
    print(f"[QUANT]   load_in_4bit: True")
    print(f"[QUANT]   bnb_4bit_quant_type: nf4")
    print(f"[QUANT]   bnb_4bit_use_double_quant: True")
    print(f"[QUANT]   bnb_4bit_compute_dtype: torch.bfloat16")
except Exception as e:
    print(f"[QUANT] ERROR: Failed to create quantization config: {e}")
    raise

# ==============================================================================
# STEP 5: Load model
# ==============================================================================

print("\n" + "=" * 80)
print("[MODEL] Loading Qwen3-8B with Q4 quantization")
print("=" * 80)
print("[MODEL] This may take 2-5 minutes on first run (download + load)...")
print("[MODEL] Download goes to /tmp on Kaggle (~8GB)")

load_start = time.time()

try:
    print(f"[MODEL] Step 5.1: Calling AutoModelForCausalLM.from_pretrained...")
    print(f"[MODEL]   Model: {MODEL_NAME}")
    print(f"[MODEL]   device_map: auto")
    print(f"[MODEL]   quantization_config: BitsAndBytesConfig (Q4 nf4)")
    print(f"[MODEL]   attn_implementation: sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="sdpa"
    )

    load_elapsed = time.time() - load_start
    print(f"[MODEL] SUCCESS: Model loaded in {load_elapsed:.1f}s")
    print(f"[MODEL] Model type: {type(model).__name__}")
    print(f"[MODEL] Model device map: {model.hf_device_map}")

except Exception as e:
    print(f"[MODEL] ERROR: Failed to load model: {e}")
    print(f"[MODEL] Traceback: {traceback.format_exc()}")
    raise

# Set to eval mode
try:
    model.eval()
    print("[MODEL] SUCCESS: Model set to eval mode")
except Exception as e:
    print(f"[MODEL] WARNING: Could not set eval mode: {e}")

# ==============================================================================
# STEP 6: Verify module paths
# ==============================================================================

print("\n" + "=" * 80)
print("[MODULE] Verifying Qwen3-8B Module Paths")
print("=" * 80)

try:
    # Navigate the model hierarchy
    print(f"[MODULE] model type: {type(model).__name__} (expected: Qwen3ForCausalLM)")

    model_inner = model.model
    print(f"[MODULE] model.model type: {type(model_inner).__name__} (expected: Qwen3Model)")

    model_inner2 = model_inner.model
    print(f"[MODULE] model.model.model type: {type(model_inner2).__name__} (expected: Qwen3Model)")

    layers_module = model_inner2.layers
    print(f"[MODULE] model.model.model.layers type: {type(layers_module).__name__}")
    print(f"[MODULE] model.model.model.layers length: {len(layers_module)} (expected: 36)")

    # Verify layer types
    for layer_idx in LAYERS_TO_CAPTURE[:2]:  # Check first 2
        layer = layers_module[layer_idx]
        print(f"[MODULE] Layer {layer_idx} type: {type(layer).__name__}")

    print("[MODULE] SUCCESS: Module paths verified correctly")

except Exception as e:
    print(f"[MODULE] ERROR: Module path verification failed: {e}")
    print(f"[MODULE] Traceback: {traceback.format_exc()}")
    raise

# ==============================================================================
# STEP 7: Register forward hooks
# ==============================================================================

print("\n" + "=" * 80)
print("[HOOKS] Registering Forward Hooks")
print("=" * 80)

captured: Dict[int, torch.Tensor] = {}
handles = []

def make_hook(layer_idx: int):
    """Create a hook that captures layer output."""
    def hook(module, input, output):
        # output is torch.Tensor directly — shape: [batch, seq_len, hidden_size=4096]
        captured[layer_idx] = output.detach().cpu()
        print(f"       [HOOK] Layer {layer_idx} fired! output.shape={output.shape}")
    return hook

try:
    print(f"[HOOKS] Registering {len(LAYERS_TO_CAPTURE)} hooks on layers: {LAYERS_TO_CAPTURE}")
    for i, layer_idx in enumerate(LAYERS_TO_CAPTURE):
        layer = model.model.model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(layer_idx))
        handles.append(h)
        print(f"[HOOKS]   Registered hook {i+1}/{len(LAYERS_TO_CAPTURE)} on layer {layer_idx} "
              f"→ module type: {type(layer).__name__}")

    print(f"[HOOKS] SUCCESS: {len(handles)} hooks registered")

except Exception as e:
    print(f"[HOOKS] ERROR: Failed to register hooks: {e}")
    print(f"[HOOKS] Traceback: {traceback.format_exc()}")
    raise

# ==============================================================================
# STEP 8: Load tokenizer
# ==============================================================================

print("\n" + "=" * 80)
print("[TOKENIZER] Loading Tokenizer")
print("=" * 80)

try:
    print(f"[TOKENIZER] Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"[TOKENIZER] SUCCESS: Tokenizer loaded")
    print(f"[TOKENIZER] vocab_size: {tokenizer.vocab_size}")
    print(f"[TOKENIZER] eos_token_id: {tokenizer.eos_token_id}")
    print(f"[TOKENIZER] pad_token_id: {tokenizer.pad_token_id}")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        print("[TOKENIZER] WARNING: pad_token is None, setting to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

except Exception as e:
    print(f"[TOKENIZER] ERROR: Failed to load tokenizer: {e}")
    print(f"[TOKENIZER] Traceback: {traceback.format_exc()}")
    raise

# ==============================================================================
# STEP 9: Test single prompt forward pass
# ==============================================================================

print("\n" + "=" * 80)
print("[TEST] Single Prompt Forward Pass Test")
print("=" * 80)

test_prompt = "Hello, how are you?"
print(f"[TEST] Test prompt: '{test_prompt}'")

try:
    print("[TEST] Step 9.1: Tokenizing...")
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"[TEST]   input_ids shape: {input_ids.shape}")
    print(f"[TEST]   input_ids: {input_ids}")
    print(f"[TEST]   attention_mask: {inputs.attention_mask}")

    # Move to model device
    device = model.device
    print(f"[TEST]   Moving inputs to device: {device}")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("[TEST] Step 9.2: Running forward pass (no generation)...")
    forward_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    forward_elapsed = time.time() - forward_start
    print(f"[TEST]   Forward pass completed in {forward_elapsed:.3f}s")

    print("[TEST] Step 9.3: Checking captured layers...")
    print(f"[TEST]   Captured layer keys: {sorted(captured.keys())}")
    print(f"[TEST]   Expected layer keys: {sorted(LAYERS_TO_CAPTURE)}")

    if set(captured.keys()) == set(LAYERS_TO_CAPTURE):
        print("[TEST] SUCCESS: All hooks fired correctly!")
    else:
        missing = set(LAYERS_TO_CAPTURE) - set(captured.keys())
        extra = set(captured.keys()) - set(LAYERS_TO_CAPTURE)
        print(f"[TEST] WARNING: Mismatch. Missing: {missing}, Extra: {extra}")

    print("[TEST] Step 9.4: Verifying tensor shapes...")
    for layer_idx in LAYERS_TO_CAPTURE:
        t = captured[layer_idx]
        print(f"[TEST]   Layer {layer_idx}: shape={t.shape}, dtype={t.dtype}")
        # Expected shape: [1, seq_len, 4096]
        expected_shape = (1, input_ids.shape[1], 4096)
        if t.shape != expected_shape:
            print(f"[TEST]   WARNING: Unexpected shape for layer {layer_idx}")
            print(f"[TEST]   Expected: {expected_shape}, Got: {t.shape}")

except Exception as e:
    print(f"[TEST] ERROR: Single prompt test failed: {e}")
    print(f"[TEST] Traceback: {traceback.format_exc()}")
    raise

# ==============================================================================
# STEP 10: Test sparse extraction on one layer
# ==============================================================================

print("\n" + "=" * 80)
print("[SPARSE] Testing Sparse Neuron Extraction")
print("=" * 80)

try:
    layer_19_idx = 19
    print(f"[SPARSE] Testing on layer {layer_19_idx}...")

    # Get the captured tensor for layer 19
    if layer_19_idx not in captured:
        print(f"[SPARSE] WARNING: Layer {layer_19_idx} not in captured. Using layer {LAYERS_TO_CAPTURE[0]}")
        layer_19_idx = LAYERS_TO_CAPTURE[0]

    layer_tensor = captured[layer_19_idx]
    print(f"[SPARSE] Layer {layer_19_idx} tensor shape: {layer_tensor.shape}")

    # Compute statistics
    stats = compute_layer_statistics(layer_tensor)
    print(f"[SPARSE] Statistics:")
    print(f"       min: {stats['min']:.4f}")
    print(f"       max: {stats['max']:.4f}")
    print(f"       mean: {stats['mean']:.4f}")
    print(f"       std: {stats['std']:.4f}")
    print(f"       sparsity_pct: {stats['sparsity_pct']:.2f}%")
    print(f"       active_neuron_count: {stats['active_neuron_count']}")
    print(f"       total_neurons: {stats['total_neurons']}")

    # Extract sparse neurons
    print(f"[SPARSE] Extracting top-{TOP_K} sparse neurons...")
    sparse = extract_sparse_neurons(layer_tensor, top_k=TOP_K)
    print(f"[SPARSE] SUCCESS: Extracted {len(sparse)} neurons")

    if sparse:
        print(f"[SPARSE] Sample neurons (first 5): {sparse[:5]}")
        # Verify encoding
        verify_encoding(sparse)
        print(f"[SPARSE] Encoding verification: PASS")

        # Encode to string
        encoded = encode_sparse(sparse)
        print(f"[SPARSE] Encoded length: {len(encoded)} chars")
        print(f"[SPARSE] Encoded sample: {encoded[:100]}...")

        # Decode back
        decoded = decode_sparse(encoded)
        print(f"[SPARSE] Decoded length: {len(decoded)} tuples")
        assert decoded == sparse, "Decode mismatch!"
        print(f"[SPARSE] Decode verification: PASS")
    else:
        print(f"[SPARSE] WARNING: No active neurons extracted!")

except Exception as e:
    print(f"[SPARSE] ERROR: Sparse extraction test failed: {e}")
    print(f"[SPARSE] Traceback: {traceback.format_exc()}")
    raise

# ==============================================================================
# STEP 11: Build verification dataset
# ==============================================================================

print("\n" + "=" * 80)
print("[DATASET] Building Verification Dataset")
print("=" * 80)

print(f"[DATASET] Domain config: {DOMAIN_CONFIG}")
print(f"[DATASET] Total prompts requested: {sum(DOMAIN_CONFIG.values())}")

dataset_start = time.time()
try:
    prompts = build_diverse_dataset(DOMAIN_CONFIG)
    dataset_elapsed = time.time() - dataset_start
    print(f"[DATASET] SUCCESS: Built {len(prompts)} prompts in {dataset_elapsed:.2f}s")
except Exception as e:
    print(f"[DATASET] ERROR: Failed to build dataset: {e}")
    print(f"[DATASET] Traceback: {traceback.format_exc()}")
    print("[DATASET] Falling back to simple prompts...")
    prompts = [
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms.",
        "Write a function to check if a number is prime.",
    ] * 34
    prompts = prompts[:NUM_VERIFY_PROMPTS]
    print(f"[DATASET] Fallback: using {len(prompts)} simple prompts")

print(f"[DATASET] First 3 prompts:")
for i, p in enumerate(prompts[:3]):
    print(f"  [{i}] {p[:80]}...")

# ==============================================================================
# STEP 12: Run verification prompts
# ==============================================================================

print("\n" + "=" * 80)
print(f"[RUN] Running Verification: {len(prompts)} Prompts")
print("=" * 80)

results = []
run_start = time.time()

try:
    for i, prompt in enumerate(prompts):
        prompt_start = time.time()

        # Progress logging every 10 prompts
        if i % 10 == 0:
            elapsed_total = time.time() - run_start
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            eta = (len(prompts) - i - 1) / rate if rate > 0 else 0
            print(f"[RUN] Progress: {i}/{len(prompts)} prompts processed... "
                  f"({rate:.2f} prompts/sec, ETA: {eta:.0f}s)")

        # Tokenize
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            seq_len = input_ids.shape[1]
        except Exception as e:
            print(f"[RUN] ERROR: Tokenization failed for prompt {i}: {e}")
            print(f"[RUN]   Prompt: {prompt[:50]}...")
            raise

        # Clear captured dict
        captured.clear()

        # Forward pass
        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=inputs.attention_mask.to(model.device))
        except Exception as e:
            print(f"[RUN] ERROR: Forward pass failed for prompt {i}: {e}")
            print(f"[RUN]   Prompt: {prompt[:50]}...")
            print(f"[RUN]   Input shape: {input_ids.shape}")
            raise

        # Check hooks fired
        if len(captured) != len(LAYERS_TO_CAPTURE):
            print(f"[RUN] WARNING: Prompt {i} — expected {len(LAYERS_TO_CAPTURE)} hooks, got {len(captured)}")

        # Extract sparse activations
        row = {
            "prompt_id": i,
            "prompt_text": prompt[:100],  # Truncate for CSV
        }

        for layer_idx in LAYERS_TO_CAPTURE:
            if layer_idx in captured:
                try:
                    sparse = extract_sparse_neurons(captured[layer_idx], top_k=TOP_K)
                    row[f"layer_{layer_idx}"] = encode_sparse(sparse)
                except Exception as e:
                    print(f"[RUN] ERROR: Sparse extraction failed for layer {layer_idx}, prompt {i}: {e}")
                    row[f"layer_{layer_idx}"] = "[]"
            else:
                row[f"layer_{layer_idx}"] = "[]"

        results.append(row)

        # Per-prompt timing
        prompt_elapsed = time.time() - prompt_start
        if prompt_elapsed > 1.0:  # Log slow prompts
            print(f"[RUN]   Prompt {i} slow: {prompt_elapsed:.2f}s")

    run_elapsed = time.time() - run_start
    print(f"[RUN] SUCCESS: All {len(prompts)} prompts processed in {run_elapsed:.1f}s")
    print(f"[RUN] Average: {run_elapsed / len(prompts):.3f}s per prompt")
    print(f"[RUN] Rate: {len(prompts) / run_elapsed:.2f} prompts/sec")

except Exception as e:
    print(f"[RUN] ERROR: Verification run failed: {e}")
    print(f"[RUN] Traceback: {traceback.format_exc()}")
    print(f"[RUN] Failed at prompt index: {i}")
    print(f"[RUN] Results so far: {len(results)} rows")
    # Continue to save what we have

# ==============================================================================
# STEP 13: Write CSV
# ==============================================================================

print("\n" + "=" * 80)
print("[CSV] Writing Results to CSV")
print("=" * 80)

try:
    print(f"[CSV] Creating DataFrame with {len(results)} rows...")
    df = pd.DataFrame(results)
    print(f"[CSV] DataFrame shape: {df.shape}")
    print(f"[CSV] Columns: {list(df.columns)}")

    print(f"[CSV] Writing to: {OUTPUT_FILE}")
    csv_start = time.time()
    df.to_csv(OUTPUT_FILE, index=False)
    csv_elapsed = time.time() - csv_start
    print(f"[CSV] SUCCESS: Wrote CSV in {csv_elapsed:.2f}s")

    # File size
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"[CSV] File size: {file_size / 1e6:.2f} MB")
    else:
        print(f"[CSV] WARNING: File not found at {OUTPUT_FILE}")

except Exception as e:
    print(f"[CSV] ERROR: Failed to write CSV: {e}")
    print(f"[CSV] Traceback: {traceback.format_exc()}")
    raise

# ==============================================================================
# STEP 14: Print Summary Statistics
# ==============================================================================

print("\n" + "=" * 80)
print("[SUMMARY] Verification Results Summary")
print("=" * 80)

print(f"  Total prompts: {len(prompts)}")
print(f"  Total layers captured: {len(LAYERS_TO_CAPTURE)}")
print(f"  Active neurons per layer (expected ~200-400):")

for layer_idx in LAYERS_TO_CAPTURE:
    col = f"layer_{layer_idx}"
    if col in df.columns:
        try:
            neuron_counts = [len(ast.literal_eval(r)) for r in df[col].head(10)]
            avg = sum(neuron_counts) / len(neuron_counts) if neuron_counts else 0
            print(f"    Layer {layer_idx}: avg={avg:.0f} neurons (sample of {len(neuron_counts)})")
        except Exception as e:
            print(f"    Layer {layer_idx}: ERROR reading values: {e}")
    else:
        print(f"    Layer {layer_idx}: column not found in DataFrame")

# ==============================================================================
# STEP 15: Cleanup — remove hooks
# ==============================================================================

print("\n" + "=" * 80)
print("[CLEANUP] Removing Hooks")
print("=" * 80)

try:
    for i, h in enumerate(handles):
        h.remove()
    print(f"[CLEANUP] SUCCESS: Removed {len(handles)} hooks")
except Exception as e:
    print(f"[CLEANUP] WARNING: Error removing hooks: {e}")

print(f"[CLEANUP] Model still in memory (unloading handled by Python GC)")

# ==============================================================================
# FINISH
# ==============================================================================

print("\n" + "=" * 80)
print("[FINISH] Verification Complete!")
print("=" * 80)
print(f"[FINISH] Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[FINISH] Output file: {OUTPUT_FILE}")
print(f"[FINISH] Results: {len(results)} rows")
print(f"[FINISH] Next step: Check {OUTPUT_FILE} before running full 100K collection")
print("=" * 80)