# Brain Index — Activation Collection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kaggle notebook that runs 100K diverse prompts through Qwen3-8B, captures sparse neuron activations at 8 intermediate layers, and writes results to CSV.

**Architecture:** Single Kaggle notebook (CPU + GPU). Downloads model to Kaggle's `/tmp` (120GB), loads to T4 VRAM with bitsandbytes Q4 quantization. Registers PyTorch forward hooks on Qwen3DecoderLayer modules. For each prompt: single forward pass → ReLU → top-k active neurons → normalized 0-255 → CSV row.

**Tech Stack:**
- Model: `Qwen/Qwen3-8B` via `transformers` + `bitsandbytes` (Q4 quantization)
- Hooks: PyTorch `register_forward_hook`
- Dataset: `datasets` library loading from HuggingFace
- Output: CSV via `pandas`
- Logging: excessive `print()` statements at every step

---

## Context

From `research_session.md`:
- Model: Qwen3-8B (36 layers, hidden_size=4096, ~8B params)
- **Module path:** `model.model.model.layers[i]` → `Qwen3DecoderLayer`
- **Layer forward returns:** `torch.Tensor` directly (not BaseModelOutput)
- Hooks register per-layer on `Qwen3DecoderLayer.forward()`
- Layers captured: [4, 9, 14, 19, 24, 29, 34, 36] — 8 layers
- Storage: ~700MB for 100K prompts (8 layers × ~7KB/prompt)
- Speed: ~33-83 min for 100K prompts (single forward pass, no generation)
- Dataset: 100K diverse prompts across 7 domains

From Obsidian vault skills/tools:
- **Available MCP:** `context7` (for HuggingFace/transformers docs), `plugin:context7:context7`
- **Available skills:** `writing-plans`, `systematic-debugging`, `executing-plans`, `subagent-driven-development`, `verification-before-completion`, `code-review`, `frontend-design`
- **Tools:** `Bash`, `Write`, `Edit`, `Read`, `Glob`, `Grep` (standard), `mcp__plugin_context7_context7__query-docs` (HuggingFace docs)

---

## Execution Options

**"Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?"**"

---

## Phase 1: Write Plan → `brain_index_implementation_plan.md`

Write this plan to `docs/superpowers/plans/YYYY-MM-DD-brain-index-activation-collection.md` and commit.

---

## Phase 2: Implementation Tasks

### Task 1: Create Directory Structure
**Files:**
- Create: `brain_index/scripts/`
- Create: `brain_index/scripts/utils/`
- Create: `brain_index/data/`
- Create: `brain_index/docs/superpowers/plans/`

### Task 2: Write `brain_index/scripts/utils/dataset_builder.py`
**Files:** `brain_index/scripts/utils/dataset_builder.py`

Builds 100K diverse prompts from HuggingFace datasets. 7 domains.

```python
# DOMAIN → SOURCE mapping (exact):
reasoning: "open-thoughts/AgentTrove"
creative: "cambridgeltl/OpenAssistant/oa_top_level_2023"
code: "bigcode/the-stack-matquiz"
general: "yahma/alpaca"
summarization: "samsum"
# translation: use cambridgeltl/OpenAssistant filtered to language
# agentic: open-thoughts/AgentTrove filtered to tool-use / planning
```

**Every single line logs:**
```python
print(f"[DATASET] Step 1: Loading dataset {name} from HF...")
print(f"[DATASET] SUCCESS: Dataset {name} loaded, rows={len(ds)}")
# On error:
print(f"[DATASET] ERROR loading {name}: {e}")
raise
```

### Task 3: Write `brain_index/scripts/utils/activation_extractor.py`
**Files:** `brain_index/scripts/utils/activation_extractor.py`

```python
def extract_sparse_neurons(hidden_tensor: torch.Tensor, top_k: int = 300) -> list[tuple[int, int]]:
    # 1. Apply ReLU: F.relu(tensor) → shape [4096]
    # 2. Get last token: tensor[-1, :] → shape [4096]
    # 3. Flatten → top-k values + indices
    # 4. Normalize: val/max_val * 255 → round to uint8
    # Returns: [(neuron_idx, normalized_byte_val), ...]
```

**Every tensor operation logs shape at each stage:**
```python
print(f"[HOOK] hidden_tensor shape before ReLU: {tensor.shape}")
tensor_relu = F.relu(tensor)
print(f"[HOOK] hidden_tensor shape after ReLU: {tensor_relu.shape}")
# ...etc
```

### Task 4: Write `brain_index/scripts/utils/sparse_storage.py`
**Files:** `brain_index/scripts/utils/sparse_storage.py`

```python
def encode_sparse(neurons: list[tuple[int, int]]) -> str:
    # Encode: [(1234, 189), (4567, 234)] → "[(1234,189),(4567,234)]"
    return str(neurons)

def decode_sparse(encoded: str) -> list[tuple[int, int]]:
    # Decode string back to list
    import ast
    return ast.literal_eval(encoded)
```

### Task 5: Write `brain_index/scripts/verify_activation_capture.py`
**Files:** `brain_index/scripts/verify_activation_capture.py`

**THIS IS THE MAIN KAGGLE NOTEBOOK SCRIPT — written first as verification.**

Purpose: 100-prompt run on Kaggle to validate the entire pipeline before 100K run.

```python
# ===== CONFIGURATION =====
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS_TO_CAPTURE = [4, 9, 14, 19, 24, 29, 34, 36]
TOP_K = 300
NUM_VERIFY_PROMPTS = 100

# ===== STEP 1: Print config =====
print(f"[CONFIG] Model: {MODEL_NAME}")
print(f"[CONFIG] Layers to capture: {LAYERS_TO_CAPTURE}")
print(f"[CONFIG] Top-K active neurons per layer: {TOP_K}")
print(f"[CONFIG] Verification prompts: {NUM_VERIFY_PROMPTS}")

# ===== STEP 2: Check GPU =====
import torch
print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] GPU count: {torch.cuda.device_count()}")
    print(f"[GPU] Current GPU: {torch.cuda.current_device()}")
    print(f"[GPU] GPU name: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===== STEP 3: Import and version check =====
print("[LIBS] Importing transformers...")
import transformers
print(f"[LIBS] transformers version: {transformers.__version__}")
print("[LIBS] SUCCESS: transformers imported")

# ===== STEP 4: Load model =====
print("[MODEL] Loading Qwen3-8B with bitsandbytes Q4 quantization...")
print("[MODEL] This may take 2-5 minutes on first run (download + load)...")

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
print(f"[MODEL] SUCCESS: Model loaded to device_map=auto")
print(f"[MODEL] Model type: {type(model)}")

# ===== STEP 5: Verify module paths =====
print("[MODULE] Verifying module paths...")
print(f"[MODULE] model type: {type(model).__name__}")  # Should be Qwen3ForCausalLM
print(f"[MODULE] model.model type: {type(model.model).__name__}")  # Should be Qwen3Model
print(f"[MODULE] model.model.model type: {type(model.model.model).__name__}")  # Should be Qwen3Model (same)
print(f"[MODULE] model.model.model.layers length: {len(model.model.model.layers)}")  # Should be 36

# ===== STEP 6: Register hooks =====
print("[HOOKS] Registering forward hooks...")
captured = {}
handles = []

def make_hook(layer_idx):
    def hook(module, input, output):
        # output is torch.Tensor directly — shape: [batch, seq_len, hidden_size=4096]
        captured[layer_idx] = output.detach().cpu()
        print(f"[HOOK] Layer {layer_idx} fired! output.shape={output.shape}")
    return hook

for layer_idx in LAYERS_TO_CAPTURE:
    layer = model.model.model.layers[layer_idx]
    h = layer.register_forward_hook(make_hook(layer_idx))
    handles.append(h)
    print(f"[HOOK] Registered hook on layer {layer_idx} → module type: {type(layer).__name__}")

print(f"[HOOKS] SUCCESS: {len(handles)} hooks registered")

# ===== STEP 7: Load tokenizer =====
print("[TOKENIZER] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"[TOKENIZER] SUCCESS: tokenizer loaded, vocab_size={tokenizer.vocab_size}")

# ===== STEP 8: Test with single prompt =====
print("\n[TEST] Running test forward pass with single prompt...")
test_prompt = "Hello, how are you?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
print(f"[TEST] Input shape: {inputs.input_ids.shape}")
print(f"[TEST] Input tokens: {inputs.input_ids}")

with torch.no_grad():
    outputs = model(**inputs)

print(f"[TEST] Captured layers: {list(captured.keys())}")
print(f"[TEST] Expected layers: {LAYERS_TO_CAPTURE}")
if set(captured.keys()) == set(LAYERS_TO_CAPTURE):
    print("[TEST] SUCCESS: All hooks fired correctly!")
else:
    print(f"[TEST] FAIL: Mismatch. Missing: {set(LAYERS_TO_CAPTURE) - set(captured.keys())}")

# Print tensor shapes
for layer_idx in LAYERS_TO_CAPTURE:
    t = captured[layer_idx]
    print(f"[TEST] Layer {layer_idx} tensor shape: {t.shape}")

# ===== STEP 9: Test sparse extraction on one layer =====
from activation_extractor import extract_sparse_neurons
print("\n[SPARSE] Testing sparse extraction on layer 19...")
layer_19_tensor = captured[19][0, -1, :]  # batch=0, last token, all hidden
print(f"[SPARSE] Last token tensor shape: {layer_19_tensor.shape}")
print(f"[SPARSE] Tensor min: {layer_19_tensor.min():.4f}, max: {layer_19_tensor.max():.4f}")
print(f"[SPARSE] Tensor mean: {layer_19_tensor.mean():.4f}, std: {layer_19_tensor.std():.4f}")

sparse = extract_sparse_neurons(captured[19][0], top_k=TOP_K)
print(f"[SPARSE] Extracted {len(sparse)} active neurons (expected ~200-400)")
print(f"[SPARSE] Sample neurons: {sparse[:5]}")

# ===== STEP 10: Load dataset =====
from dataset_builder import build_diverse_dataset
print("\n[DATASET] Building 100-prompt verification dataset...")
domain_config = {
    "reasoning": 15,
    "creative": 10,
    "code": 15,
    "general": 25,
    "summarization": 10,
    "translation": 10,
    "agentic": 15
}  # Total = 100

prompts = build_diverse_dataset(domain_config)
print(f"[DATASET] SUCCESS: Built {len(prompts)} prompts")

# ===== STEP 11: Run 100 prompts =====
print(f"\n[RUN] Starting verification run: {len(prompts)} prompts...")
results = []
for i, prompt in enumerate(prompts):
    if i % 10 == 0:
        print(f"[RUN] Progress: {i}/{len(prompts)} prompts processed...")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract sparse activations
    row = {
        "prompt_id": i,
        "prompt_text": prompt[:100],  # Truncate for CSV
    }
    for layer_idx in LAYERS_TO_CAPTURE:
        sparse = extract_sparse_neurons(captured[layer_idx][0], top_k=TOP_K)
        row[f"layer_{layer_idx}"] = encode_sparse(sparse)

    results.append(row)

    # Clear captured for next iteration
    captured.clear()

print(f"[RUN] SUCCESS: All {len(prompts)} prompts processed!")

# ===== STEP 12: Write CSV =====
import pandas as pd
print("\n[CSV] Writing verification_results.csv...")
df = pd.DataFrame(results)
df.to_csv("brain_index/data/verification_results.csv", index=False)
print(f"[CSV] SUCCESS: Wrote {len(df)} rows to verification_results.csv")
print(f"[CSV] File size: {os.path.getsize('brain_index/data/verification_results.csv') / 1e6:.2f} MB")

# ===== STEP 13: Print summary stats =====
print("\n[SUMMARY] Verification Results:")
print(f"  Total prompts: {len(prompts)}")
print(f"  Total layers captured: {len(LAYERS_TO_CAPTURE)}")
print(f"  Active neurons per layer (expected ~200-400):")

for layer_idx in LAYERS_TO_CAPTURE:
    col = f"layer_{layer_idx}"
    neuron_counts = [len(ast.literal_eval(r[col])) for r in results[:10]]
    print(f"    Layer {layer_idx}: avg={sum(neuron_counts)/len(neuron_counts):.0f} neurons (sample of 10)")

print("\n[VERIFICATION] Pipeline validation complete!")
print("[VERIFICATION] Check verification_results.csv before running full 100K collection.")

# ===== Cleanup =====
for h in handles:
    h.remove()
print("[CLEANUP] Hooks removed, model unloaded.")
```

### Task 6: Write `brain_index/scripts/collect_activations.py`
**Files:** `brain_index/scripts/collect_activations.py`

Full 100K collection run. Same structure as verify but:
- 100K prompts (not 100)
- Progress bar (`tqdm`)
- Batch-level CSV writes every 5000 prompts
- Resumable via checkpoint file
- Silent except every 1000 prompts + final summary

### Task 7: Write `brain_index/scripts/utils/__init__.py`
**Files:** `brain_index/scripts/utils/__init__.py`

Exports for the utils package.

---

## Output Files

```
brain_index/
├── research_session.md              # From earlier research
├── README.md                         # Project overview
├── scripts/
│   ├── verify_activation_capture.py # Main Kaggle notebook (verification)
│   ├── collect_activations.py       # Main Kaggle notebook (full 100K)
│   └── utils/
│       ├── __init__.py
│       ├── dataset_builder.py
│       ├── activation_extractor.py
│       └── sparse_storage.py
├── data/
│   ├── brain_index_activations.csv   # Output (100K × 8 layers)
│   └── verification_results.csv       # Verification output
└── docs/superpowers/plans/
    └── 2026-05-09-brain-index-activation-collection.md
```

---

## CSV Output Format

```csv
prompt_id,prompt_text,layer_4,layer_9,layer_14,layer_19,layer_24,layer_29,layer_34,layer_36
0,How do I bake a cake?,[(1234,189),(4567,234),...],[(89,156),...],...
1,Explain quantum entanglement,[...],[...],...
```

`[(neuron_idx, normalized_byte_val), ...]` — each neuron: 2 bytes index + 1 byte value = 3 bytes.

---

## Kaggle Notebook URL

**User will provide:** Kaggle notebook URL when ready to run.

When provided, I'll push the scripts to the notebook via Kaggle API or direct file write.

---

## Verification

1. Run `verify_activation_capture.py` on Kaggle — 100 prompts (~2-5 min, ~$0.50-1)
2. Check printed output:
   - [ ] All 8 hooks fired (one print per hook)
   - [ ] Tensor shapes: `[1, seq_len, 4096]`
   - [ ] Active neurons: 100-500 per layer (not 0, not 4096)
   - [ ] CSV written: 100 rows, ~700KB
3. If all pass → run `collect_activations.py` for full 100K

---

## Success Criteria

1. `verify_activation_capture.py` runs on Kaggle T4 without errors
2. All 8 layer hooks fire (print confirmation for each)
3. Sparse extraction produces 100-500 active neurons per layer
4. `verification_results.csv` written with non-zero data
5. Full `collect_activations.py` runs 100K prompts in ~1-2 hours
6. `brain_index_activations.csv` written with 100K rows