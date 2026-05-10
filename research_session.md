# Brain Index — Session Research Document

## Concept Recap

A small trained model ('index') learns to predict which neurons activate in a larger 'main' model for a given prompt. Instead of retrieving stored activations (RAG-style), the index learns the function that maps prompts → activation patterns.

```
Prompt → Small Index Model (1-20M params, trained)
    → predicts which neurons should fire in main model
    → injects activation priors/bias into main model
    → main model generates with pre-warmed state
```

### Core Feasibility: Sparse Activations (<1%)

30B model has ~28,672 neurons per layer. With ReLU, only ~1% fire per forward pass = ~280-350 active neurons per layer. This is the core insight — we only store the sparse active set, not the full dense vector.

Storage per active neuron: neuron index (2 bytes) + activation value (1 byte) = 3 bytes. Per layer: ~1KB. Per prompt: varies by layer count.

---

## Target Setup

### Model: Qwen3-8B (from HuggingFace)

```
HuggingFace: Qwen/Qwen3-8B
- 36 layers
- hidden_size: 4096
- intermediate_size: 12288
- num_attention_heads: 32
- num_key_value_heads: 8
- vocab_size: 151936
- head_dim: 128
- ~8B parameters
- Available on HuggingFace ✅
- Pure transformer architecture ✅
- Hooks will work ✅
```

**⚠️ Critical Finding: Qwen3-9B Does NOT Exist on HuggingFace**

The model `Qwen/Qwen3-9B` does not exist on HuggingFace — that URL returns 404. The only Qwen3 models on HF are:
- `Qwen/Qwen3-8B` (36 layers) — **use this**
- `Qwen/Qwen3-32B` (64 layers, ~32B params, needs 2 GPUs at Q4)

The Ollama `qwen3.5:9b` is a **Mamba SSM + transformer hybrid** architecture — not pure Qwen3. Cannot use HuggingFace transformers directly.

### Hardware: Kaggle 2× T4

- Each T4: 16GB GDDR6 VRAM, 2,560 CUDA cores
- Combined: 30GB VRAM, 5,120 CUDA cores
- **Plan:** Use 1× T4 with bitsandbytes Q4 (~3-4GB VRAM footprint)
- Second T4: free for parallel work (index training, preprocessing)

### Storage Budget

- tmp: 120GB — model downloaded here once, reused every run
- Persistent: 20GB — activations CSV + prompt dataset
- **Total:** Well under budget (~725MB for 100K activations)

---

## Corrected Qwen3 Architecture (Qwen3-8B)

| Property | Value |
|----------|-------|
| **num_hidden_layers** | **36** |
| **hidden_size** | **4096** |
| **intermediate_size** | 12288 |
| **num_attention_heads** | 32 |
| **num_key_value_heads** | 8 |
| **vocab_size** | 151936 |
| **head_dim** | 128 |
| **Total params** | ~8B |

---

## Layer Selection (Qwen3-8B = 36 layers)

With 36 layers, sampling every 5th + last = **8 layers**:
```
Layers captured:  4,  9, 14, 19, 24, 29, 34, 36 (last)
```

| Layer | What it captures |
|-------|-----------------|
| 4 | Token processing, basic syntax, surface patterns |
| 9 | Early semantic understanding |
| 14 | Core reasoning, concept binding |
| 19 | Complex reasoning, multi-step logic |
| 24 | Deep semantic processing |
| 29 | Task specialization, domain detection |
| 34 | Output preparation, style control |
| 36 | Full accumulated state — most predictive |

### Storage Per Prompt

- Per layer: ~300 active neurons × 3 bytes = ~900 bytes
- Per prompt (8 layers): ~7KB
- 100K prompts: ~700MB
- 300K prompts: ~2.1GB

---

## Data Collection Plan

### What We Collect

Per prompt:
1. Tokenize prompt → embed in model
2. Run single forward pass (NO autoregressive generation)
3. Capture intermediate layer hidden states via PyTorch hooks
4. Apply ReLU → extract top-k active neurons → normalize to 0-255
5. Store: `(prompt_id, prompt_text, layer_4_active_neurons, ...)`

### Dataset: 100K Diverse Prompts (7 Domains)

| Domain | Prompts | Source |
|--------|--------|--------|
| Reasoning (math, logic, multi-step) | 20K | open-thoughts/AgentTrove |
| Creative writing (stories, scripts) | 15K | cambridgeltl/OpenAssistant |
| Code generation & explanation | 15K | bigcode/the-stack-matquiz |
| General Q&A & instruction following | 20K | yahma/alpaca + fineweb-edu |
| Summarization & extraction | 10K | samsum |
| Translation | 10K | bigscience/xmt, wmt16 |
| Agentic (planning, tool use) | 10K | open-thoughts/AgentTrove filtered |

---

## Implementation Approach (No llama.cpp Modification)

### Libraries
```python
transformers     # AutoModelForCausalLM + hooks
torch            # PyTorch hooks
datasets        # Prompt dataset loading
bitsandbytes    # Q4 quantization inline
```

### Activation Capture via PyTorch Hooks (Verified Paths)

**Key insight:** No modification to llama.cpp needed. Pure HuggingFace + PyTorch hooks.

**Verified from Qwen3 source code:**
- Layer forward returns `torch.Tensor` directly (not `BaseModelOutput`) ✅
- Module path: `model.model.model.layers[i]` → `Qwen3DecoderLayer` ✅
- Attention submodule: `self.self_attn`
- MLP submodule: `self.mlp`

### Hook approach: Register on Qwen3DecoderLayer forward()

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="sdpa"
)
model.eval()

# Capture dict: layer_idx -> tensor [batch, seq_len, hidden_size=4096]
captured_hidden_states = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output is torch.Tensor directly (not BaseModelOutput)
        # Shape: [batch, seq_len, hidden_size=4096]
        captured_hidden_states[layer_idx] = output.detach().cpu()
    return hook

# Module path (verified from source):
# model = Qwen3ForCausalLM
# model.model = Qwen3Model
# model.model.model.layers = nn.ModuleList([Qwen3DecoderLayer x 36])
handles = []
for layer_idx, layer in enumerate(model.model.model.layers):
    handles.append(layer.register_forward_hook(make_hook(layer_idx)))

# Single forward pass — no generation
inputs = tokenizer("Your prompt here", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# captured_hidden_states: dict[layer_idx -> torch.Tensor]
# Shape per layer: [batch=1, seq_len, hidden_size=4096]
# Then: apply ReLU -> extract top-k -> normalize

for h in handles:
    h.remove()
```

---

## Critical: Module Path for Hooks ✅ RESOLVED

From Qwen3 source code — verified paths:

| Component | Path |
|-----------|------|
| Full model (Qwen3ForCausalLM) | `model` |
| Inner model (Qwen3Model) | `model.model` |
| Transformer layers container | `model.model.model.layers` |
| Individual layer (Qwen3DecoderLayer) | `model.model.model.layers[0...35]` |
| Attention submodule | `model.model.model.layers[i].self_attn` |
| MLP submodule | `model.model.model.layers[i].mlp` |
| Pre-attention RMSNorm | `model.model.model.layers[i].input_layernorm` |
| Post-attention RMSNorm | `model.model.model.layers[i].post_attention_layernorm` |

**Note:** Three-level `.model.model.layers` — `Qwen3ForCausalLM` wraps `Qwen3Model` which holds the layers.

### Qwen3DecoderLayer.forward() Returns:
- **`torch.Tensor`** — raw hidden_states directly (not a named tuple)
- Shape: `[batch, seq_len, hidden_size=4096]`
- Clean for hooks — `output` IS the tensor, no `.last_hidden_state` needed

---

## Speed Estimates

| Operation | Time |
|-----------|------|
| Single forward pass (1 token, no generation) | ~20-50ms |
| 100K prompts | ~33-83 minutes |
| Full data collection run | ~1-2 hours |
| Kaggle session usage | ~5-10% of weekly 30hr budget |

---

## CSV Output Format

```csv
prompt_id, prompt_text, layer_4, layer_9, layer_14, layer_19, layer_24, layer_29, layer_34, layer_36
0, "How do I bake a cake?", "[(1234,189),(4567,234),...]", "[(89,156),...]", ...
```

Each layer column stores compressed active neuron data: `[(neuron_idx, normalized_value), ...]`

---

## Verification Checklist (Before Full 100K Run)

Run on Kaggle with 100 prompts first — costs ~$0.50-1:

- [ ] Layer hook fires for all selected layers (4, 9, 14, 19, 24, 29, 34, 36)
- [ ] Hidden state tensor shapes are correct: `[batch, seq_len, hidden_size=4096]`
- [ ] After ReLU + top-k extraction: ~200-400 active neurons per layer (not 0, not all)
- [ ] Activation std dev > 0 across different prompts
- [ ] No memory leaks from hooks not removed
- [ ] CSV writes correctly with non-zero data

---

## Storage Breakdown (20GB Persistent)

| Item | Size |
|------|------|
| Prompt dataset (100K × ~250 bytes) | ~25MB |
| Activation CSV (100K × 7KB) | ~700MB |
| Layer-wise activation cache (temp) | ~700MB |
| Index model weights (20-50M params, Q4) | ~100-200MB |
| **Total** | **~1.5GB** |

Room for 2M prompts comfortably within 20GB.

---

## What's Changed from Original Plan

| Item | Was | Now |
|------|-----|-----|
| Model | Qwen3.5-9B (Ollama) | **Qwen3-8B** (HuggingFace) |
| Layers | 40 | **36** |
| Hidden size | 3584 | **4096** |
| Module path | `model.model.layers` | **`model.model.model.layers`** |
| Layer forward return | Assumed BaseModelOutput | **`torch.Tensor` directly** |
| Hook target | Generic assumption | **`Qwen3DecoderLayer.register_forward_hook`** |

---

## Next Steps

1. [x] Get Qwen3 source code info ✅ DONE
2. [x] Update hook code with correct module paths for Qwen3 ✅ DONE
3. [ ] Write verification script (100 prompts on Kaggle ~$1)
4. [ ] Run verification — confirm layer shapes + activation stats
5. [ ] If verification passes → full 100K data collection run
6. [ ] Train index model on collected activations
7. [ ] Validate index prediction accuracy vs random baseline
