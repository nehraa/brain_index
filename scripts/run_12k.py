"""
Brain Index - 12K Prompt Collection with Optimized Parameters
===========================================================
Using TOP_K=10 and optimized layers [4, 9, 14, 19, 24, 29, 31, 32]

Total: 12,000 prompts
Estimated time: ~12 hours
"""

print("[LOAD] Loading Qwen3-8B (bf16, no quantization)...")

import time
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import os

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='sdpa')
model.eval()
print(f"[LOAD] Model loaded in {time.time()-t0:.1f}s")

# Optimized layers: [4, 9, 14, 19, 24, 29, 31, 32]
# Dropped 33, 34, 35 (inactive in parameter exploration)
ALL_LAYERS = [4, 9, 14, 19, 24, 29, 31, 32]

captured = {}
handles = []
def make_hook(i):
    def h(module, input, output):
        captured[i] = output.detach().cpu()
    return h
for i in ALL_LAYERS:
    handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))
print(f"[LOAD] Hooks registered on {len(handles)} layers: {ALL_LAYERS}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[LOAD] Tokenizer loaded")

# Core neurons to exclude from analysis (fire in ~100% of prompts)
CORE_NEURONS = {1838, 2276, 994, 3828, 1214, 3010}
print(f"[LOAD] Core neurons to exclude: {CORE_NEURONS}")

# Extraction functions
def extract_sparse_neurons(hidden_tensor, top_k=10):
    lt = hidden_tensor[0, -1, :] if hidden_tensor.dim() == 3 else (hidden_tensor[-1, :] if hidden_tensor.dim() == 2 else hidden_tensor)
    activated = F.relu(lt)
    values, indices = activated.topk(k=min(top_k, activated.numel()))
    max_val = values.max().item()
    normalized = ((values / max_val) * 255).round().int() if max_val > 0 else values.new_zeros(len(values), dtype=torch.int)
    return [(idx.item(), val.item()) for idx, val in zip(indices, normalized)]

def encode_sparse(n): return str(n)

# Dataset builder
from datasets import load_dataset
DOMAIN_SOURCES = {
    'reasoning': ('open-thoughts/AgentTrove', None, 'conversations'),
    'creative': ('stingning/ultrachat', None, 'data'),
    'code': ('stingning/ultrachat', None, 'data'),
    'general': ('stingning/ultrachat', None, 'data'),
    'summarization': ('stingning/ultrachat', None, 'data'),
    'translation': ('stingning/ultrachat', None, 'data'),
    'agentic': ('open-thoughts/AgentTrove', None, 'conversations'),
}
FILTER_KW = {
    'reasoning': ['solve','calculate','explain','prove','find','determine','reasoning','logic','math'],
    'creative': ['write','story','poem','script','creative','imagine','invent'],
    'code': ['code','function','python','javascript','programming','implement','algorithm','debug'],
    'translation': ['translate','translation','from english','to spanish','to french'],
    'agentic': ['plan','steps','tool','agent','search','find','look up','research'],
}

def extract_prompt(item, col):
    if col == 'conversations' and col in item:
        for conv in reversed(item[col]):
            if isinstance(conv, dict) and conv.get('role') in ('user','human'):
                v = conv.get('value','')
                if v and isinstance(v, str) and len(v) > 10: return v.strip()
    if col == 'data' and col in item:
        for msg in item[col]:
            if isinstance(msg, str) and len(msg) > 10: return msg.strip()
    for c in [col, 'text', 'instruction', 'prompt', 'input', 'message', 'content']:
        if c and c in item:
            v = item[c]
            if v and isinstance(v, str) and len(v) > 10: return v.strip()
            if isinstance(v, list) and len(v) > 0:
                first = v[0]
                if isinstance(first, dict):
                    val = first.get('content', first.get('text', ''))
                    if val and isinstance(val, str) and len(val) > 10: return val.strip()
    return None

def build_dataset(domain_counts):
    all_prompts = []
    for domain, count in domain_counts.items():
        ds_name, cfg, text_col = DOMAIN_SOURCES[domain]
        try:
            ds = load_dataset(ds_name, cfg, split='train', streaming=True) if cfg else load_dataset(ds_name, split='train', streaming=True)
        except:
            ds = load_dataset('stingning/ultrachat', split='train', streaming=True)
            text_col = 'data'
        raw = []
        for j, item in enumerate(iter(ds)):
            pt = extract_prompt(item, text_col)
            if pt: raw.append(pt)
            if j >= 9999: break
        if domain in FILTER_KW:
            raw = [p for p in raw if any(kw in p.lower() for kw in FILTER_KW[domain])]
        if not raw:
            raw = [f'Sample {domain} prompt {i}' for i in range(count)]
        if len(raw) >= count:
            sampled = raw[::max(1, len(raw)//count)][:count]
        else:
            shortage = count - len(raw)
            extra = (raw * ((shortage//len(raw))+1))[:shortage]
            sampled = raw + extra
        all_prompts.extend(sampled)
    import random
    random.shuffle(all_prompts)
    return all_prompts

print(f"[LOAD] Setup complete in {time.time()-t0:.1f}s")

# ============================================================
# 12K PROMPT COLLECTION
# ============================================================

print("\n" + "="*60)
print("12K PROMPT COLLECTION - OPTIMIZED PARAMETERS")
print("="*60)
print(f"Layers: {ALL_LAYERS}")
print(f"TOP_K: 10")
print(f"Core neurons excluded: {CORE_NEURONS}")

# Build 12K prompts (scaled up from 300 prompt config)
domain_config = {
    "reasoning": 1720,   # ~14%
    "creative": 1200,    # ~10%
    "code": 1720,        # ~14%
    "general": 2840,     # ~24%
    "summarization": 1200, # ~10%
    "translation": 1200,  # ~10%
    "agentic": 2120      # ~18%
}
# Total: 12000

all_prompts = build_dataset(domain_config)
print(f"\n[RUN] Built {len(all_prompts)} prompts")

# Output CSV
csv_path = "brain_index/data/brain_index_12k.csv"
os.makedirs("brain_index/data", exist_ok=True)

# Header
header = "prompt_id,prompt_text," + ",".join([f"layer_{li}" for li in ALL_LAYERS])
header += "," + ",".join([f"layer_{li}_excluded_core" for li in ALL_LAYERS])
header += "\n"

with open(csv_path, "w") as f:
    f.write(header)

import gc

# Run collection
total_done = 0
run_start = time.time()
checkpoint_interval = 100  # Every 100 prompts

for i, prompt in enumerate(all_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    captured.clear()

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

    row = {
        "prompt_id": i,
        "prompt_text": prompt[:200].replace(",", ";"),
    }

    excluded_counts = {}
    for li in ALL_LAYERS:
        sparse = extract_sparse_neurons(captured[li], top_k=10)
        # Count how many core neurons were in the top 10
        core_included = len([n for n in sparse if n[0] in CORE_NEURONS])
        excluded_counts[li] = core_included
        row[f"layer_{li}"] = encode_sparse(sparse)
        row[f"layer_{li}_excluded_core"] = core_included

    # Build CSV line
    line = f"{row['prompt_id']},{row['prompt_text']}"
    for li in ALL_LAYERS:
        line += f",{row[f'layer_{li}']}"
    for li in ALL_LAYERS:
        line += f",{row[f'layer_{li}_excluded_core']}"
    line += "\n"

    with open(csv_path, "a") as f:
        f.write(line)

    # EXPLICIT MEMORY CLEANUP - prevent GPU OOM accumulation
    del inputs, outputs
    for li in ALL_LAYERS:
        if li in captured:
            del captured[li]
    captured.clear()
    torch.cuda.empty_cache()
    gc.collect()

    total_done += 1

    # Checkpoint every 100 prompts
    if total_done % checkpoint_interval == 0:
        elapsed = time.time() - run_start
        rate = total_done / elapsed
        remaining = (len(all_prompts) - total_done) / rate if rate > 0 else 0
        size_mb = os.path.getsize(csv_path) / 1e6
        print(f"[CHECKPOINT] {total_done}/{len(all_prompts)} ({rate:.2f}/sec, ~{remaining/60:.1f}min remaining, {size_mb:.1f}MB)")

total_elapsed = time.time() - run_start
print(f"\n{'='*60}")
print(f"COLLECTION COMPLETE!")
print(f"{'='*60}")
print(f"Total prompts: {total_done}")
print(f"Time: {total_elapsed/60:.1f} minutes")
print(f"Output: {csv_path} ({os.path.getsize(csv_path)/1e6:.1f}MB)")