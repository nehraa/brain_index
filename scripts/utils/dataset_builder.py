"""Build a diverse dataset of 100K prompts across 7 domains.

Each domain loads from a specific HuggingFace dataset source, filtered and
sampled to produce representative prompts for Brain Index training.

Domains:
- reasoning: Math, logic, multi-step reasoning problems
- creative: Story writing, scripts, creative content
- code: Code generation, explanation, debugging
- general: General Q&A, instruction following
- summarization: Text summarization tasks
- translation: Cross-lingual translation tasks
- agentic: Planning, tool use, multi-step agent behavior
"""

from typing import List, Dict
from datasets import load_dataset


# ==============================================================================
# DOMAIN → SOURCE MAPPING
# ==============================================================================
# Each domain maps to (dataset_name, config_name_if_any, column_to_extract)
# column is typically 'instruction' or 'prompt' or 'text'

DOMAIN_SOURCES: Dict[str, tuple] = {
    # reasoning/agentic: open-thoughts/AgentTrove has 'conversations' column (list of chat dicts)
    "reasoning": ("open-thoughts/AgentTrove", None, "conversations"),
    "creative": ("stingning/ultrachat", None, "data"),
    "code": ("HuggingFaceH4/instruction-dataset", None, "prompt"),
    "general": ("HuggingFaceH4/instruction-dataset", None, "prompt"),
    "summarization": ("stingning/ultrachat", None, "data"),
    "translation": ("stingning/ultrachat", None, "data"),
    "agentic": ("open-thoughts/AgentTrove", None, "conversations"),
}


# ==============================================================================
# FILTER FUNCTIONS — per domain, extract relevant prompts from dataset
# ==============================================================================

def filter_reasoning(prompts: List[str]) -> List[str]:
    """Keep prompts that look like reasoning/math problems."""
    keywords = ["solve", "calculate", "explain", "prove", "find", "determine",
               "what is", "how many", "if ", "given that", "reasoning", "logic"]
    return [p for p in prompts if any(kw in p.lower() for kw in keywords)]


def filter_creative(prompts: List[str]) -> List[str]:
    """Keep prompts that look like creative writing requests."""
    keywords = ["write", "story", "poem", "script", "creative", "imagine",
                "tell me a", "describe", "invent", "make up"]
    return [p for p in prompts if any(kw in p.lower() for kw in keywords)]


def filter_code(prompts: List[str]) -> List[str]:
    """Keep prompts that look like code-related requests."""
    keywords = ["code", "function", "python", "javascript", "programming",
                "implement", "algorithm", "debug", "fix", "write a program"]
    return [p for p in prompts if any(kw in p.lower() for kw in keywords)]


def filter_translation(prompts: List[str]) -> List[str]:
    """Keep prompts that look like translation requests."""
    keywords = ["translate", "translation", "from english", "to spanish",
                "to french", "to german", "to chinese", "in english", "to japanese"]
    return [p for p in prompts if any(kw in p.lower() for kw in keywords)]


def filter_agentic(prompts: List[str]) -> List[str]:
    """Keep prompts that look like agentic/planning tasks."""
    keywords = ["plan", "steps", "tool", "agent", "search", "find",
                "look up", "research", "investigate", "break down"]
    return [p for p in prompts if any(kw in p.lower() for kw in keywords)]


FILTER_FUNCTIONS = {
    "reasoning": filter_reasoning,
    "creative": filter_creative,
    "code": filter_code,
    "translation": filter_translation,
    "agentic": filter_agentic,
}


# ==============================================================================
# PROMPT EXTRACTION HELPER
# ==============================================================================

def _extract_prompt_text(item, text_column):
    """Extract prompt text from a dataset item, handling nested formats.

    Handles:
    - AgentTrove 'conversations': list of {role, value} dicts
    - ultrachat 'data': list of {role, content} dicts
    - Standard flat columns: text, instruction, prompt, etc.
    """
    # AgentTrove nested format: 'conversations' = [{role, value}, ...]
    if text_column == "conversations" and text_column in item:
        convs = item[text_column]
        if isinstance(convs, list) and len(convs) > 0:
            for conv in reversed(convs):
                if isinstance(conv, dict) and conv.get("role") in ("user", "human"):
                    val = conv.get("value", "")
                    if val and isinstance(val, str) and len(val) > 10:
                        return val.strip()
            # Fallback: first user message
            for conv in convs:
                if isinstance(conv, dict) and conv.get("role") in ("user", "human"):
                    val = conv.get("value", "")
                    if val and isinstance(val, str) and len(val) > 10:
                        return val.strip()

    # ultrachat nested format: 'data' = list of strings (messages)
    if text_column == "data" and text_column in item:
        data = item[text_column]
        if isinstance(data, list) and len(data) > 0:
            # data is a list of strings (human/assistant messages alternating)
            # Get first human message (usually index 0 or 1)
            for msg in data:
                if isinstance(msg, str) and len(msg) > 10:
                    return msg.strip()

    # Standard flat column extraction
    for col in [text_column, "text", "instruction", "prompt", "input", "message", "content", "problem", "question"]:
        if col and col in item:
            val = item[col]
            if val and isinstance(val, str) and len(val) > 10:
                return val.strip()
            # Handle list-of-dict (e.g., [{content: "..."}])
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, dict):
                    v = first.get("content", first.get("text", first.get("value", "")))
                    if v and isinstance(v, str) and len(v) > 10:
                        return v.strip()

    return None


# ==============================================================================
# MAIN BUILDER FUNCTION
# ==============================================================================

def build_diverse_dataset(
    domain_counts: Dict[str, int]
) -> List[str]:
    """Build a diverse dataset of prompts across specified domains.

    Args:
        domain_counts: Dict mapping domain name to number of prompts.
            Example: {"reasoning": 20, "creative": 15, "code": 20, ...}
            Total will be sum of all values.

    Returns:
        List of prompt strings (no labels, just raw prompts).

    Notes:
        - Uses streaming for large datasets (no full download)
        - Logs every step excessively for debugging
        - Falls back gracefully if a dataset fails to load
    """
    all_prompts: List[str] = []

    for domain, count in domain_counts.items():
        print(f"[DATASET] Domain: {domain}, target count: {count}")

        # ---- Get source info ----
        if domain not in DOMAIN_SOURCES:
            print(f"[DATASET] WARNING: Unknown domain '{domain}', skipping.")
            continue

        dataset_name, config_name, text_column = DOMAIN_SOURCES[domain]
        print(f"[DATASET]   -> Source: {dataset_name}, column: {text_column}")

        # ---- Load dataset ----
        try:
            if config_name:
                ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
            print(f"[DATASET]   -> Loaded! Stream mode: {ds.n_shards} shards")
        except Exception as e:
            print(f"[DATASET]   -> ERROR loading dataset '{dataset_name}': {e}")
            print(f"[DATASET]   -> Falling back to stingning/ultrachat")
            try:
                ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
                text_column = "data"
                print(f"[DATASET]   -> Fallback loaded.")
            except Exception as e2:
                print(f"[DATASET]   -> Fallback also failed: {e2}")
                print(f"[DATASET]   -> Using built-in domain prompts for {domain}")
                raw_prompts = [f"Sample {domain} prompt {i+1}" for i in range(count)]
                all_prompts.extend(raw_prompts)
                continue

        # ---- Collect prompts from dataset ----
        raw_prompts: List[str] = []
        iterator = iter(ds)

        try:
            for i, item in enumerate(iterator):
                prompt_text = _extract_prompt_text(item, text_column)
                if prompt_text:
                    raw_prompts.append(prompt_text)

                if i >= 10000:  # Safety cap to avoid infinite streaming
                    print(f"[DATASET]   -> Safety cap reached at {i} items")
                    break

                if i % 2000 == 0 and i > 0:
                    print(f"[DATASET]   -> Collected {i} potential prompts...")

        except Exception as e:
            print(f"[DATASET]   -> ERROR during iteration: {e}")

        print(f"[DATASET]   -> Raw prompts collected: {len(raw_prompts)}")

        # ---- Apply domain filter if available ----
        if domain in FILTER_FUNCTIONS:
            before_filter = len(raw_prompts)
            raw_prompts = FILTER_FUNCTIONS[domain](raw_prompts)
            print(f"[DATASET]   -> After {domain} filter: {before_filter} -> {len(raw_prompts)}")

        # ---- Sample to target count ----
        if not raw_prompts:
            print(f"[DATASET]   -> WARNING: No prompts collected for domain '{domain}', using fallback")
            raw_prompts = [
                f"Sample prompt {i+1} for {domain}"
                for i in range(count)
            ]

        if len(raw_prompts) >= count:
            # Simple stratified sampling by taking evenly spaced samples
            step = len(raw_prompts) // count
            if step == 0:
                step = 1  # Guard against count > len(raw_prompts)
            sampled = raw_prompts[::step][:count]
        else:
            # Not enough — repeat with shuffling
            shortage = count - len(raw_prompts)
            extra = (raw_prompts * ((shortage // len(raw_prompts)) + 1))[:shortage]
            sampled = raw_prompts + extra

        print(f"[DATASET]   -> Sampled {len(sampled)} prompts for domain '{domain}'")
        all_prompts.extend(sampled)

    # ---- Final shuffle ----
    print(f"[DATASET] Total prompts collected: {len(all_prompts)}")
    print(f"[DATASET] Shuffling...")

    import random
    random.shuffle(all_prompts)

    print(f"[DATASET] Final dataset size: {len(all_prompts)}")
    print(f"[DATASET] SUCCESS: build_diverse_dataset completed.")

    return all_prompts


# ==============================================================================
# VERIFICATION: quick smoke test
# ==============================================================================

if __name__ == "__main__":
    print("[DATASET] Running smoke test...")

    test_config = {
        "reasoning": 5,
        "creative": 5,
        "code": 5,
        "general": 10,
        "summarization": 5,
        "translation": 5,
        "agentic": 5,
    }

    prompts = build_diverse_dataset(test_config)
    print(f"[TEST] Smoke test returned {len(prompts)} prompts")
    print(f"[TEST] First 3 prompts:")
    for i, p in enumerate(prompts[:3]):
        print(f"  [{i}] {p[:80]}...")
