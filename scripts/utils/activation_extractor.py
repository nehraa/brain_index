"""Extract sparse active neurons from hidden state tensor.

For each layer's hidden state:
1. Apply ReLU (zero out negative values)
2. Get last token's hidden state (position of interest)
3. Extract top-k active neurons by activation magnitude
4. Normalize values to 0-255 byte range
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def extract_sparse_neurons(
    hidden_tensor: torch.Tensor,
    top_k: int = 300
) -> List[Tuple[int, int]]:
    """Extract top-k active neurons from a hidden state tensor.

    Uses ReLU activation to identify firing neurons, then selects the top-k
    by activation magnitude. Values normalized to 0-255 for storage efficiency.

    Args:
        hidden_tensor: Tensor of shape [seq_len, hidden_size] or [hidden_size].
            - For full tensor: uses last token position [-1, :]
            - For last-token tensor: uses [:]
            Supports shape [1, seq_len, hidden_size] as well.
        top_k: Maximum number of active neurons to extract. Default 300.
            Typical sparsity: ~200-400 neurons per layer (out of 4096).

    Returns:
        List of (neuron_idx, normalized_byte_val) tuples.
        neuron_idx: int in range [0, 4095]
        normalized_byte_val: int in range [0, 255]

    Example:
        >>> tensor = torch.randn(1, 20, 4096)  # batch=1, seq_len=20, hidden=4096
        >>> neurons = extract_sparse_neurons(tensor, top_k=300)
        >>> len(neurons)
        300
        >>> neurons[:3]
        [(1234, 189), (4567, 234), (789, 156)]
    """
    # ---- Handle shape: [batch, seq_len, hidden_size] or [1, seq_len, hidden_size] ----
    if hidden_tensor.dim() == 3:
        # Shape: [batch, seq_len, hidden_size]
        # Use last token's hidden state: position of interest for generation
        last_token = hidden_tensor[0, -1, :]  # shape: [4096]
    elif hidden_tensor.dim() == 2:
        # Shape: [batch, seq_len, hidden_size] where batch is actually seq
        last_token = hidden_tensor[-1, :]  # last position
    elif hidden_tensor.dim() == 1:
        # Shape: [hidden_size] — already extracted
        last_token = hidden_tensor
    else:
        raise ValueError(f"Unexpected tensor dim: {hidden_tensor.dim()}, shape: {hidden_tensor.shape}")

    # ---- Step 1: Apply ReLU — zero out non-firing neurons ----
    activated = F.relu(last_token)  # shape: [4096]

    # ---- Step 2: Find top-k by magnitude ----
    # Get magnitude of activation (already non-negative after ReLU)
    values, indices = activated.topk(k=min(top_k, activated.numel()))

    # ---- Step 3: Normalize to 0-255 byte range ----
    # Scale so max activation becomes 255
    max_val = values.max().item()
    if max_val > 0:
        normalized = ((values / max_val) * 255).round().int()
    else:
        # All zeros — return empty (no active neurons)
        normalized = values.new_zeros(len(values), dtype=torch.int)

    # ---- Step 4: Build list of (index, value) tuples ----
    neurons: List[Tuple[int, int]] = [
        (idx.item(), val.item())
        for idx, val in zip(indices, normalized)
    ]

    return neurons


def compute_layer_statistics(hidden_tensor: torch.Tensor) -> dict:
    """Compute statistics on a layer's hidden state for debugging/verification.

    Args:
        hidden_tensor: Tensor of shape [seq_len, hidden_size] or [hidden_size].

    Returns:
        Dictionary with statistics: min, max, mean, std, sparsity_pct,
        active_neuron_count.
    """
    if hidden_tensor.dim() == 3:
        last_token = hidden_tensor[0, -1, :]
    elif hidden_tensor.dim() == 2:
        last_token = hidden_tensor[-1, :]
    else:
        last_token = hidden_tensor

    activated = F.relu(last_token)
    total_neurons = activated.numel()
    active_count = (activated > 0).sum().item()
    sparsity_pct = 100.0 * (total_neurons - active_count) / total_neurons

    return {
        "min": activated.min().item(),
        "max": activated.max().item(),
        "mean": activated.mean().item(),
        "std": activated.std().item(),
        "sparsity_pct": sparsity_pct,
        "active_neuron_count": active_count,
        "total_neurons": total_neurons,
        "active_pct": 100.0 * active_count / total_neurons,
    }
