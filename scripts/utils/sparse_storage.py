"""Sparse storage utilities for neuron activation data.

Encodes/decodes sparse neuron activations to/from string representation for CSV storage.
Each neuron: (index, normalized_byte_value) where index < 4096 and value is 0-255.
"""

import ast
from typing import List, Tuple


def encode_sparse(neurons: List[Tuple[int, int]]) -> str:
    """Encode a list of (neuron_idx, normalized_value) tuples to a string.

    Args:
        neurons: List of (neuron_idx, normalized_byte_val) tuples.
            neuron_idx: int in range [0, 4095]
            normalized_byte_val: int in range [0, 255]

    Returns:
        String representation: "[(1234,189),(4567,234),...]"
    """
    return str(neurons)


def decode_sparse(encoded: str) -> List[Tuple[int, int]]:
    """Decode a string representation back to a list of (neuron_idx, value) tuples.

    Args:
        encoded: String like "[(1234,189),(4567,234),...]"

    Returns:
        List of (neuron_idx, normalized_byte_val) tuples.
    """
    return ast.literal_eval(encoded)


def verify_encoding(neurons: List[Tuple[int, int]]) -> bool:
    """Verify that neuron data is valid before encoding.

    Args:
        neurons: List of (neuron_idx, normalized_byte_val) tuples.

    Returns:
        True if all values are in valid range.

    Raises:
        ValueError: If any value is out of range.
    """
    if not neurons:
        return True

    for idx, val in neurons:
        if not (0 <= idx < 4096):
            raise ValueError(f"Neuron index {idx} out of range [0, 4095]")
        if not (0 <= val <= 255):
            raise ValueError(f"Neuron value {val} out of range [0, 255]")

    return True
