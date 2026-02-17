"""Tests whether the ReadProcessEncoder class behaves as expected."""

import torch

from afabench.afa_rl.shim2018.models import ReadProcessEncoder


def test_is_permutation_invariant() -> None:
    """Tests whether the ReadProcessEncoder class is invariant to permutations."""
    # Create a ReadProcessEncoder instance
    encoder = ReadProcessEncoder(
        set_element_size=2,
        output_size=3,
        reading_block_cells=(2, 2),
        writing_block_cells=(2, 2),
        memory_size=4,
        processing_steps=5,
    )

    # Test two batches of input data
    input_set_11 = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8], [0, 0]], dtype=torch.float32
    )
    length_11 = torch.tensor(4, dtype=torch.int64)
    input_set_12 = torch.tensor(
        [[8, 9], [10, 11], [11, 12], [0, 0], [0, 0]], dtype=torch.float32
    )
    length_12 = torch.tensor(3, dtype=torch.int64)
    input_set_1 = torch.stack([input_set_11, input_set_12])
    length_1 = torch.stack([length_11, length_12])
    output_1 = encoder(input_set_1, length_1)

    # Same input data, but permuted
    input_set_21 = torch.tensor(
        [[3, 4], [1, 2], [7, 8], [5, 6], [0, 0]], dtype=torch.float32
    )
    length_21 = torch.tensor(4, dtype=torch.int64)
    input_set_22 = torch.tensor(
        [[11, 12], [10, 11], [8, 9], [0, 0], [0, 0]], dtype=torch.float32
    )
    length_22 = torch.tensor(3, dtype=torch.int64)
    input_set_2 = torch.stack([input_set_21, input_set_22])
    length_2 = torch.stack([length_21, length_22])
    output_2 = encoder(input_set_2, length_2)

    # Check that the outputs are equal
    assert torch.allclose(output_1, output_2), (
        "The outputs are not equal for the same input data with different permutations."
    )
