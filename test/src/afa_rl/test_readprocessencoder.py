"""Tests whether the ReadProcessEncoder class behaves as expected."""

import torch

from afabench.components.methods.rl.jafa.models import (
    JAFAEmbedder,
    ReadProcessEncoder,
)


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


def test_ignores_padding_and_represents_empty_sets() -> None:
    encoder = ReadProcessEncoder(
        set_element_size=2,
        output_size=3,
        reading_block_cells=(2,),
        writing_block_cells=(2,),
        memory_size=4,
        processing_steps=2,
    )
    lengths = torch.tensor([2, 0])
    input_set = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    changed_padding = input_set.clone()
    changed_padding[0, 2] = torch.tensor([100.0, -100.0])
    changed_padding[1] = 100.0

    output = encoder(input_set, lengths)
    changed_output = encoder(changed_padding, lengths)

    assert torch.allclose(output, changed_output)
    assert torch.equal(output[1], encoder.empty_set_vector)


def test_jafa_embedder_matches_per_instance_loop() -> None:
    torch.manual_seed(0)
    n_features = 4
    encoder = ReadProcessEncoder(
        set_element_size=n_features + 1,
        output_size=3,
        reading_block_cells=(8,),
        writing_block_cells=(8,),
        memory_size=6,
        processing_steps=3,
    )
    with torch.no_grad():
        encoder.empty_set_vector.copy_(torch.tensor([0.25, -0.5, 1.0]))
    embedder = JAFAEmbedder(encoder)
    masked_features = torch.tensor(
        [
            [1.5, 0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, -1.0, 2.5, 3.0],
            [0.0, 0.0, 0.0, -0.75],
        ]
    )
    feature_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, False, False, False],
            [True, True, True, True],
            [False, False, False, True],
        ]
    )

    expected = []
    for features, mask in zip(masked_features, feature_mask, strict=True):
        observed_indices = mask.nonzero(as_tuple=True)[0]
        if len(observed_indices) == 0:
            expected.append(encoder.empty_set_vector)
            continue
        elements = torch.cat(
            (
                features[observed_indices].unsqueeze(-1),
                torch.eye(n_features)[observed_indices],
            ),
            dim=-1,
        )
        expected.append(
            encoder(
                elements.unsqueeze(0),
                torch.tensor([len(observed_indices)]),
            ).squeeze(0)
        )

    embedding = embedder(masked_features, feature_mask)

    assert torch.allclose(embedding, torch.stack(expected), atol=1e-6)
    assert torch.equal(embedding[1], encoder.empty_set_vector)
