import torch

from afabench.common.datasets.datasets import XORNoisyShortcutDataset


def test_xor_noisy_shortcut_matches_theoretical_structure() -> None:
    dataset = XORNoisyShortcutDataset(
        n_samples=4096,
        seed=0,
        shortcut_accuracy=0.51,
    )
    features, labels = dataset.get_all_data()

    x1 = features[:, 0].long()
    x2 = features[:, 1].long()
    x3 = features[:, 2].long()
    y = labels.argmax(dim=1)

    assert torch.equal(torch.bitwise_xor(x1, x2), y)

    shortcut_acc = (x3 == y).float().mean().item()
    assert abs(shortcut_acc - 0.51) < 0.04

    assert torch.equal(
        dataset.get_feature_acquisition_costs(),
        torch.tensor([1.0, 1.0, 2.0]),
    )
