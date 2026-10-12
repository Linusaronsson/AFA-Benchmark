from typing import ClassVar, final, override

import torch

from afabench.core.types import Features, Label
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot


@final
class SyntheticMNISTDataset(InMemoryAFADataset):
    """
    Synthetic MNIST-like dataset with the same shape as MNIST (28x28 = 784 features, 10 classes).

    Generates synthetic image-like data with patterns that can be learned.
    The left half (14 pixels) contains only noise, while the right half contains
    class-specific patterns for label identification.

    Implements the AFADataset protocol.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "noise_std",
        "pattern_intensity",
    )

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size(
            [1, 28, 28]
        )  # (channels, height, width) - proper image format

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([10])  # 10 classes, same as MNIST

    def __init__(
        self,
        seed: int = 123,
        # Memory-friendly default (60k samples ≈ 200MB RAM)
        n_samples: int = 10000,
        noise_std: float = 0.1,
        pattern_intensity: float = 0.8,
    ):
        self.seed = seed
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.pattern_intensity = pattern_intensity
        super().__init__()

    @override
    def _build(self) -> tuple[Features, Label]:  # noqa: C901, PLR0915, PLR0912
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Generate labels uniformly
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=rng,
        )

        # Initialize features with noise in image format (C, H, W)
        features = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=(self.n_samples, 1, 28, 28),
            generator=rng,
        )

        # Pre-compute coordinate grids for optimized pattern generation
        y_coords, x_coords = torch.meshgrid(
            torch.arange(28, dtype=torch.float32),
            torch.arange(28, dtype=torch.float32),
            indexing="ij",
        )

        # Add class-specific patterns to make the data learnable (right half only)
        for i in range(self.n_samples):
            label = int(y_int[i].item())

            # Create simple geometric patterns for each class
            img = features[i, 0]  # Get the single channel (28, 28)

            # Ensure left half (0:14) remains pure noise - regenerate it
            img[:, 0:14] = torch.normal(
                mean=0.0,
                std=self.noise_std,
                size=(28, 14),
                generator=rng,
            )

            if label == 0:  # Circle-like pattern (right half only)
                center_y, center_x = 14.0, 21.0  # Center in right half
                dist = torch.sqrt(
                    (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
                )
                circle_mask = (dist >= 4) & (
                    dist <= 6
                )  # Smaller circle for half space
                # Only apply to right half
                circle_mask = circle_mask & (x_coords >= 14)
                img[circle_mask] += self.pattern_intensity

            elif label == 1:  # Vertical line (right half)
                img[:, 19:23] += self.pattern_intensity  # Moved to right half

            elif label == 2:  # Horizontal line (right half only)
                img[12:16, 14:] += self.pattern_intensity  # Only right half

            elif label == 3:  # Diagonal line (right half)
                # Create diagonal in right half only
                for j in range(14):
                    if j < 28 and (j + 14) < 28:
                        img[j, j + 14] += self.pattern_intensity

            elif label == 4:  # Square (right half)
                img[8:20, 16:26] = (
                    self.pattern_intensity
                )  # Moved to right half
                img[8:10, 16:26] = self.pattern_intensity
                img[18:20, 16:26] = self.pattern_intensity
                img[8:20, 16:18] = self.pattern_intensity
                img[8:20, 24:26] = self.pattern_intensity

            elif label == 5:  # Cross pattern (right half)
                img[:, 19:23] = (
                    self.pattern_intensity
                )  # Vertical line in right half
                img[12:16, 14:] = (
                    self.pattern_intensity
                )  # Horizontal line in right half

            elif label == 6:  # Triangle-like (right half)
                for y in range(6, 22):
                    width = (y - 6) // 2
                    start_x = max(14, 21 - width)  # Centered in right half
                    end_x = min(28, 21 + width + 1)
                    img[y, start_x:end_x] += self.pattern_intensity

            elif label == 7:  # L-shape (right half)
                img[6:22, 16:20] = (
                    self.pattern_intensity
                )  # Vertical part in right half
                img[18:22, 16:26] = (
                    self.pattern_intensity
                )  # Horizontal part in right half

            elif label == 8:  # X pattern (right half)
                # Main diagonal in right half
                for j in range(14):
                    if j < 28 and (j + 14) < 28:
                        img[j, j + 14] += self.pattern_intensity
                # Anti-diagonal in right half
                for j in range(14):
                    if j < 28 and (27 - j) >= 14:
                        img[j, 27 - j] += self.pattern_intensity

            elif label == 9:  # Dot pattern (right half only)
                img[6:8, 16:18] += (
                    self.pattern_intensity
                )  # Top row, right half
                img[6:8, 20:22] += self.pattern_intensity
                img[6:8, 24:26] += self.pattern_intensity
                img[10:12, 16:18] += (
                    self.pattern_intensity
                )  # Second row, right half
                img[10:12, 20:22] += self.pattern_intensity
                img[10:12, 24:26] += self.pattern_intensity
                img[14:16, 16:18] += (
                    self.pattern_intensity
                )  # Third row, right half
                img[14:16, 20:22] += self.pattern_intensity
                img[14:16, 24:26] += self.pattern_intensity
                img[18:20, 16:18] += (
                    self.pattern_intensity
                )  # Fourth row, right half
                img[18:20, 20:22] += self.pattern_intensity
                img[18:20, 24:26] += self.pattern_intensity

            # Update the channel
            features[i, 0] = img

        # Normalize features to [0, 1] range like MNIST
        features = torch.clamp(features, 0, 1)
        return features, one_hot(y_int, self.label_shape[0])
