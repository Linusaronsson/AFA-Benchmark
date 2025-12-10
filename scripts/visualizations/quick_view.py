#!/usr/bin/env python3
"""Quick visualization script to view a few samples from a pre-generated SyntheticMNISTDataset bundle."""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import torch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from afabench.common.bundle import load_bundle

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset


def quick_view(
    bundle_path: Path,
    output_file: str = "extra/visualizations/synthetic_mnist/quick_view.png",
) -> None:
    """Show a quick sample of images from each class."""
    # Load the dataset bundle
    print(f"Loading dataset from {bundle_path}")
    dataset, _ = load_bundle(bundle_path)
    dataset = cast("AFADataset", cast("object", dataset))
    print(f"Loaded dataset with {len(dataset)} samples")

    # Get data
    features, labels = dataset.get_all_data()
    class_indices = torch.argmax(labels, dim=1)

    # Create a simple grid
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(
        f"SyntheticMNISTDataset from {bundle_path.name}: Left=Noise, Right=Pattern",
        fontsize=14,
    )

    for class_idx in range(10):
        row = class_idx // 5
        col = class_idx % 5
        ax = axes[row, col]

        # Find first sample of this class
        class_mask = class_indices == class_idx
        if class_mask.any():
            sample = features[class_mask][0, 0]  # First sample, remove channel

            # Display image
            ax.imshow(sample, cmap="gray", vmin=0, vmax=1)

            # Add red line to show division
            ax.axvline(
                x=13.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8
            )

            ax.set_title(f"Class {class_idx}", fontsize=12)

            # Print sample count for this class
            print(f"Class {class_idx}: {torch.sum(class_mask).item()} samples")
        else:
            ax.text(
                0.5,
                0.5,
                f"Class {class_idx}\nNo samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            print(f"Class {class_idx}: 0 samples")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Visualization saved as '{output_file}'")
    print(
        "Red dashed line shows the division: Left half = noise, Right half = class-specific pattern"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick visualization of SyntheticMNISTDataset from a bundle file"
    )
    parser.add_argument(
        "bundle_path",
        type=Path,
        help="Path to the dataset bundle file (e.g., train.bundle)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extra/visualizations/synthetic_mnist/quick_view.png",
        help="Output filename for the visualization (default: extra/visualizations/synthetic_mnist/quick_view.png)",
    )

    args = parser.parse_args()

    if not args.bundle_path.exists():
        print(f"Error: Bundle path {args.bundle_path} does not exist")
        sys.exit(1)

    if args.bundle_path.suffix != ".bundle":
        print("Error: Path must end with .bundle extension")
        sys.exit(1)

    try:
        quick_view(args.bundle_path, args.output)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
