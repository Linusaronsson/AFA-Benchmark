#!/usr/bin/env python3
"""
Visualization script for the modified SyntheticMNISTDataset.
This script loads a pre-generated dataset bundle and visualizes the images to verify that:
1. The left half contains only noise
2. The right half contains class-specific patterns.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add the project root to the path to import the dataset
sys.path.append(str(Path(__file__).parent.parent.parent))

from afabench.common.bundle import load_bundle


def visualize_dataset_from_bundle(bundle_path: Path, n_samples_per_class=5):
    """
    Visualize samples from a pre-generated SyntheticMNISTDataset bundle.

    Args:
        bundle_path: Path to the dataset bundle file
        n_samples_per_class: Number of samples to show per class
    """
    # Load the dataset bundle
    print(f"Loading dataset from {bundle_path}")
    dataset, metadata = load_bundle(bundle_path)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Get all data
    features, labels = dataset.get_all_data()

    # Convert one-hot labels back to class indices
    class_indices = torch.argmax(labels, dim=1)

    # Find samples for each class
    samples_by_class = {}
    for class_idx in range(10):
        class_mask = class_indices == class_idx
        class_samples = features[class_mask]
        samples_by_class[class_idx] = class_samples[:n_samples_per_class]
        print(f"Class {class_idx}: {torch.sum(class_mask).item()} samples")

    # Create visualization
    fig, axes = plt.subplots(10, n_samples_per_class, figsize=(15, 20))
    fig.suptitle(
        f"SyntheticMNISTDataset from {bundle_path.name}\n"
        "Left Half = Noise, Right Half = Class Pattern",
        fontsize=16,
    )

    for class_idx in range(10):
        for sample_idx in range(n_samples_per_class):
            if sample_idx < len(samples_by_class[class_idx]):
                img = samples_by_class[class_idx][
                    sample_idx, 0
                ]  # Remove channel dimension

                # Plot the image
                ax = axes[class_idx, sample_idx]
                im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)

                # Add vertical line to show the division between left (noise) and right (pattern)
                ax.axvline(
                    x=13.5, color="red", linestyle="--", linewidth=1, alpha=0.7
                )

                # Labels
                if sample_idx == 0:
                    ax.set_ylabel(
                        f"Class {class_idx}", fontsize=12, fontweight="bold"
                    )
                if class_idx == 0:
                    ax.set_title(f"Sample {sample_idx + 1}", fontsize=10)

                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # If we don't have enough samples, hide the axis
                axes[class_idx, sample_idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def analyze_noise_vs_pattern_from_bundle(bundle_path: Path):
    """
    Analyze the statistical difference between left half (noise) and right half (pattern)
    from a pre-generated dataset bundle.
    """
    # Load the dataset bundle
    dataset, metadata = load_bundle(bundle_path)
    features, labels = dataset.get_all_data()
    class_indices = torch.argmax(labels, dim=1)

    print("Statistical Analysis of Left Half (Noise) vs Right Half (Pattern)")
    print("=" * 70)
    print(f"Dataset: {bundle_path}")
    if metadata:
        print(f"Metadata: {metadata}")
    print("=" * 70)

    for class_idx in range(10):
        class_mask = class_indices == class_idx
        class_samples = features[class_mask][
            :10
        ]  # Take first 10 samples of this class

        if len(class_samples) == 0:
            continue

        # Split into left and right halves
        left_half = class_samples[:, 0, :, :14]  # Shape: (samples, height, 14)
        right_half = class_samples[
            :, 0, :, 14:
        ]  # Shape: (samples, height, 14)

        # Calculate statistics
        left_mean = torch.mean(left_half).item()
        left_std = torch.std(left_half).item()
        right_mean = torch.mean(right_half).item()
        right_std = torch.std(right_half).item()

        print(f"Class {class_idx}:")
        print(
            f"  Left Half  (Noise):   Mean={left_mean:.4f}, Std={left_std:.4f}"
        )
        print(
            f"  Right Half (Pattern): Mean={right_mean:.4f}, Std={right_std:.4f}"
        )
        print(f"  Difference in Mean: {right_mean - left_mean:.4f}")
        print()


def save_sample_images_from_bundle(
    bundle_path: Path, save_dir="extra/visualizations/synthetic_mnist"
):
    """Save sample images to disk for further inspection."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Load the dataset bundle
    dataset, metadata = load_bundle(bundle_path)
    features, labels = dataset.get_all_data()
    class_indices = torch.argmax(labels, dim=1)

    # Save one sample per class
    for class_idx in range(10):
        class_mask = class_indices == class_idx
        if class_mask.any():
            sample = features[class_mask][
                0, 0
            ].numpy()  # First sample, remove channel dim

            # Save as image
            plt.figure(figsize=(4, 4))
            plt.imshow(sample, cmap="gray", vmin=0, vmax=1)
            plt.axvline(
                x=13.5, color="red", linestyle="--", linewidth=2, alpha=0.8
            )
            plt.title(f"Class {class_idx}")
            plt.axis("off")
            plt.tight_layout()

            bundle_name = bundle_path.stem.replace(".", "_")
            filename = f"{bundle_name}_class_{class_idx}.png"
            plt.savefig(
                save_path / filename,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    print(f"Sample images saved to {save_path}/")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SyntheticMNISTDataset from a bundle file"
    )
    parser.add_argument(
        "bundle_path",
        type=Path,
        help="Path to the dataset bundle file (e.g., train.bundle)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=3,
        help="Number of samples to show per class (default: 3)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save sample images to disk",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run statistical analysis, skip visualization",
    )

    args = parser.parse_args()

    if not args.bundle_path.exists():
        print(f"Error: Bundle path {args.bundle_path} does not exist")
        sys.exit(1)

    if not args.bundle_path.suffix == ".bundle":
        print("Error: Path must end with .bundle extension")
        sys.exit(1)

    print("Visualizing SyntheticMNISTDataset from bundle...")
    print("Red dashed line shows division: Left=Noise, Right=Pattern")
    print()

    try:
        # Always run statistical analysis
        analyze_noise_vs_pattern_from_bundle(args.bundle_path)

        # Run visualization unless analysis-only is specified
        if not args.analysis_only:
            visualize_dataset_from_bundle(
                args.bundle_path, n_samples_per_class=args.samples_per_class
            )

        # Save images if requested
        if args.save_images:
            save_sample_images_from_bundle(args.bundle_path)

        print("Visualization complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
