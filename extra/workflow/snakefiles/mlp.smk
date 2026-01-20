"""
Snakemake workflow for training masked MLP classifiers.

This workflow trains masked MLP classifiers that serve as prerequisites for
AACO and other AFA methods. Run this workflow first to have classifiers ready
for downstream experiments.

Usage:
    snakemake -s extra/workflow/snakefiles/mlp.smk \
        --configfile extra/workflow/conf/mlp.yaml -j 4
"""

# Configuration with defaults
DATASET_PATH_PREFIX = config.get("dataset_path_prefix", "extra/data")
OUTPUT_PATH_PREFIX = config.get("output_path_prefix", "extra/output/classifiers")
DEVICE = config.get("device", "cpu")
USE_WANDB = config.get("use_wandb", False)
SMOKE_TEST = config.get("smoke_test", False)

# Datasets to process (required)
DATASETS = config.get("datasets", None)
if DATASETS is None:
    raise ValueError("'datasets' must be provided in config")

# Dataset instances (seeds) to train on
DATASET_INSTANCE_INDICES = config.get("dataset_instance_indices", [0, 1, 2, 3, 4])

# Training hyperparameters (with sensible defaults)
EPOCHS = config.get("epochs", 100)
BATCH_SIZE = config.get("batch_size", 128)
LR = config.get("lr", 1e-3)
_num_cells_raw = config.get("num_cells", [128, 128])
# Format as Hydra list: [128,128] (no spaces)
NUM_CELLS = "[" + ",".join(str(x) for x in _num_cells_raw) + "]"
DROPOUT = config.get("dropout", 0.1)
MIN_MASKING_PROB = config.get("min_masking_probability", 0.0)
MAX_MASKING_PROB = config.get("max_masking_probability", 0.9)


# ============================================================================
# MAIN TARGETS
# ============================================================================

rule all:
    """Train classifiers for all configured datasets and instances."""
    input:
        [
            f"{OUTPUT_PATH_PREFIX}/masked_mlp_classifier/"
                f"dataset-{dataset}+"
                f"instance_idx-{idx}/"
                    f"seed-{idx}.bundle"
            for dataset in DATASETS
            for idx in DATASET_INSTANCE_INDICES
        ]


rule train_single_dataset:
    """Train classifiers for a single dataset (all instances)."""
    input:
        lambda wc: [
            f"{OUTPUT_PATH_PREFIX}/masked_mlp_classifier/"
                f"dataset-{wc.dataset}+"
                f"instance_idx-{idx}/"
                    f"seed-{idx}.bundle"
            for idx in DATASET_INSTANCE_INDICES
        ]


# ============================================================================
# TRAINING RULES
# ============================================================================

rule train_masked_mlp_classifier:
    """Train a masked MLP classifier for a specific dataset instance."""
    input:
        train=lambda wc: f"{DATASET_PATH_PREFIX}/{wc.dataset}/{wc.instance_idx}/train.bundle",
        val=lambda wc: f"{DATASET_PATH_PREFIX}/{wc.dataset}/{wc.instance_idx}/val.bundle",
    output:
        directory(
            f"{OUTPUT_PATH_PREFIX}/masked_mlp_classifier/"
                "dataset-{dataset}+"
                "instance_idx-{instance_idx}/"
                    "seed-{seed}.bundle"
        ),
    params:
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_cells=NUM_CELLS,
        dropout=DROPOUT,
        min_mask=MIN_MASKING_PROB,
        max_mask=MAX_MASKING_PROB,
    shell:
        """
        python scripts/train/masked_mlp_classifier.py \
            train_dataset_path={input.train} \
            val_dataset_path={input.val} \
            save_path={output} \
            epochs={params.epochs} \
            batch_size={params.batch_size} \
            lr={params.lr} \
            "num_cells={params.num_cells}" \
            dropout={params.dropout} \
            min_masking_probability={params.min_mask} \
            max_masking_probability={params.max_mask} \
            device={DEVICE} \
            seed={wildcards.seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        """


# ============================================================================
# UTILITY RULES
# ============================================================================

rule list_classifiers:
    """List all available trained classifiers."""
    shell:
        """
        echo "Trained classifiers:"
        find {OUTPUT_PATH_PREFIX}/masked_mlp_classifier -name "*.bundle" -type d 2>/dev/null || echo "No classifiers found"
        """


rule clean_classifiers:
    """Remove all trained classifiers (use with caution)."""
    shell:
        """
        echo "This would remove: {OUTPUT_PATH_PREFIX}/masked_mlp_classifier"
        echo "Run 'rm -rf {OUTPUT_PATH_PREFIX}/masked_mlp_classifier' manually to confirm"
        """
