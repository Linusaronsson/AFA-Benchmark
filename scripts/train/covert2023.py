import gc
import wandb
import torch
import hydra
import logging

from torch import nn
from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory

from afabench import SAVE_PATH
from afabench.afa_discriminative.afa_methods import (
    Covert2023AFAMethod,
    GreedyDynamicSelection,
)
from afabench.afa_discriminative.models import fc_Net
from afabench.afa_discriminative.utils import MaskLayer
from afabench.afa_discriminative.datasets import prepare_datasets

from afabench.common.config_classes import (
    Covert2023TrainingConfig,
)

from afabench.common.utils import (
    get_class_probabilities,
    load_pretrained_model,
    load_dataset,
    save_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/covert2023",
    config_name="config",
)
def main(cfg: Covert2023TrainingConfig):
    log.debug(cfg)
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["GDFS", "covert2023"],
        dir="extra/wandb",
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Load pretrained Covert2023 method
    pretrained_method, metadata, pretrain_config = load_pretrained_model(
        f"{cfg.pretrained_model_artifact_name}_seed_{cfg.seed}",
        Covert2023AFAMethod,
        device,
    )

    # Load dataset used in pretraining
    train_dataset, val_dataset, _, dataset_metadata = load_dataset(
        metadata["dataset_artifact_name"]
    )

    dataset_type = dataset_metadata["dataset_type"]
    split = dataset_metadata["split_idx"]

    log.info(f"Dataset: {dataset_type}, Split: {split}")
    log.info(f"Training samples: {len(train_dataset)}")

    # Prepare loaders
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)

    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    # Build predictor + load pretrained weights
    predictor = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_out,
        hidden_layer_num=len(cfg.hidden_units),
        hidden_unit=cfg.hidden_units,
        activations=cfg.activations,
        drop_out_rate=cfg.dropout,
        flag_drop_out=cfg.flag_drop_out,
        flag_only_output_layer=cfg.flag_only_output_layer,
    )
    predictor.load_state_dict(pretrained_method.predictor.state_dict())

    # Build fresh selector for supervised GDFS
    selector = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_in,
        hidden_layer_num=len(cfg.hidden_units),
        hidden_unit=cfg.hidden_units,
        activations=cfg.activations,
        drop_out_rate=cfg.dropout,
        flag_drop_out=cfg.flag_drop_out,
        flag_only_output_layer=cfg.flag_only_output_layer,
    )

    # Train GDFS
    mask_layer = MaskLayer(append=True)
    gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)

    gdfs.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.patience,
        verbose=True,
        feature_costs=train_dataset.get_feature_acquisition_costs().to(device),
    )

    # Wrap into a final AFA method
    afa_method = Covert2023AFAMethod(
        selector=gdfs.selector.cpu(),
        predictor=gdfs.predictor.cpu(),
        device=torch.device("cpu"),
        modality="tabular",
    )

    # Save to temporary directory then promote to artifact
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        afa_method.save(temp_path)

        artifact_identifier = f"{dataset_type.lower()}_split_{split}_budget_{
            cfg.hard_budget
        }_seed_{cfg.seed}"
        artifact_dir = SAVE_PATH / artifact_identifier

        # Metadata follows the AACO pattern
        metadata = {
            "method_type": afa_method.__class__.__name__,
            "dataset_type": dataset_type,
            "dataset_artifact_name": metadata["dataset_artifact_name"],
            "budget": cfg.hard_budget,
            "seed": cfg.seed,
            "split_idx": split,
        }

        save_artifact(
            artifact_dir=artifact_dir,
            files={f.name: f for f in temp_path.iterdir() if f.is_file()},
            metadata=metadata,
        )

        log.info(f"Covert2023 method saved to: {artifact_dir}")

    run.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
