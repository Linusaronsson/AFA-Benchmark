import gc
import torch
import wandb
import hydra
import logging
import numpy as np

from torch import nn
from tqdm import tqdm
from pathlib import Path
from typing import Any, cast
from torchmetrics import AUROC
from torchrl.modules import MLP
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader


from afabench.static.models import BaseModel
from afabench.static.utils import transform_dataset
from afabench.static.static_methods import StaticBaseMethod
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.common.config_classes import PermutationTrainingConfig

from afabench.common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/permutation",
    config_name="config",
)
def main(cfg: PermutationTrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["permutation"],
        dir="extra/wandb",
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    def auroc_metric(pred, y):
        return AUROC(task="multiclass", num_classes=d_out)(
            pred.softmax(dim=1), y
        )

    predictors: dict[int, nn.Module] = {}
    selected_history: dict[int, list[int]] = {}
    num_features = list(range(1, cfg.hard_budget + 1))

    model = MLP(
        in_features=d_in,
        out_features=d_out,
        num_cells=cfg.selector.num_cells,
        activation_class=nn.ReLU,
    )
    basemodel = BaseModel(model).to(device)
    basemodel.fit(
        train_loader,
        val_loader,
        lr=cfg.selector.lr,
        nepochs=cfg.selector.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        verbose=False,
    )

    permutation_importance = np.zeros(d_in)
    x_train = train_dataset.features
    for i in tqdm(range(d_in)):
        x_val = val_dataset.features.clone()
        y_val = val_dataset.labels
        x_val[:, i] = x_train[
            np.random.choice(len(x_train), size=len(x_val)), i
        ]
        with torch.no_grad():
            pred = model(x_val.to(device)).cpu()
            permutation_importance[i] = -auroc_metric(pred, y_val)

    ranked_features = np.argsort(permutation_importance)[::-1]

    for num in num_features:
        selected_features = ranked_features[:num]
        selected_history[num] = selected_features.tolist()
        train_subset = transform_dataset(
            train_dataset, selected_features.copy()
        )
        val_subset = transform_dataset(val_dataset, selected_features.copy())
        train_subset_loader = DataLoader(
            train_subset,
            batch_size=128,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_subset_loader = DataLoader(
            val_subset, batch_size=1024, pin_memory=True
        )

        model = MLP(
            in_features=num,
            out_features=d_out,
            num_cells=cfg.classifier.num_cells,
            activation_class=nn.ReLU,
        )
        predictor = BaseModel(model).to(device)
        predictor.fit(
            train_subset_loader,
            val_subset_loader,
            lr=cfg.classifier.lr,
            nepochs=cfg.classifier.nepochs,
            loss_fn=nn.CrossEntropyLoss(weight=class_weights),
            verbose=False,
        )

        predictors[num] = model

    static_method = StaticBaseMethod(selected_history, predictors, device)

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        static_method.save(tmp_path)
        del static_method
        static_method = StaticBaseMethod.load(tmp_path, device=device)
        static_method_artifact = wandb.Artifact(
            name=f"train_permutation-{
                cfg.dataset_artifact_name.split(':')[0]
            }-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "permutation",
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )
        static_method_artifact.add_dir(str(tmp_path))
        run.log_artifact(
            static_method_artifact, aliases=cfg.output_artifact_aliases
        )

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
