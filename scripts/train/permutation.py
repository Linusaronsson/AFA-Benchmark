import gc
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchrl.modules import MLP
from tqdm import tqdm

from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.utils import afa_discriminative_training_prep
from afabench.common.config_classes import PermutationTrainingConfig
from afabench.common.bundle import save_bundle
from afabench.common.utils import set_seed
from afabench.static.models import BaseModel
from afabench.static.static_methods import StaticBaseMethod
from afabench.static.utils import transform_dataset

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/permutation",
    config_name="config",
)
def main(cfg: PermutationTrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
    if cfg.smoke_test:
        cfg.selector.nepochs = 1
        cfg.selector.patience = 1
        cfg.classifier.nepochs = 1

    train_dataset, val_dataset, _, _, class_weights = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
        )
    )
    assert class_weights is not None
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    def auroc_metric(pred: torch.Tensor, y: torch.Tensor):
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
    x_train, _ = train_dataset.get_all_data()
    for i in tqdm(range(d_in)):
        x_val, y_val = val_dataset.get_all_data()
        x_val = x_val.clone()
        y_val = y_val.argmax(dim=1).long()
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

    save_bundle(
        obj=static_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )
    log.info(f"Permutation method saved to: {cfg.save_path}")

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
