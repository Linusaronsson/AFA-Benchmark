import gc
import wandb
import hydra
import torch
import logging

from torch import nn
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory

from afabench.afa_discriminative.utils import MaskLayer2d
from afabench.common.config_classes import Covert2023Pretraining2DConfig

from afabench.afa_discriminative.models import (
    MaskingPretrainer,
    Predictor,
    ResNet18Backbone,
    resnet18,
)
from afabench.common.utils import (
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/pretrain/covert2023",
    config_name="config",
)
def main(cfg: Covert2023Pretraining2DConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        group="pretrain_covert2023",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        tags=["GDFS"],
        dir="extra/wandb",
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_dataset, val_dataset, _, _ = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,  # type: ignore
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,  # type: ignore
    )
    d_out = train_dataset.n_classes
    # train_class_probabilities = get_class_probabilities(train_dataset.labels)
    # class_weights = len(train_class_probabilities) / (
    #     len(train_class_probabilities) * train_class_probabilities
    # )
    # class_weights = class_weights.to(device)
    # train_loader, val_loader, d_in, d_out = prepare_datasets(
    #     train_dataset, val_dataset, cfg.batch_size
    # )

    base = resnet18(pretrained=True)
    backbone, expansion = ResNet18Backbone(base)
    predictor = Predictor(backbone, expansion, num_classes=d_out).to(device)

    # default 224x224 image, 14x14 mask grid (16 patch size)
    image_size = cfg.image_size
    patch_size = cfg.patch_size
    assert image_size % patch_size == 0, (
        "image_size must be divisible by patch_size"
    )
    mask_width = image_size // patch_size

    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)

    pretrain.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        # loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
    )

    pretrained_model_artifact = wandb.Artifact(
        name=f"pretrain_covert2023-{cfg.dataset_artifact_name.split(':')[0]}",
        type="pretrained_model",
    )
    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        torch.save(
            {
                "predictor_state_dict": pretrain.model.state_dict(),
                "architecture": {
                    "backbone": "resnet18",
                    "image_size": image_size,
                    "patch_size": patch_size,
                    "mask_width": mask_width,
                    "d_out": d_out,
                },
            },
            tmp_path / "model.pt",
        )

    pretrained_model_artifact.add_file(str(tmp_path / "model.pt"))
    run.log_artifact(
        pretrained_model_artifact,
        aliases=[
            *cfg.output_artifact_aliases,
            datetime.now().strftime("%b%d"),
        ],
    )
    run.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
