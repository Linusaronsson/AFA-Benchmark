import gc
import torch
import wandb
import hydra
import logging

from torch import nn
from pathlib import Path
from typing import Any, cast
from dacite import from_dict
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader


from afabench.afa_discriminative.utils import MaskLayer2d

from afabench.afa_discriminative.afa_methods import (
    Covert2023AFAMethod,
    GreedyDynamicSelection,
)
from afabench.afa_discriminative.models import (
    ConvNet,
    Predictor,
    ResNet18Backbone,
    resnet18,
)
from afabench.common.config_classes import (
    Covert2023Pretraining2DConfig,
    Covert2023Training2DConfig,
)
from afabench.common.utils import (
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/covert2023",
    config_name="config",
)
def main(cfg: Covert2023Training2DConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["GDFS"],
        dir="extra/wandb",
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    pretrained_model_artifact = wandb.use_artifact(
        cfg.pretrained_model_artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    artifact_filenames = [
        f.name for f in pretrained_model_artifact_dir.iterdir()
    ]
    assert {"model.pt"}.issubset(
        artifact_filenames
    ), f"Dataset artifact must contain a model.pt file. Instead found: {
        artifact_filenames
    }"
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    pretrained_model_config: Covert2023Pretraining2DConfig = from_dict(
        data_class=Covert2023Pretraining2DConfig,
        data=pretrained_model_config_dict,
    )
    train_dataset, val_dataset, test_dataset, dataset_metadata = (
        load_dataset_artifact(pretrained_model_config.dataset_artifact_name)
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

    checkpoint = torch.load(
        str(pretrained_model_artifact_dir / "model.pt"),
        map_location=device,
        weights_only=False,
    )
    predictor.load_state_dict(checkpoint["predictor_state_dict"])

    arch = checkpoint["architecture"]
    image_size = arch["image_size"]
    patch_size = arch["patch_size"]
    assert image_size % patch_size == 0
    mask_width = arch["mask_width"]
    n_patches = int(mask_width) ** 2

    selector = ConvNet(backbone, expansion).to(device)
    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        logits0 = selector(
            mask_layer(
                x0.to(device), torch.zeros(len(x0), n_patches, device=device)
            )
        )
    assert logits0.shape[1] == n_patches, (
        f"Selector outputs {logits0.shape[1]} != n_patches {n_patches}"
    )

    gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
    gdfs.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        min_lr=cfg.min_lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        # loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.patience,
        verbose=True,
    )

    afa_method = Covert2023AFAMethod(
        gdfs.selector.cpu(),
        gdfs.predictor.cpu(),
        modality="image",
        n_patches=n_patches,
    )
    afa_method.image_size = image_size
    afa_method.patch_size = patch_size
    afa_method.mask_width = mask_width

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)
        del afa_method
        afa_method = Covert2023AFAMethod.load(tmp_path, device=device)
        afa_method_artifact = wandb.Artifact(
            name=f"train_covert2023-{
                pretrained_model_config.dataset_artifact_name.split(':')[0]
            }-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "covert2023",
                "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )
        afa_method_artifact.add_file(str(tmp_path / "model.pt"))
        run.log_artifact(
            afa_method_artifact, aliases=cfg.output_artifact_aliases
        )

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
