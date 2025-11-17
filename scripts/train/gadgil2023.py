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
from torchmetrics import Accuracy
from tempfile import TemporaryDirectory

from afabench.afa_discriminative.models import fc_Net
from afabench.afa_discriminative.utils import MaskLayer
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.afa_methods import (
    CMIEstimator,
    Gadgil2023AFAMethod,
)

from afabench.common.config_classes import (
    Gadgil2023PretrainingConfig,
    Gadgil2023TrainingConfig,
)
from afabench.common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023TrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["DIME"],
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
    pretrained_model_config: Gadgil2023PretrainingConfig = from_dict(
        data_class=Gadgil2023PretrainingConfig,
        data=pretrained_model_config_dict,
    )
    train_dataset, val_dataset, test_dataset, dataset_metadata = (
        load_dataset_artifact(pretrained_model_config.dataset_artifact_name)
    )
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

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

    checkpoint = torch.load(
        str(pretrained_model_artifact_dir / "model.pt"),
        map_location=device,
        weights_only=False,
    )
    predictor.load_state_dict(checkpoint["predictor_state_dict"])

    value_network = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_in,
        hidden_layer_num=len(cfg.hidden_units),
        hidden_unit=cfg.hidden_units,
        activations=cfg.activations,
        drop_out_rate=cfg.dropout,
        flag_drop_out=cfg.flag_drop_out,
        flag_only_output_layer=cfg.flag_only_output_layer,
    )

    value_network.hidden[0] = predictor.hidden[0]
    value_network.hidden[1] = predictor.hidden[1]
    mask_layer = MaskLayer(append=True)

    greedy_cmi_estimator = CMIEstimator(
        value_network, predictor, mask_layer
    ).to(device)
    greedy_cmi_estimator.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        eps=cfg.eps,
        loss_fn=nn.CrossEntropyLoss(reduction="none", weight=class_weights),
        val_loss_fn=Accuracy(task="multiclass", num_classes=d_out).to(device),
        val_loss_mode="max",
        eps_decay=cfg.eps_decay,
        eps_steps=cfg.eps_steps,
        patience=cfg.patience,
        feature_costs=train_dataset.get_feature_acquisition_costs().to(device),
    )

    afa_method = Gadgil2023AFAMethod(
        greedy_cmi_estimator.value_network.cpu(),
        greedy_cmi_estimator.predictor.cpu(),
    )

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)
        del afa_method
        afa_method = Gadgil2023AFAMethod.load(tmp_path, device=device)
        afa_method_artifact = wandb.Artifact(
            name=f"train_gadgil2023-{
                pretrained_model_config.dataset_artifact_name.split(':')[0]
            }-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "gadgil2023",
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
