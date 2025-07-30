import os
import gc
import wandb
import hydra
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torchmetrics import Accuracy
from tempfile import TemporaryDirectory
from omegaconf import DictConfig, OmegaConf
from afa_discriminative.utils import MaskLayer
from afa_discriminative.models import MaskingPretrainer, fc_Net
from afa_discriminative.datasets import prepare_datasets
from common.utils import load_dataset_artifact, set_seed, get_class_probabilities
from common.config_classes import Gadgil2023PretrainingConfig


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/pretrain/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023PretrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        group="pretrain_gadgil2023",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_dataset, val_dataset, _, _ = load_dataset_artifact(cfg.dataset_artifact_name)
    train_loader, val_loader, d_in, d_out = prepare_datasets(train_dataset, val_dataset, cfg.batch_size)
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    class_weights = F.softmax(1 / train_class_probabilities, dim=-1).to(device)

    predictor = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_out,
        hidden_layer_num=len(cfg.hidden_units),
        hidden_unit=cfg.hidden_units,
        activations=cfg.activations,
        drop_out_rate=cfg.dropout,
        flag_drop_out=cfg.flag_drop_out,
        flag_only_output_layer=cfg.flag_only_output_layer
    )
    
    mask_layer = MaskLayer(append=True)
    print('Pretraining predictor')
    print('-'*8)
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
    pretrain.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        val_loss_fn=Accuracy(task='multiclass', num_classes=d_out),
        val_loss_mode='max',
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
        tag="eval_accuracy")
    
    pretrained_model_artifact = wandb.Artifact(
        name=f"pretrain_gadgil2023-{cfg.dataset_artifact_name.split(':')[0]}",
        type="pretrained_model",
    )
    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        torch.save({
            'predictor_state_dict': pretrain.model.state_dict(),
            'architecture': {
                'd_in': d_in,
                'd_out': d_out,
                'predictor_hidden_layers': cfg.hidden_units,
                'dropout': cfg.dropout,
            }}, tmp_path / 'model.pt')
    
    pretrained_model_artifact.add_file(str(tmp_path / 'model.pt'))

    run.log_artifact(
        pretrained_model_artifact,
        aliases=[*cfg.output_artifact_aliases, datetime.now().strftime("%b%d")]
    )
    run.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
