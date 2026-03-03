import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from afabench.common.bundle import load_bundle
from afabench.common.config_classes import (
    EvalConfig,
    InitializerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
    FeatureMask,
    SelectionMask,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers import AFAContextUnmasker
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    set_seed,
)
from afabench.eval.eval import eval_afa_method

log = logging.getLogger(__name__)


def _adapt_forbidden_mask_to_selection_space(
    forbidden_mask: SelectionMask,
    *,
    n_selection_choices: int,
    feature_shape: torch.Size,
    unmasker: AFAUnmasker,
) -> SelectionMask:
    """
    Ensure forbidden mask is expressed in selection space.

    MissingnessInitializer returns feature-level masks by default. For grouped
    unmaskers (e.g. AFAContextUnmasker), convert to selection-level masks.
    """
    if forbidden_mask.shape[-1] == n_selection_choices:
        return forbidden_mask

    n_features = int(torch.prod(torch.tensor(feature_shape)).item())
    if forbidden_mask.shape[-1] != n_features:
        msg = (
            "Initializer forbidden mask has incompatible shape. "
            f"Expected trailing dim {n_selection_choices} (selection space) or "
            f"{n_features} (feature space), got {forbidden_mask.shape[-1]}."
        )
        raise ValueError(msg)

    # Feature-space -> selection-space conversion for context grouped selections.
    if isinstance(unmasker, AFAContextUnmasker):
        n_contexts = unmasker.n_contexts
        expected_n_selections = 1 + (n_features - n_contexts)
        if n_selection_choices != expected_n_selections:
            msg = (
                "Unexpected selection-space size for AFAContextUnmasker. "
                f"Expected {expected_n_selections}, got {n_selection_choices}."
            )
            raise ValueError(msg)

        flat_forbidden = forbidden_mask.reshape(-1, n_features)
        sel_forbidden = torch.zeros(
            (flat_forbidden.shape[0], n_selection_choices),
            dtype=torch.bool,
            device=forbidden_mask.device,
        )
        # Selection 0 corresponds to acquiring all context features at once.
        sel_forbidden[:, 0] = flat_forbidden[:, :n_contexts].any(dim=1)
        sel_forbidden[:, 1:] = flat_forbidden[:, n_contexts:]
        batch_shape = forbidden_mask.shape[:-1]
        return sel_forbidden.reshape(*batch_shape, n_selection_choices)

    # For non-grouped unmaskers, this mismatch is usually all-false masks from
    # initializers that only define feature-level forbidden masks.
    if forbidden_mask.any():
        msg = (
            "Cannot convert feature-level forbidden mask to selection space for "
            f"unmasker {type(unmasker).__name__}."
        )
        raise ValueError(msg)

    return torch.zeros(
        (*forbidden_mask.shape[:-1], n_selection_choices),
        dtype=torch.bool,
        device=forbidden_mask.device,
    )


def load(
    method_bundle_path: Path,
    unmasker_cfg: UnmaskerConfig,
    initializer_cfg: InitializerConfig,
    dataset_bundle_path: Path,
    classifier_bundle_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[
    AFAMethod,
    AFAUnmasker,
    AFAInitializer,
    AFADataset,
    AFAClassifier | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
]:
    # Load method
    device = torch.device("cpu") if device is None else device
    method, method_manifest = load_bundle(
        method_bundle_path,
        device=device,
    )
    method = cast("AFAMethod", cast("object", method))
    log.info(f"Loaded AFA method from {method_bundle_path}")

    # Load unmasker
    unmasker: AFAUnmasker = get_afa_unmasker_from_config(unmasker_cfg)
    log.info(f"Loaded {unmasker_cfg.class_name}")

    # Load initializer
    initializer: AFAInitializer = get_afa_initializer_from_config(
        initializer_cfg
    )
    log.info(f"Loaded {initializer_cfg.class_name} initializer")

    # Load dataset
    dataset, dataset_manifest = load_bundle(dataset_bundle_path)
    dataset = cast("AFADataset", cast("object", dataset))
    log.info(f"Loaded dataset from {dataset_bundle_path}")

    # Load external classifier if specified
    if classifier_bundle_path is not None:
        classifier, classifier_manifest = load_bundle(
            classifier_bundle_path,
            device=device,
        )
        classifier = cast("AFAClassifier", cast("object", classifier))
        log.info(f"Loaded external classifier from {classifier_bundle_path}.")
        classifier_metadata = classifier_manifest["metadata"]
    else:
        classifier = None
        classifier_metadata = None
        log.info("No external classifier provided; using builtin classifier.")

    return (
        method,
        unmasker,
        initializer,
        dataset,
        classifier,
        method_manifest["metadata"],
        dataset_manifest["metadata"],
        classifier_metadata,
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/eval",
    config_name="config",
)
def main(cfg: EvalConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = wandb.init(
            job_type="evaluation",
            config=cast(
                "dict[str, Any]",
                OmegaConf.to_container(cfg, resolve=True),
            ),
            dir="extra/logs/wandb",
        )
        log.info(f"W&B run initialized: {run.name} ({run.id})")
        log.info(f"W&B run URL: {run.url}")
    else:
        run = None

    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.eval_only_n_samples = 10
        cfg.batch_size = 2

    # Load everything
    (
        afa_method,
        unmasker,
        initializer,
        dataset,
        external_classifier,
        _method_metadata,
        _dataset_metadata,
        _external_classifier_metadata,
    ) = load(
        method_bundle_path=Path(cfg.method_bundle_path),
        unmasker_cfg=cfg.unmasker,
        initializer_cfg=cfg.initializer,
        dataset_bundle_path=Path(cfg.dataset_bundle_path),
        classifier_bundle_path=(
            Path(cfg.classifier_bundle_path)
            if cfg.classifier_bundle_path is not None
            else None
        ),
        device=torch.device(cfg.device),
    )

    # Set the seed of everything
    afa_method.set_seed(cfg.seed)
    unmasker.set_seed(cfg.seed)
    initializer.set_seed(cfg.seed)

    # Some methods require a soft budget parameter set during evaluation instead of training
    if cfg.soft_budget_param is not None:
        afa_method.set_cost_param(cost_param=cfg.soft_budget_param)

    if cfg.hard_budget is not None:
        hard_budget_str = f"hard budget {cfg.hard_budget}"
    else:
        hard_budget_str = "no hard budget"
    log.info(
        "Starting evaluation with batch size %s and %s.",
        cfg.batch_size,
        hard_budget_str,
    )
    selection_costs = unmasker.get_selection_costs(
        feature_costs=dataset.get_feature_acquisition_costs()
    )
    log.info(
        "Selection costs summary: n=%d, min=%.4f, max=%.4f, mean=%.4f.",
        selection_costs.numel(),
        selection_costs.min().item(),
        selection_costs.max().item(),
        selection_costs.mean().item(),
    )

    n_selection_choices = unmasker.get_n_selections(
        feature_shape=dataset.feature_shape
    )

    forbidden_mask_fn = None
    maybe_forbidden_mask_fn = getattr(
        initializer, "get_forbidden_selection_mask", None
    )
    if callable(maybe_forbidden_mask_fn):

        def forbidden_mask_fn(
            observed_mask: FeatureMask,
            feature_shape: torch.Size,
        ) -> SelectionMask:
            raw_mask = maybe_forbidden_mask_fn(observed_mask, feature_shape)
            return _adapt_forbidden_mask_to_selection_space(
                raw_mask,
                n_selection_choices=n_selection_choices,
                feature_shape=feature_shape,
                unmasker=unmasker,
            )

        log.info(
            "Using initializer-provided forbidden selection mask function."
        )

    # Enable CMI logging for DIME if requested.
    if cfg.log_cmi and hasattr(afa_method, "enable_cmi_logging"):
        afa_method.enable_cmi_logging()
        log.info("CMI logging enabled for %s.", type(afa_method).__name__)

    df_eval = eval_afa_method(
        afa_action_fn=afa_method.act,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=n_selection_choices,
        afa_initialize_fn=initializer.initialize,
        dataset=dataset,
        external_afa_predict_fn=external_classifier.__call__
        if external_classifier is not None
        else None,
        builtin_afa_predict_fn=afa_method.predict
        if afa_method.has_builtin_classifier
        else None,
        only_n_samples=cfg.eval_only_n_samples,
        device=torch.device(cfg.device),
        selection_budget=cfg.hard_budget,
        batch_size=cfg.batch_size,
        selection_costs=selection_costs.tolist(),
        forbidden_mask_fn=forbidden_mask_fn,
    )

    # Add eval_seed and eval_hard_budget to dataframe
    df_eval["eval_seed"] = cfg.seed
    df_eval["eval_hard_budget"] = cfg.hard_budget

    # Save CSV directly
    csv_path = Path(cfg.save_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Use explicit null strings to avoid missing values in the pipeline.
    df_eval.to_csv(csv_path, index=False, na_rep="null")
    log.info(f"Saved evaluation data to CSV at: {csv_path}")

    log.info(f"Evaluation results saved to: {cfg.save_path}")

    # Save CMI log if available.
    cmi_log = getattr(afa_method, "get_cmi_log", None)
    if callable(cmi_log) and cmi_log():
        import pickle

        entries = cmi_log()
        cmi_log_path = csv_path.parent / "cmi_log.pkl"
        with cmi_log_path.open("wb") as f:
            pickle.dump(entries, f)
        log.info(
            "Saved CMI log (%d entries) to %s.",
            len(entries),
            cmi_log_path,
        )
        afa_method.clear_cmi_log()

    if run:
        run.finish()


if __name__ == "__main__":
    main()
