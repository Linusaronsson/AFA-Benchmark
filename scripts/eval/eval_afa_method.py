from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from afabench.common.bundle import load_bundle
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers import AFAContextUnmasker, CubeNMARUnmasker
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import set_seed
from afabench.eval.cube_nm_ar import (
    augment_cube_nm_ar_eval_df,
    summarize_cube_nm_ar_episodes,
)
from afabench.eval.eval import eval_afa_method
from afabench.eval.stop_shielding import (
    DualizedStopWrapper,
    StopShieldWrapper,
)

if TYPE_CHECKING:
    import pandas as pd

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

log = logging.getLogger(__name__)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(float).mean())


def _get_cube_nm_ar_eval_summary(
    df_eval: pd.DataFrame,
) -> dict[str, float | int] | None:
    episode_df = summarize_cube_nm_ar_episodes(df_eval)
    if episode_df is None or episode_df.empty:
        return None

    risky_mask = episode_df["cube_nm_ar_is_risky_context"].astype(bool)
    risky_episode_df = episode_df[risky_mask]

    return {
        "episode_count": len(episode_df),
        "risky_episode_count": len(risky_episode_df),
        "risky_episode_rate": _safe_mean(risky_mask),
        "risky_accuracy": _safe_mean(risky_episode_df["correct"]),
        "risky_rescue_rate": _safe_mean(risky_episode_df["rescue_acquired"]),
        "risky_relevant_block_acquisition_rate": _safe_mean(
            risky_episode_df["relevant_block_acquired"]
        ),
        "risky_unsafe_stop_rate": _safe_mean(risky_episode_df["unsafe_stop"]),
    }


def _log_cube_nm_ar_eval_summary(
    df_eval: pd.DataFrame,
) -> dict[str, float | int] | None:
    risk_summary = _get_cube_nm_ar_eval_summary(df_eval)
    if risk_summary is None:
        return None

    log.info(
        "CUBE-NM-AR risk: risky_episode_rate=%.4f, risky_accuracy=%.4f, "
        "risky_rescue_rate=%.4f, risky_unsafe_stop_rate=%.4f.",
        risk_summary["risky_episode_rate"],
        risk_summary["risky_accuracy"],
        risk_summary["risky_rescue_rate"],
        risk_summary["risky_unsafe_stop_rate"],
    )
    return risk_summary


def _resolve_shield_predictor(
    *,
    afa_method: AFAMethod,
    external_classifier: AFAClassifier | None,
) -> tuple[Any, str]:
    if external_classifier is not None:
        return external_classifier.__call__, type(external_classifier).__name__
    if afa_method.has_builtin_classifier:
        return afa_method.predict, type(afa_method).__name__
    msg = (
        "Stop-control evaluation requires either an external classifier or "
        "a method with a builtin classifier."
    )
    raise ValueError(msg)


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

    # Feature-space -> selection-space conversion for grouped selections.
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

    if isinstance(unmasker, CubeNMARUnmasker):
        excluded_start = unmasker.n_contexts
        expected_n_selections = 1 + (n_features - excluded_start)
        if n_selection_choices != expected_n_selections:
            msg = (
                "Unexpected selection-space size for CubeNMARUnmasker. "
                f"Expected {expected_n_selections}, got {n_selection_choices}."
            )
            raise ValueError(msg)

        flat_forbidden = forbidden_mask.reshape(-1, n_features)
        sel_forbidden = torch.zeros(
            (flat_forbidden.shape[0], n_selection_choices),
            dtype=torch.bool,
            device=forbidden_mask.device,
        )
        sel_forbidden[:, 0] = flat_forbidden[:, : unmasker.n_contexts].any(
            dim=1
        )
        sel_forbidden[:, 1:] = flat_forbidden[:, excluded_start:]
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
def main(cfg: EvalConfig) -> None:  # noqa: C901, PLR0912, PLR0915
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
        if hasattr(afa_method, "force_acquisition"):
            afa_method.force_acquisition = False
            log.info("Disabled force_acquisition for soft-budget evaluation.")

    selection_costs = unmasker.get_selection_costs(
        feature_costs=dataset.get_feature_acquisition_costs()
    )
    n_selection_choices = unmasker.get_n_selections(
        feature_shape=dataset.feature_shape
    )

    stop_shield = None
    afa_action_fn = afa_method.act
    if cfg.stop_shield_delta is not None and cfg.dual_lambda is not None:
        msg = (
            "Use either stop_shield_delta or dual_lambda during evaluation, "
            "not both."
        )
        raise ValueError(msg)

    if cfg.stop_shield_delta is not None:
        shield_predict_fn, predictor_name = _resolve_shield_predictor(
            afa_method=afa_method,
            external_classifier=external_classifier,
        )

        stop_shield = StopShieldWrapper(
            afa_method=afa_method,
            afa_predict_fn=shield_predict_fn,
            risk_threshold=cfg.stop_shield_delta,
            predictor_name=predictor_name,
        )
        afa_action_fn = stop_shield
        log.info(
            "Enabled stop shield with delta=%.4f using predictor %s.",
            cfg.stop_shield_delta,
            predictor_name,
        )
    elif cfg.dual_lambda is not None:
        shield_predict_fn, predictor_name = _resolve_shield_predictor(
            afa_method=afa_method,
            external_classifier=external_classifier,
        )

        stop_shield = DualizedStopWrapper(
            afa_method=afa_method,
            afa_predict_fn=shield_predict_fn,
            afa_unmask_fn=unmasker.unmask,
            predictor_name=predictor_name,
            selection_costs=selection_costs,
            dual_lambda=cfg.dual_lambda,
        )
        afa_action_fn = stop_shield
        log.info(
            "Enabled dualized stop wrapper with lambda=%.4f using predictor %s.",
            cfg.dual_lambda,
            predictor_name,
        )

    if cfg.hard_budget is not None:
        hard_budget_str = f"hard budget {cfg.hard_budget}"
    else:
        hard_budget_str = "no hard budget"
    log.info(
        "Starting evaluation with batch size %s and %s.",
        cfg.batch_size,
        hard_budget_str,
    )
    log.info(
        "Selection costs summary: n=%d, min=%.4f, max=%.4f, mean=%.4f.",
        selection_costs.numel(),
        selection_costs.min().item(),
        selection_costs.max().item(),
        selection_costs.mean().item(),
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
        afa_action_fn=afa_action_fn,
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
    df_eval = augment_cube_nm_ar_eval_df(df_eval, dataset)

    csv_path = Path(cfg.save_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    risk_summary = _log_cube_nm_ar_eval_summary(df_eval)
    if risk_summary is not None:
        risk_path = csv_path.parent / "cube_nm_ar_risk_summary.json"
        risk_path.write_text(
            json.dumps(risk_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log.info("Saved CUBE-NM-AR risk summary to %s.", risk_path)

    if stop_shield is not None:
        shield_summary = stop_shield.get_summary()
        shield_path = csv_path.parent / "stop_shield_summary.json"
        shield_path.write_text(
            json.dumps(shield_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log.info("Saved stop-shield summary to %s.", shield_path)

    helper_columns = [
        column for column in df_eval.columns if column.startswith("cube_")
    ]
    if helper_columns:
        df_eval = df_eval.drop(columns=helper_columns)

    # Save CSV directly
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
