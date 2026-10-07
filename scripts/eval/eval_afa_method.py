import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast, final

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from pandera.typing import DataFrame

from afabench.components.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.components.unmaskers.utils import (
    get_afa_unmasker_from_config,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import SupportsForcedAcquisition
from afabench.core.utils import (
    set_seed,
)
from afabench.evaluation.config import EvalConfig
from afabench.evaluation.eval import eval_afa_method
from afabench.evaluation.schemas import SavedEvaluationSchema
from afabench.fit.smoke_test import eval_settings

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from afabench.core.types import (
        AFAClassifier,
        AFADataset,
        AFAInitializer,
        AFAMethod,
        AFAUnmasker,
    )

log = logging.getLogger(__name__)


@final
class AFAEvaluator:
    def __init__(self, cfg: EvalConfig):
        self._cfg = cfg
        self._wandb_run: Run | None = None
        self._method: AFAMethod | None = None
        self._unmasker: AFAUnmasker | None = None
        self._initializer: AFAInitializer | None = None
        self._dataset: AFADataset | None = None
        self._external_classifier: AFAClassifier | None = None
        self._n_selection_choices: int | None = None
        self._selection_costs: torch.Tensor | None = None
        self._df_eval: DataFrame[SavedEvaluationSchema] | None = None

    def run(self) -> None:
        self._init_wandb()
        self._smoke_test_override()
        self._load()
        self._set_seeds()
        self._set_soft_budget()
        self._set_hard_budget()
        self._set_selection_info()
        self._exec()
        self._save()

    def _load(
        self,
    ) -> None:
        # Load method
        device = torch.device(self._cfg.device)
        method, _ = load_bundle(
            Path(self._cfg.method_bundle_path),
            device=device,
        )
        self._method = cast("AFAMethod", cast("object", method))
        log.info(f"Loaded AFA method from {self._cfg.method_bundle_path}")

        # Load unmasker
        self._unmasker = get_afa_unmasker_from_config(self._cfg.unmasker)
        log.info(f"Loaded {self._cfg.unmasker.class_name}")

        # Load initializer
        self._initializer = get_afa_initializer_from_config(
            self._cfg.initializer
        )
        log.info(f"Loaded {self._cfg.initializer.class_name} initializer")

        # Load dataset
        dataset, _ = load_bundle(Path(self._cfg.dataset_bundle_path))
        self._dataset = cast("AFADataset", cast("object", dataset))
        log.info(f"Loaded dataset from {self._cfg.dataset_bundle_path}")

        # Load external classifier if specified
        if self._cfg.classifier_bundle_path is not None:
            classifier, _ = load_bundle(
                Path(self._cfg.classifier_bundle_path),
                device=device,
            )
            self._external_classifier = cast(
                "AFAClassifier", cast("object", classifier)
            )
            log.info(
                f"Loaded external classifier from {self._cfg.classifier_bundle_path}."
            )
        else:
            self._external_classifier = None
            log.info(
                "No external classifier provided; using builtin classifier."
            )

    def _init_wandb(self) -> None:
        if self._cfg.use_wandb:
            self._wandb_run = wandb.init(
                job_type="evaluation",
                config=asdict(self._cfg),
                dir="extra/logs/wandb",
            )
            log.info(
                f"W&B run initialized: {self._wandb_run.name} ({self._wandb_run.id})"
            )
            log.info(f"W&B run URL: {self._wandb_run.url}")
        else:
            self._wandb_run = None

    def _smoke_test_override(self) -> None:
        if self._cfg.smoke_test:
            default_n_samples = (
                self._cfg.eval_only_n_samples
                if self._cfg.eval_only_n_samples is not None
                else self._cfg.batch_size * 2
            )
            self._cfg.eval_only_n_samples, self._cfg.batch_size = (
                eval_settings(
                    smoke_test=True,
                    default_n_samples=default_n_samples,
                    default_batch_size=self._cfg.batch_size,
                )
            )

    def _set_seeds(self) -> None:
        # Set the seed of everything
        assert self._method is not None
        self._method.set_seed(self._cfg.seed)
        assert self._unmasker is not None
        self._unmasker.set_seed(self._cfg.seed)
        assert self._initializer is not None
        self._initializer.set_seed(self._cfg.seed)

    def _set_soft_budget(self) -> None:
        assert self._method is not None

        # Some methods require a soft budget parameter set during evaluation instead of training
        if self._cfg.soft_budget_param is not None:
            self._method.set_cost_param(cost_param=self._cfg.soft_budget_param)

    def _set_hard_budget(self) -> None:
        if self._cfg.hard_budget is not None:
            # Optional optimisation: preserve the method's preferred selection
            # rather than falling back to the first available selection.
            # Evaluation enforces acquisition even if the method still stops.
            if isinstance(self._method, SupportsForcedAcquisition):
                self._method.force_acquisition = True
            log.info(
                "Enabled evaluator-enforced acquisition for hard-budget evaluation."
            )

    def _set_selection_info(self) -> None:
        assert self._unmasker is not None
        assert self._dataset is not None

        self._selection_costs = self._unmasker.get_selection_costs(
            feature_costs=self._dataset.get_feature_acquisition_costs()
        )
        log.info(
            "Selection costs summary: n=%d, min=%.4f, max=%.4f, mean=%.4f.",
            self._selection_costs.numel(),
            self._selection_costs.min().item(),
            self._selection_costs.max().item(),
            self._selection_costs.mean().item(),
        )
        self._n_selection_choices = self._unmasker.get_n_selections(
            feature_shape=self._dataset.feature_shape
        )

    def _exec(self) -> None:
        assert self._method is not None
        assert self._unmasker is not None
        assert self._n_selection_choices is not None
        assert self._initializer is not None
        assert self._dataset is not None
        assert self._selection_costs is not None

        hard_budget_str = (
            f"hard budget {self._cfg.hard_budget}"
            if self._cfg.hard_budget is not None
            else "no hard budget"
        )
        log.info(
            "Starting evaluation with batch size %s and hard budget %s.",
            self._cfg.batch_size,
            hard_budget_str,
        )

        df_eval = eval_afa_method(
            afa_action_fn=self._method.act,
            afa_unmask_fn=self._unmasker.unmask,
            n_selection_choices=self._n_selection_choices,
            afa_initialize_fn=self._initializer.initialize,
            dataset=self._dataset,
            external_afa_predict_fn=self._external_classifier.__call__
            if self._external_classifier is not None
            else None,
            builtin_afa_predict_fn=self._method.predict
            if self._method.has_builtin_classifier
            else None,
            only_n_samples=self._cfg.eval_only_n_samples,
            device=torch.device(self._cfg.device),
            selection_budget=self._cfg.hard_budget,
            batch_size=self._cfg.batch_size,
            selection_costs=self._selection_costs.tolist(),
            seed=self._cfg.seed,
            force_acquisition=self._cfg.hard_budget is not None,
        )

        # Add eval_seed and eval_hard_budget to dataframe
        df_eval["eval_seed"] = self._cfg.seed
        df_eval["eval_hard_budget"] = self._cfg.hard_budget
        self._df_eval = DataFrame[SavedEvaluationSchema](df_eval)

    def _save(self) -> None:
        assert self._df_eval is not None
        save_path = Path(self._cfg.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        SavedEvaluationSchema.validate(self._df_eval).to_parquet(
            save_path, index=False
        )
        log.info(f"Saved evaluation data to Parquet at: {save_path}")

        log.info(f"Evaluation results saved to: {self._cfg.save_path}")

        if self._wandb_run:
            self._wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/eval",
    config_name="config",
)
def main(cfg: EvalConfig) -> None:
    cfg = cast("EvalConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    evaluator = AFAEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
