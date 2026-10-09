"""
The workflow config loader: typed settings, rejected at load time when wrong.

Each config mistake below once surfaced late, as a `KeyError` while
Snakemake built the DAG, or never, as a silently different benchmark.
"""

import warnings
from pathlib import Path

import pytest
import yaml

from afabench.core.output_layout import OutputLayout
from afabench.core.workflow_settings import load_config
from afabench.release.workflow_config import resolve_workflow_config

REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize("profile", ["all", "kdd26"])
def test_shipped_configs_load(profile: str) -> None:
    profile_file = REPO_ROOT / "workflow/profiles/config" / profile
    configfiles = yaml.safe_load((profile_file / "config.yaml").read_text())[
        "configfile"
    ]
    record = resolve_workflow_config(
        profile=None,
        configfiles=[REPO_ROOT / path for path in configfiles],
        overrides=[],
    )

    settings = load_config(record.merged)

    assert settings.methods


def test_load_config_uses_pipeline_defaults() -> None:
    settings = load_config(_config())

    assert settings.eval_dataset_split == "test"
    assert settings.dataset_realization_indices == [0, 1, 2, 3, 4]
    assert settings.smoke_test is False
    assert settings.use_wandb is True
    assert settings.initializer == "cold"
    assert settings.output_root == "output/production"


def test_smoke_test_writes_under_its_own_output_root() -> None:
    settings = load_config(_config() | {"smoke_test": True})

    assert settings.output_root == "output/smoke"


@pytest.mark.parametrize("smoke_test", [False, True])
def test_an_explicit_output_root_wins(*, smoke_test: bool) -> None:
    settings = load_config(
        _config() | {"smoke_test": smoke_test, "output_root": "elsewhere"}
    )

    assert settings.output_root == "elsewhere"


@pytest.mark.parametrize(
    ("smoke_test", "output_root"),
    [
        (True, "output/production"),
        (True, "output/production/"),
        (True, str(Path("output/production").resolve())),
        (True, "output/production/nested"),
        (True, "output"),
        (False, "output/smoke"),
        (False, "output/smoke/nested"),
        (False, "output"),
    ],
)
def test_an_output_root_cannot_overlap_the_other_kind_of_run(
    *, smoke_test: bool, output_root: str
) -> None:
    config = _config() | {"smoke_test": smoke_test, "output_root": output_root}

    with pytest.raises(ValueError, match=r"output_root"):
        load_config(config)


@pytest.mark.parametrize(
    ("smoke_test", "output_root"),
    [
        (True, "output/smoke/nested"),
        (False, "output/production/nested"),
        (True, "/scratch/afabench/output"),
        (False, "output_elsewhere"),
    ],
)
def test_an_output_root_apart_from_the_other_kind_of_run_is_allowed(
    *, smoke_test: bool, output_root: str
) -> None:
    config = _config() | {"smoke_test": smoke_test, "output_root": output_root}

    assert load_config(config).output_root == output_root


def test_aaco_eval_batch_size_is_pinned_per_dataset() -> None:
    """AACO batches its acquisition, so all.yaml pins its eval batch size."""
    method_options = yaml.safe_load(
        (REPO_ROOT / "workflow/conf/method_options/all.yaml").read_text()
    )["method_options"]
    config = _config(
        method_options=method_options,
        methods=["aaco", "aaco_nn"],
        datasets=["cube", "mnist", "fashion_mnist", "synthetic_mnist"],
        pretrain_mapping={"aaco": {"pretrain_script_name": "aaco"}},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settings = load_config(config)

    # Image datasets get the smaller pinned size, every other dataset the
    # default. The loader keeps the `default` key alongside the datasets.
    expected = {
        "default": 128,
        "cube": 128,
        "mnist": 32,
        "fashion_mnist": 32,
        "synthetic_mnist": 32,
        "synthetic_mnist_without_noise": 32,
    }
    assert settings.eval_batch_sizes["aaco"] == expected
    assert settings.eval_batch_sizes["aaco_nn"] == expected


def test_missing_eval_batch_size_warns_and_falls_back_to_one() -> None:
    config = _config(
        method_options={"unbatched": {"train_script_name": "unbatched"}},
        methods=["unbatched"],
        datasets=["cube", "mnist"],
    )

    with pytest.warns(UserWarning, match=r"unbatched.*eval_batch_size"):
        settings = load_config(config)

    assert settings.eval_batch_sizes["unbatched"] == {"cube": 1, "mnist": 1}


def test_eval_batch_size_mapping_without_default_is_rejected() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": {"cube": 8},
            }
        }
    )

    with pytest.raises(ValueError, match=r"'alpha'.*eval_batch_size.*default"):
        load_config(config)


@pytest.mark.parametrize(
    "typo",
    [
        "pretrained_model_nme",
        "hard_budget_ignored_dataset",
        "soft_budget_ignored_dataset",
        "eval_batch_sise",
    ],
)
def test_misspelt_method_option_is_rejected(typo: str) -> None:
    config = _config(
        method_options={
            "alpha": {"train_script_name": "alpha", typo: "anything"}
        }
    )

    with pytest.raises(ValueError, match=rf"'alpha'.*{typo}"):
        load_config(config)


# A config passed through JSON, as a release config is, has string keys.
@pytest.mark.parametrize("eval_budget_key", [1, "1"])
def test_eval_to_train_hard_budget_mapping_sets_the_train_hard_budget(
    eval_budget_key: int | str,
) -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "eval_to_train_hard_budget_mapping": {
                    "cube": {eval_budget_key: 3}
                },
            }
        }
    )

    settings = load_config(config)

    assert (3, 1, "null", "null") in settings.budget_params["alpha"]["cube"]


def test_method_without_train_script_name_is_rejected() -> None:
    config = _config(method_options={"alpha": {"eval_batch_size": 1}})

    with pytest.raises(ValueError, match=r"'alpha'.*train_script_name"):
        load_config(config)


def test_pretrained_model_missing_from_pretrain_mapping_is_rejected() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "pretrained_model_name": "shard",
                "eval_batch_size": 1,
            }
        },
        pretrain_mapping={"shared": {"pretrain_script_name": "shared"}},
    )

    with pytest.raises(
        ValueError, match=r"'alpha'.*'shard'.*pretrain_mapping"
    ):
        load_config(config)


@pytest.mark.parametrize(
    "model_config",
    [
        {"pretrain_params": []},
        {"pretrain_script_name": "shared", "pretrain_parms": []},
    ],
)
def test_malformed_pretrain_mapping_entry_is_rejected(
    model_config: dict[str, object],
) -> None:
    config = _config(pretrain_mapping={"shared": model_config})

    with pytest.raises(ValueError, match=r"pretrain_mapping\['shared'\]"):
        load_config(config)


def test_method_missing_from_method_options_is_rejected() -> None:
    config = _config(methods=["alpha", "beta"])

    with pytest.raises(ValueError, match=r"'beta'.*method_options"):
        load_config(config)


def test_method_missing_from_soft_budget_params_is_rejected() -> None:
    config = _config()
    config["soft_budget_params"] = {}

    with pytest.raises(ValueError, match=r"'alpha'.*soft_budget_params"):
        load_config(config)


def test_reference_method_missing_from_soft_budget_params_is_rejected() -> (
    None
):
    config = _config(
        method_options={
            name: {"train_script_name": name, "eval_batch_size": 1}
            for name in ["alpha", "beta"]
        },
        methods=["beta"],
    )
    config["reference_methods"] = ["alpha"]

    with pytest.raises(ValueError, match=r"'alpha'.*soft_budget_params"):
        load_config(config)


@pytest.mark.parametrize(
    "option", ["hard_budget_ignored_datasets", "soft_budget_ignored_datasets"]
)
def test_unknown_ignored_dataset_key_is_rejected(option: str) -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                option: ["imagenete"],
            }
        }
    )

    with pytest.raises(ValueError, match=rf"'alpha'.*{option}.*'imagenete'"):
        load_config(config)


def test_ignored_dataset_need_not_be_in_this_run() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "hard_budget_ignored_datasets": ["imagenette"],
            }
        },
        datasets=["cube"],
    )

    settings = load_config(config)

    assert settings.budget_params["alpha"]["cube"] == [
        (1, 1, "null", "null"),
        ("null", "null", 0.1, 0.1),
    ]


def test_misspelt_classifier_key_is_rejected() -> None:
    config = _config()
    config["classifier_names"] = {
        "default": {"script_name": "masked_mlp_classifier", "script_parms": []}
    }

    with pytest.raises(ValueError, match=r"'default'.*script_parms"):
        load_config(config)


def _compared_methods_config() -> dict[str, object]:
    """
    Compare two methods and a reference method.

    Alpha has no pretraining stage and no built-in classifier, beta has
    both, and gamma, the reference method, has a pretraining stage.
    """
    config = _config(
        method_options={
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 1},
            "beta": {
                "train_script_name": "beta",
                "pretrained_model_name": "shared",
                "classifier": {"script_name": "beta_classifier"},
                "eval_batch_size": 1,
            },
            "gamma": {
                "train_script_name": "gamma",
                "pretrained_model_name": "shared",
            },
        },
        methods=["alpha", "beta"],
        pretrain_mapping={"shared": {"pretrain_script_name": "shared"}},
    )
    config["reference_methods"] = ["gamma"]
    config["soft_budget_params"] = {
        method: {"default": [[0.1, 0.2]]}
        for method in ["alpha", "beta", "gamma"]
    }
    return config


@pytest.mark.parametrize(
    ("method", "pretrain_folder"),
    [
        ("alpha", "NO_PRETRAIN"),
        ("beta", "pretrain_seed-2"),
        ("gamma", "pretrain_seed-2"),
    ],
)
def test_evaluation_run_is_seeded_with_its_dataset_realization_index(
    method: str, pretrain_folder: str
) -> None:
    settings = load_config(_compared_methods_config())

    run = settings.evaluation_run(
        method=method,
        dataset="cube",
        dataset_realization_index=2,
        budget_combination=("null", "null", 0.1, 0.2),
    )

    assert run.training.pretrain_folder == pretrain_folder
    assert run.training.train_seed == 2
    assert run.eval_seed == 2
    assert (run.training.train_hard_budget, run.eval_hard_budget) == (
        "null",
        "null",
    )
    assert (
        run.training.train_soft_budget_param,
        run.eval_soft_budget_param,
    ) == (0.1, 0.2)


@pytest.mark.parametrize(
    ("method", "bundle"),
    [
        (
            "alpha",
            "output/trained_classifiers/initializer-cold/"
            "dataset-cube+realization_index-2.bundle",
        ),
        (
            "beta",
            "output/trained_classifiers/initializer-cold/"
            "method-beta+dataset-cube+realization_index-2.bundle",
        ),
    ],
)
def test_classifier_bundle_is_the_built_in_one_if_the_method_has_one(
    method: str, bundle: str
) -> None:
    settings = load_config(_compared_methods_config())
    layout = OutputLayout(root="output", initializer="cold", eval_split="test")

    assert (
        settings.classifier_bundle(
            layout, method=method, dataset="cube", dataset_realization_index=2
        )
        == bundle
    )


def _config(
    *,
    method_options: dict[str, object] | None = None,
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
    pretrain_mapping: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a minimal valid config, with the given parts replaced."""
    if method_options is None:
        method_options = {
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 1}
        }
    if methods is None:
        methods = list(method_options)
    return {
        "pretrain_mapping": pretrain_mapping or {},
        "method_options": method_options,
        "methods": methods,
        "datasets": datasets or ["cube"],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [1]},
        "soft_budget_params": {
            method: {"default": [[0.1, 0.1]]} for method in methods
        },
        "classifier_names": {"default": "masked_mlp_classifier"},
    }
