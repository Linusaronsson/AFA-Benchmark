"""
Temporary (#55): the port keeps every resolved RL method config unchanged.

Composes each root config of jafa, odin and ol (training and pretraining)
for every dataset key, once from `extra/conf` at the commit before the port
and once from the working tree, and compares the resolved configs. The port
removes the `mdp.hard_budget` alias on purpose (the trainer reads the
contract's `hard_budget`), and the current `AFAMDPConfig` schema rejects
that key, so the alias block is stripped from the base tree's root configs.
"""

import subprocess
import tarfile
import warnings
from io import BytesIO
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from afabench.components.methods.rl.jafa.config import (
    JAFAPretrainConfig,
    JAFATrainConfig,
)
from afabench.components.methods.rl.odin.config import ODINTrainConfig
from afabench.components.methods.rl.ol.config import (
    OLPretrainConfig,
    OLTrainConfig,
)

BASE_COMMIT = "955b4446"
REPO_ROOT = Path(__file__).parents[2]
DATASET_KEYS = sorted(
    p.stem for p in (REPO_ROOT / "extra/conf/dataset_key").glob("*.yaml")
)
PRETRAINING_OVERRIDES = [
    "train_dataset_bundle_path=train.bundle",
    "val_dataset_bundle_path=val.bundle",
    "classifier_bundle_path=classifier.bundle",
    "save_path=out.bundle",
    "initializer=cold",
    "unmasker=direct",
    "device=cpu",
    "seed=0",
]
TRAINING_OVERRIDES = [
    *PRETRAINING_OVERRIDES,
    "pretrained_model_bundle_path=model.bundle",
    "hard_budget=5",
    "soft_budget_param=null",
]
# Importing the config classes registers the root configs' schemas.
CONFIG_CLASSES = (
    JAFAPretrainConfig,
    JAFATrainConfig,
    ODINTrainConfig,
    OLPretrainConfig,
    OLTrainConfig,
)
METHOD_CONFIGS = {
    "train_method/jafa": TRAINING_OVERRIDES,
    "train_method/odin": TRAINING_OVERRIDES,
    "train_method/ol": TRAINING_OVERRIDES,
    "pretrain_model/jafa": PRETRAINING_OVERRIDES,
    "pretrain_model/ol": PRETRAINING_OVERRIDES,
}
HARD_BUDGET_ALIAS_BLOCK = (
    "# The script copies the contract's hard_budget onto mdp.hard_budget.\n"
    "mdp:\n"
    "  hard_budget: ???\n"
)


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    archive = subprocess.run(
        ["git", "archive", BASE_COMMIT, "extra/conf"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    root = tmp_path_factory.mktemp("base_tree")
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(root, filter="data")
    for method_config_name in METHOD_CONFIGS:
        if not method_config_name.startswith("train_method/"):
            continue
        root_config = (
            root / "extra/conf/scripts" / method_config_name / "config.yaml"
        )
        content = root_config.read_text()
        assert HARD_BUDGET_ALIAS_BLOCK in content, root_config
        root_config.write_text(content.replace(HARD_BUDGET_ALIAS_BLOCK, ""))
    return root


def _resolved_config(
    tree: Path, method_config_name: str, dataset_key: str
) -> dict[str, object]:
    with (
        initialize_config_dir(
            config_dir=str(tree / "extra/conf/scripts" / method_config_name),
            version_base=None,
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        cfg = compose(
            config_name="config",
            overrides=[
                *METHOD_CONFIGS[method_config_name],
                f"dataset_key={dataset_key}",
            ],
        )
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    return container


@pytest.mark.parametrize("dataset_key", DATASET_KEYS)
@pytest.mark.parametrize("method_config_name", sorted(METHOD_CONFIGS))
def test_resolved_config_is_unchanged_by_the_port(
    base_tree: Path,
    method_config_name: str,
    dataset_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Root configs name their search path relative to the working directory.
    monkeypatch.chdir(base_tree)
    base_config = _resolved_config(base_tree, method_config_name, dataset_key)
    monkeypatch.chdir(REPO_ROOT)
    ported_config = _resolved_config(
        REPO_ROOT, method_config_name, dataset_key
    )

    assert ported_config == base_config
