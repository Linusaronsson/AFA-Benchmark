"""
Every `set-resources` key in a cluster profile must name a real rule.

Snakemake silently ignores a key that matches no rule, so a name that drifts
does not fail, it downgrades the job to `default-resources`. On Vera that is
120 minutes and 4 GB, which walltime-kills anything substantial after it has
already consumed queue time. That is exactly what had happened: both tracked
profiles targeted `orchestration/pipeline.smk` rule names while the
missing-data study runs `orchestration/missing_data.smk`, and not one key
matched.

A profile may legitimately serve more than one workflow, so a key is accepted
if any snakefile in the repository defines it.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SNAKEFILE_DIR = REPO / "extra/workflow/snakefiles"
PROFILE_DIR = REPO / "extra/workflow/profiles"

# Snakemake defines these itself; they are not workflow rules.
RESERVED = frozenset()


def _known_rules() -> set[str]:
    rules: set[str] = set()
    for path in SNAKEFILE_DIR.rglob("*.smk"):
        rules |= set(
            re.findall(
                r"^rule\s+([A-Za-z_][A-Za-z0-9_]*)\s*:",
                path.read_text(),
                re.MULTILINE,
            )
        )
    return rules


def _cluster_profiles() -> list[Path]:
    # Profiles under `config/` select configfiles, not execution resources.
    return sorted(
        p
        for p in PROFILE_DIR.glob("*/config.yaml")
        if p.parent.parent == PROFILE_DIR and p.parent.name != "config"
    )


def test_repository_has_cluster_profiles() -> None:
    """Guard the guard: a bad glob would make every test below vacuous."""
    assert _cluster_profiles(), "no cluster profiles found to check"
    assert _known_rules(), "no rules parsed out of the snakefiles"


@pytest.mark.parametrize(
    "profile", _cluster_profiles(), ids=lambda p: p.parent.name
)
def test_set_resources_keys_are_real_rules(profile: Path) -> None:
    config = yaml.safe_load(profile.read_text()) or {}
    declared = set(config.get("set-resources", {})) - RESERVED
    unknown = sorted(declared - _known_rules())
    assert not unknown, (
        f"{profile.relative_to(REPO)} sets resources for rules that do not "
        f"exist: {unknown}. These are silently ignored, so the jobs would fall "
        f"back to default-resources instead of failing."
    )


@pytest.mark.parametrize(
    "profile", _cluster_profiles(), ids=lambda p: p.parent.name
)
def test_expensive_missing_data_rules_are_resourced(profile: Path) -> None:
    """
    The long-running rules must not inherit a short default.

    These are the ones that train networks; everything else in the workflow is
    minutes. Listing them explicitly means adding a new training rule without
    resourcing it fails here rather than on the cluster.
    """
    config = yaml.safe_load(profile.read_text()) or {}
    declared = set(config.get("set-resources", {}))
    expensive = {
        "pretrain_incomplete_restoration_pvae",
        "pretrain_oracle_restoration_pvae",
        "pretrain_missing_data_method",
        "train_missing_data_method_with_pretraining",
        "train_missing_data_method_without_pretraining",
        "eval_missing_data_method",
    }
    missing = sorted(expensive - declared)
    assert not missing, (
        f"{profile.relative_to(REPO)} leaves these on default-resources: "
        f"{missing}"
    )
