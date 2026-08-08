"""The shared method palette must stay consistent and stay readable."""

from pathlib import Path

import pytest
import yaml

from afabench.plotting.methods import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    PRIMARY_METHODS,
)

COMMON_CONFIG = Path("extra/conf/scripts/plotting/common/default.yaml")


@pytest.mark.parametrize(
    "mapping",
    [METHOD_LABELS, METHOD_MARKERS, METHOD_LINESTYLES],
    ids=["labels", "markers", "linestyles"],
)
def test_every_method_has_every_channel(mapping: dict[str, str]) -> None:
    assert set(mapping) == set(METHOD_COLORS)


def test_primary_methods_are_methods() -> None:
    assert set(PRIMARY_METHODS) <= set(METHOD_COLORS)


@pytest.mark.parametrize(
    "channel",
    [METHOD_COLORS, METHOD_MARKERS],
    ids=["colors", "markers"],
)
def test_channel_separates_every_method(channel: dict[str, str]) -> None:
    """
    No two methods may share a colour, or a marker.

    Family colour used to alias ol_with_mask onto ol_full_state, which drew the
    study's headline contrast as a single line.
    """
    assert len(set(channel.values())) == len(channel)


def test_yaml_overrides_match_the_module() -> None:
    """The plotnine path reads colours from Hydra, the rest import them."""
    config = yaml.safe_load(COMMON_CONFIG.read_text())
    assert config["method_color_overrides"] == METHOD_COLORS


def test_yaml_names_every_method_the_module_colours() -> None:
    config = yaml.safe_load(COMMON_CONFIG.read_text())
    assert set(METHOD_COLORS) <= set(config["method_policy_family_mapping"])
    assert set(METHOD_COLORS) <= set(config["method_name_mapping"])
