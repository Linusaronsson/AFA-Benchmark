"""The shared method palette must stay consistent and stay readable."""

import itertools
import math
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


# Machado, Oliveira and Fernandes (2009), severity 1.0, applied in linear sRGB.
CVD_MATRICES = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Floors in OKLab dE x100. Normal vision is the stricter bar because a reader
# with full colour vision who cannot separate two series has no fallback; the
# CVD floor is lower because marker shape carries identity redundantly there.
NORMAL_FLOOR, CVD_FLOOR = 15.0, 8.0


def _oklab(
    hex_color: str, matrix: tuple[tuple[float, ...], ...]
) -> tuple[float, ...]:
    """One colour as OKLab, after simulating a vision type in linear sRGB."""
    raw = hex_color.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    red, green, blue = (
        sum(matrix[i][j] * linear[j] for j in range(3)) for i in range(3)
    )
    long, medium, short = (
        0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue,
        0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue,
        0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue,
    )
    long, medium, short = (
        math.copysign(abs(v) ** (1 / 3), v) for v in (long, medium, short)
    )
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


@pytest.mark.parametrize(
    ("vision", "matrix", "floor"),
    [
        ("normal", IDENTITY, NORMAL_FLOOR),
        *((name, matrix, CVD_FLOOR) for name, matrix in CVD_MATRICES.items()),
    ],
)
def test_every_pair_stays_separable(
    vision: str,
    matrix: tuple[tuple[float, ...], ...],
    floor: float,
) -> None:
    """
    Every pair of method colours, not only adjacent ones.

    The module docstring promises this and nothing used to check it, so
    re-stepping one entry could quietly collapse a pair under deuteranopia while
    every other test still passed. Measured over all pairs because a figure puts
    arbitrary methods beside each other.
    """
    worst, pair = min(
        (
            (
                100
                * math.dist(
                    _oklab(METHOD_COLORS[left], matrix),
                    _oklab(METHOD_COLORS[right], matrix),
                ),
                (left, right),
            )
            for left, right in itertools.combinations(METHOD_COLORS, 2)
        ),
    )
    assert worst >= floor, (
        f"{vision}: {pair[0]} and {pair[1]} differ by {worst:.1f}, "
        f"below the {floor} floor. Re-step the palette as a set."
    )


def test_yaml_overrides_match_the_module() -> None:
    """The plotnine path reads colours from Hydra, the rest import them."""
    config = yaml.safe_load(COMMON_CONFIG.read_text())
    assert config["method_color_overrides"] == METHOD_COLORS


def test_yaml_names_every_method_the_module_colours() -> None:
    config = yaml.safe_load(COMMON_CONFIG.read_text())
    assert set(METHOD_COLORS) <= set(config["method_policy_family_mapping"])
    assert set(METHOD_COLORS) <= set(config["method_name_mapping"])
