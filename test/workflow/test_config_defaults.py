from pathlib import Path

from omegaconf import OmegaConf


def test_direct_unmasker_kwargs_are_empty_mapping() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "conf"
        / "components"
        / "unmasker"
        / "direct.yaml"
    )

    config = OmegaConf.load(config_path)

    assert dict(config.kwargs) == {}
