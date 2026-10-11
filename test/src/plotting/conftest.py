import pytest

from afabench.plotting.config import PlottingDisplayConfig


@pytest.fixture
def plotting_config() -> PlottingDisplayConfig:
    return PlottingDisplayConfig(
        plot_width=6,
        plot_height=3,
        plot_font_family="DejaVu Sans",
        method_name_mapping={"jafa": "JAFA", "gdfs": "GDFS"},
        method_policy_family_mapping={"jafa": "jafa", "gdfs": "gdfs"},
        method_family_color_schemes={
            "test": {"jafa": "#117733", "gdfs": "#88CCEE"}
        },
        active_method_color_scheme="test",
        dataset_name_mapping={"cube": "Cube", "mnist": "MNIST"},
        datasets_with_f_score=["mnist"],
        dataset_sets={"all": ["cube", "mnist"]},
        color_palette_name="Set2",
    )
