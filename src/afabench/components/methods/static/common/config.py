from dataclasses import dataclass


@dataclass
class StaticSelectorConfig:
    lr: float
    nepochs: int
    num_cells: list[int]
    patience: int


@dataclass
class StaticClassifierConfig:
    lr: float
    nepochs: int
    num_cells: list[int]
