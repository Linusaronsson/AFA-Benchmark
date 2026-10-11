from dataclasses import dataclass
from typing import Any


@dataclass
class ImagePatchUnmaskerConfig:
    image_side_length: int
    n_channels: int
    patch_size: int


@dataclass
class UnmaskerConfig:
    class_name: str
    kwargs: dict[str, Any]
