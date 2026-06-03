from dataclasses import dataclass
from typing import Any


@dataclass
class InitializerConfig:
    class_name: str
    kwargs: dict[str, Any]
