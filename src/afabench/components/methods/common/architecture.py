from enum import StrEnum


class ArchitectureKind(StrEnum):
    """Which network family a method builds: an MLP or an image backbone."""

    TABULAR = "tabular"
    IMAGE = "image"
