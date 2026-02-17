"""Loading and saving "bundles", as described in `docs/bundle_format.md`."""

import json
from pathlib import Path
from typing import Any, Protocol, Self

from afabench.common.registry import get_class

# Do not allow loading of bundles with different major version
BUNDLE_VERSION = "1.0.0"


def is_same_major_semver(sv1: str, sv2: str) -> bool:
    """Check if two semantic versions in string form have different major, versions."""
    major1, _minor1, _patch1 = map(int, sv1.split("."))
    major2, _minor2, _patch2 = map(int, sv2.split("."))
    return major1 == major2


class Saveable(Protocol):
    """
    Protocol for objects that can be saved to disk.

    Classes may optionally define a _class_version class attribute
    following semantic versioning (major.minor.patch). If not defined,
    null will be stored in the manifest.
    """

    def save(self, path: Path) -> None: ...


class Loadeable(Protocol):
    """
    Protocol for objects that can be loaded from disk.

    Classes may optionally define a _class_version class attribute.
    """

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> Self:  # noqa: ANN401
        """
        Load from path with arbitrary kwargs.

        Uses Any for kwargs because this is a generic protocol that must
        support any type of underlying object with unknown loading parameters.
        """
        ...


def get_class_version(cls: type) -> str | None:
    """
    Get the class version for a given class.

    Returns:
        The class version string if defined, None otherwise

    Example:
        >>> get_class_version(MyAFAMethod)
        "1.0.0"
        >>> get_class_version(UnversionedClass)
        None
    """
    return getattr(cls, "_class_version", None)


def validate_class_version(version: str) -> bool:
    """
    Validate that a version string follows semantic versioning format.

    Args:
        version: Version string to validate (e.g., "1.0.0")

    Returns:
        True if valid semver format, False otherwise

    Example:
        >>> validate_class_version("1.0.0")
        True
        >>> validate_class_version("invalid")
        False
    """
    try:
        parts = version.split(".")
        if len(parts) != 3:
            return False
        major, minor, patch = map(int, parts)
        return all(x >= 0 for x in [major, minor, patch])
    except (ValueError, TypeError):
        return False


def save_bundle(obj: Saveable, path: Path, metadata: dict[str, Any]) -> None:
    """Save a bundle to disk. `path` is required to end with the `.bundle` extension to make it clear that this is a bundle."""
    assert (
        path.suffix == ".bundle"
    ), "Bundle path must end with .bundle extension"

    # Create parent directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    # Save object to path/data/. Object decides what this data folder should contain
    data_path = path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    obj.save(data_path)

    # Get class version (may be None)
    class_version = get_class_version(obj.__class__)

    # Validate class version format if it exists
    if class_version is not None and not validate_class_version(class_version):
        msg = f"Invalid class version format: {
            class_version
        }. Must be semantic versioning (major.minor.patch)"
        raise ValueError(msg)

    with (path / "manifest.json").open("w") as f:
        json.dump(
            {
                "bundle_version": BUNDLE_VERSION,
                "class_name": obj.__class__.__name__,
                "class_version": class_version,
                "metadata": metadata,
            },
            f,
            indent=2,
        )


def load_bundle(
    path: Path,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[Loadeable, dict[str, Any]]:
    """
    Load a bundle from disk. `path` is required to end with the `.bundle` extension to make it clear that this is a bundle.

    Args:
        path: Path to the bundle file
        **kwargs: Arbitrary keyword arguments forwarded to the object's load method.
                 Uses Any because we must support any underlying object type.

    Returns:
        A tuple of the loaded object and the manifest dictionary.
    """
    assert (
        path.suffix == ".bundle"
    ), "Bundle path must end with .bundle extension"

    with (path / "manifest.json").open("r") as f:
        manifest = json.load(f)
        # If stored major bundle version is different than the current one, panic
        if not is_same_major_semver(
            manifest["bundle_version"], BUNDLE_VERSION
        ):
            msg = (
                f"Cannot load bundle with version "
                f"{manifest['bundle_version']}; "
                f"current version is {BUNDLE_VERSION}."
            )
            raise ValueError(msg)

    # Lookup class in registry
    cls = get_class(manifest["class_name"])

    # Only check version compatibility if both stored and current versions exist
    stored_class_version = manifest["class_version"]
    current_class_version = get_class_version(cls)

    if (
        stored_class_version is not None
        and current_class_version is not None
        and not is_same_major_semver(
            stored_class_version, current_class_version
        )
    ):
        msg = (
            f"Cannot load object of class {manifest['class_name']} "
            f"with version {stored_class_version}; "
            f"current version is {current_class_version}."
        )
        raise ValueError(msg)

    # Load object
    obj = cls.load(path / "data", **kwargs)

    return obj, manifest
