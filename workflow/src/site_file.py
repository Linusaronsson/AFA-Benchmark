"""Reading a site file: the allocations of a site profile (site.yaml)."""

from collections.abc import Mapping
from pathlib import Path

import yaml


def read_site_file(path: Path) -> Mapping[str, object]:
    """Return the site file at `path`, which holds a mapping."""
    site = yaml.safe_load(path.read_text())
    if not isinstance(site, Mapping):
        message = f"Expected a mapping in the site file {path}, got {site!r}"
        raise TypeError(message)
    return site
