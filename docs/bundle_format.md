# Bundle format

Every object that needs to be saved/loaded follows the same format.

---

## Usage

Use `afabench.common.bundle.save_bundle()` to save bundles and `afabench.common.bundle.load_bundle()` to load bundles.

## Format specification

The object bundle is a folder with the following structure:

```
my_object.bundle/
    manifest.json
    data/
        ... object-specific content ...
```

- The `.bundle` suffix is **mandatory**.
- The `data/` folder contains any arbitrary representation the object chooses.
- The manifest contains all information necessary to reconstruct the object and check compatibility.

The manifest contains essential information for reconstructing the object and optional metadata:
```json
{
    "bundle_version": 1,
    "class_name": "MyClass",
    "class_version": "1.3.2",
    "metadata": {
        "param1": 32,
        "param2": 0.13,
        "seed": 5
    }
}
```

- `bundle_version`: The version of the bundle specification/protocol.
- `class_name`: A globally unique string that identifies the object's class; used to look up the appropriate class with `afabench.common.registry.get_class()`.
- `class_version`: The object's own version, following Semantic Versioning (SemVer). Major version differences indicate incompatibility.
- `metadata`: Optional, arbitrary information about the object.

## Registering classes for deserialization

When `load_bundle()` reads a manifest, it needs to know which Python class corresponds to the `class_name` in the manifest. This mapping is defined in `afabench/common/registry.py` in the `REGISTERED_CLASSES` dictionary:

```python
REGISTERED_CLASSES = {
    "MyClass": "my_module.submodule.MyClass",
    "MyDataset": "afabench.common.datasets.datasets.MyDataset",
    "RandomDummyAFAMethod": "afabench.common.afa_methods.RandomDummyAFAMethod",
    # ... more entries ...
}
```

Each entry maps a class name to its full import path. When deserializing, the framework uses this registry to dynamically import and instantiate the correct class.

**Important:** If you create a new method class or dataset class that needs to be saved and loaded as a bundle, you must add an entry to `REGISTERED_CLASSES`.

## Implementing save and load methods

Objects that can be serialized as bundles need to implement `save()` and `load()` methods. Here's an example:

```python
from pathlib import Path
from typing import Self

class MyClass:
    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def save(self, path: Path) -> None:
        """Save object data to the bundle's data/ folder.
        
        The path parameter is the data/ folder itself.
        """
        path.mkdir(parents=True, exist_ok=True)
        import json
        data = {
            "value": self.value,
            "name": self.name,
        }
        with open(path / "data.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load object from the bundle's data/ folder.
        
        The path parameter is the data/ folder itself.
        """
        import json
        with open(path / "data.json", "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.value = data["value"]
        obj.name = data["name"]
        return obj
```

The `path` parameter passed to `save()` and `load()` is the `data/` folder itself, not the bundle root. Save and load files relative to this `path`.

