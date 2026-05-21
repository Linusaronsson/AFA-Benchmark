"""Public API for downloading pretrained AFA-Benchmark artifacts from HuggingFace Hub."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from afabench.common.bundle import load_bundle
from afabench.common.registry import get_class

if TYPE_CHECKING:
    from afabench.common.custom_types import AFAInitializer, AFAUnmasker

HF_REPO_ID = "afabench/checkpoints"

_NO_PRETRAIN = "NO_PRETRAIN"


def _initializer_tag(initializer: str) -> str:
    return f"initializer-{initializer}"


def _budget_str(value: float | str | None) -> str:
    return "null" if value is None else str(value)


def _method_remote_path(
    method: str,
    dataset: str,
    instance_idx: int,
    initializer: str,
    train_seed: int,
    hard_budget: int | str | None,
    soft_budget_param: float | str | None,
    pretrain_seed: int | None,
) -> str:
    tag = _initializer_tag(initializer)
    pretrain_dir = (
        _NO_PRETRAIN
        if pretrain_seed is None
        else f"pretrain_seed-{pretrain_seed}"
    )
    train_dir = (
        f"train_seed-{train_seed}"
        f"+train_hard_budget-{_budget_str(hard_budget)}"
        f"+train_soft_budget_param-{_budget_str(soft_budget_param)}"
    )
    return (
        f"trained_methods/{tag}/{method}/"
        f"dataset-{dataset}+instance_idx-{instance_idx}/"
        f"{pretrain_dir}/"
        f"{train_dir}/"
        f"method.bundle.zip"
    )


def _model_remote_path(
    model_name: str,
    dataset: str,
    instance_idx: int,
    initializer: str,
    pretrain_seed: int,
) -> str:
    tag = _initializer_tag(initializer)
    return (
        f"pretrained_models/{tag}/{model_name}/"
        f"dataset-{dataset}+instance_idx-{instance_idx}/"
        f"pretrain_seed-{pretrain_seed}/"
        f"model.bundle.zip"
    )


def _classifier_remote_path(
    dataset: str,
    initializer: str,
    method: str | None,
) -> str:
    tag = _initializer_tag(initializer)
    name = (
        f"method-{method}+dataset-{dataset}"
        if method
        else f"dataset-{dataset}"
    )
    return f"trained_classifiers/{tag}/{name}.bundle.zip"


def _download_and_extract(
    remote_path: str,
    repo_id: str,
    cache_dir: Path | None,
) -> Path:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    zip_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    bundle_path = zip_path.with_suffix("")
    if not bundle_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
    return bundle_path


def get_method(
    method: str,
    dataset: str,
    *,
    instance_idx: int = 0,
    initializer: str = "cold",
    train_seed: int = 0,
    hard_budget: int | str | None = None,
    soft_budget_param: float | str | None = None,
    pretrain_seed: int | None = None,
    repo_id: str = HF_REPO_ID,
    cache_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Download and load a trained AFA method from HuggingFace Hub."""
    remote_path = _method_remote_path(
        method,
        dataset,
        instance_idx,
        initializer,
        train_seed,
        hard_budget,
        soft_budget_param,
        pretrain_seed,
    )
    bundle_path = _download_and_extract(remote_path, repo_id, cache_dir)
    return load_bundle(bundle_path)


def get_model(
    model_name: str,
    dataset: str,
    *,
    instance_idx: int = 0,
    initializer: str = "cold",
    pretrain_seed: int = 0,
    repo_id: str = HF_REPO_ID,
    cache_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Download and load a pretrained model from HuggingFace Hub."""
    remote_path = _model_remote_path(
        model_name, dataset, instance_idx, initializer, pretrain_seed
    )
    bundle_path = _download_and_extract(remote_path, repo_id, cache_dir)
    return load_bundle(bundle_path)


def get_classifier(
    dataset: str,
    *,
    initializer: str = "cold",
    method: str | None = None,
    repo_id: str = HF_REPO_ID,
    cache_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Download and load a trained classifier from HuggingFace Hub."""
    remote_path = _classifier_remote_path(dataset, initializer, method)
    bundle_path = _download_and_extract(remote_path, repo_id, cache_dir)
    return load_bundle(bundle_path)


def get_unmasker(name: str, **kwargs: Any) -> AFAUnmasker:  # noqa: ANN401
    """Instantiate an unmasker by class name."""
    return get_class(name)(**kwargs)


def get_initializer(name: str, **kwargs: Any) -> AFAInitializer:  # noqa: ANN401
    """Instantiate an initializer by class name."""
    return get_class(name)(**kwargs)
