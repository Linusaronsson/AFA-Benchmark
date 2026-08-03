"""Collect current-workflow benchmark TSVs with hardware provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


def _tokens(parts: tuple[str, ...]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        for token in Path(part).stem.split("+"):
            if "-" not in token:
                continue
            key, value = token.split("-", maxsplit=1)
            if key in {
                "dataset",
                "method",
                "mechanism",
                "p",
                "strategy",
                "instance",
                "train_hard_budget",
                "eval_hard_budget",
                "pretrain",
            }:
                fields[key] = value
    return fields


def benchmark_identity(path: Path, root: Path, rule: str) -> dict[str, Any]:
    """Scientific cell identity encoded by one workflow benchmark path."""
    fields = _tokens(path.relative_to(root).parts)
    record: dict[str, Any] = {"rule": rule}
    record.update(fields)
    for key in ("p", "train_hard_budget", "eval_hard_budget"):
        if key in record:
            record[key] = float(record[key])
    if "instance" in record:
        record["instance"] = int(record["instance"])
    if "pretrain" in record:
        record["pretrain_key"] = record.pop("pretrain")
    return record


def _manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    command = manifest["command"]
    host = manifest["host"]
    software = manifest["software"]
    return {
        "git_commit": manifest["git_commit"],
        "device": command["device"],
        "cores": command["cores"],
        "mem_mb": command["mem_mb"],
        "gpu_workers": command.get("gpu_workers", command.get("gpu_slots", 0)),
        "architecture": host["architecture"],
        "torch": software["torch"],
        "torch_cuda": software["torch_cuda"],
        "cuda_devices": manifest["cuda_devices"],
    }


def namespace_provenance(
    manifest_root: Path, namespace: str
) -> dict[str, Any]:
    """Return one signature, rejecting mixed hardware within a namespace."""
    paths = sorted((manifest_root / namespace).glob("*.json"))
    if not paths:
        return {
            "hardware_signature": "untracked",
            "git_commit": "untracked",
            "device": "untracked",
            "architecture": "untracked",
            "torch": "untracked",
            "torch_cuda": "untracked",
            "cuda_devices": "[]",
        }
    signatures = {
        json.dumps(
            _manifest_signature(json.loads(path.read_text())), sort_keys=True
        )
        for path in paths
    }
    if len(signatures) != 1:
        message = f"namespace {namespace} mixes execution environments"
        raise ValueError(message)
    signature = json.loads(signatures.pop())
    return {
        "hardware_signature": json.dumps(signature, sort_keys=True),
        **{
            key: json.dumps(value) if isinstance(value, list) else value
            for key, value in signature.items()
        },
    }


def namespace_gpu_telemetry(
    telemetry_root: Path, namespace: str
) -> dict[str, float | int]:
    """Summarize all allocation samples for one hardware-pure namespace."""
    paths = sorted((telemetry_root / namespace).glob("*.csv"))
    if not paths:
        return {"gpu_samples": 0}
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    numeric = [
        "utilization_percent",
        "memory_used_mb",
        "memory_total_mb",
        "power_draw_w",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric)
    if frame.empty:
        return {"gpu_samples": 0}
    active = frame[frame["utilization_percent"] > 0]
    active_utilization = cast("pd.Series", active["utilization_percent"])
    return {
        "gpu_samples": len(frame),
        "gpu_active_samples": len(active),
        "gpu_utilization_mean_percent": float(
            frame["utilization_percent"].mean()
        ),
        "gpu_utilization_active_median_percent": float(
            active_utilization.median()
        ),
        "gpu_utilization_active_p95_percent": float(
            active_utilization.quantile(0.95)
        ),
        "gpu_memory_peak_mb": float(frame["memory_used_mb"].max()),
        "gpu_memory_total_mb": float(frame["memory_total_mb"].max()),
        "gpu_power_peak_w": float(frame["power_draw_w"].max()),
    }


def _read_benchmark(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        message = f"empty benchmark: {path}"
        raise ValueError(message)
    return {
        "wall_seconds": float(frame["s"].sum()),
        "cpu_seconds": float(frame["cpu_time"].sum()),
        "peak_rss_mb": float(frame["max_rss"].max()),
    }


def collect_namespace(output_root: Path, namespace: str) -> pd.DataFrame:
    benchmark_root = output_root / "benchmark" / namespace
    rows: list[dict[str, Any]] = []
    if benchmark_root.exists():
        for path in sorted(benchmark_root.rglob("*.tsv")):
            relative = path.relative_to(benchmark_root)
            top = relative.parts[0]
            rule = top
            if top == "pretrain_restoration_pvae":
                rule = f"{top}_{relative.parts[1]}"
            row = benchmark_identity(path, benchmark_root, rule)
            row.update(_read_benchmark(path))
            row["path"] = str(path)
            rows.append(row)
    for split in ("val", "test"):
        eval_root = output_root / "eval" / split / namespace
        if not eval_root.exists():
            continue
        for path in sorted(eval_root.rglob("benchmark.tsv")):
            row = benchmark_identity(path, eval_root, "eval_method")
            row.update(_read_benchmark(path))
            row["eval_split"] = split
            row["path"] = str(path)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        message = f"no benchmark TSVs found for {namespace}"
        raise ValueError(message)
    frame.insert(0, "namespace", namespace)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", action="append", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("extra/output/missing_data"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest_root = arguments.output_root / "run_manifests"
    telemetry_root = arguments.output_root / "gpu_telemetry"
    frames = []
    for namespace in arguments.namespace:
        frame = collect_namespace(arguments.output_root, namespace)
        provenance = namespace_provenance(manifest_root, namespace)
        provenance.update(namespace_gpu_telemetry(telemetry_root, namespace))
        for key, value in provenance.items():
            frame[key] = value
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(arguments.output, index=False)
    print(
        combined.groupby(["namespace", "rule"])["wall_seconds"]
        .agg(["size", "sum", "median"])
        .round(2)
        .to_string()
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
