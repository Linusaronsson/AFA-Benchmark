import argparse
from pathlib import Path

import pandas as pd


def _read_time(path: Path | None) -> float | None:
    if path is None:
        return None
    # Some locales use ',' - replace it with '.'
    path_txt = path.read_text().replace(",", ".")
    return float(path_txt.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge pretrain/train/eval time measurements."
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--time_pretrain_path", type=Path, default=None)
    parser.add_argument("--time_train_path", type=Path, required=True)
    parser.add_argument("--time_eval_path", type=Path, required=True)
    args = parser.parse_args()

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "afa_method": args.method,
        "dataset": args.dataset,
        "time_pretrain": _read_time(args.time_pretrain_path),
        "time_train": _read_time(args.time_train_path),
        "time_eval": _read_time(args.time_eval_path),
    }

    # Write as a one-row DataFrame in Parquet format
    pd.DataFrame([row]).astype(
        {
            "afa_method": "string",
            "dataset": "string",
            "time_pretrain": "Float64",
            "time_train": "Float64",
            "time_eval": "Float64",
        }
    ).to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
