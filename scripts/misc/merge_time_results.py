import argparse
import csv
from pathlib import Path


def _read_time(path: Path | None) -> str:
    if path is None:
        return "null"
    return path.read_text().strip()


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

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "afa_method",
                "dataset",
                "time_pretrain",
                "time_train",
                "time_eval",
            ],
        )
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
