"""
Batch CSV processor for the five datasets (2, 3, 4, 6, 7).

It looks for inputs in:
- data/<name>.csv
- data/<name>/<name>.csv

Outputs are written to:
- data/processed/<name>/<name>_processed.csv

The default dataset names are:
- dblp
- digg
- last_fm
- wiki_eo
- wiki_gl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

DEFAULT_DATASETS = ["dblp", "digg", "last_fm", "wiki_eo", "wiki_gl"]


def find_input_path(name: str, data_dir: Path) -> Path:
    """Return the existing CSV path for a dataset name."""
    candidates = [
        data_dir / f"{name}.csv",
        data_dir / name / f"{name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find CSV for '{name}'. Expected at: "
        + ", ".join(str(p) for p in candidates)
    )


def process_dataset(
    name: str,
    data_dir: Path,
    output_dir: Path,
    drop_empty_rows: bool = True,
    drop_duplicates: bool = True,
) -> Path:
    """
    Process a single dataset and return the output CSV path.

    Current steps (conservative defaults):
    - drop rows that are fully empty (if drop_empty_rows)
    - drop duplicate rows (if drop_duplicates)
    """
    input_path = find_input_path(name, data_dir)
    df = pd.read_csv(input_path)

    if drop_empty_rows:
        df = df.dropna(how="all")
    if drop_duplicates:
        df = df.drop_duplicates()

    output_dataset_dir = output_dir / name
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dataset_dir / f"{name}_processed.csv"
    df.to_csv(output_path, index=False)
    return output_path


def process_all(
    dataset_names: Iterable[str],
    data_dir: Path,
    output_dir: Path,
    drop_empty_rows: bool = True,
    drop_duplicates: bool = True,
) -> Dict[str, Tuple[str, str]]:
    """
    Process all datasets and return a status dictionary.

    Status values:
    - ("ok", <output_path>)
    - ("missing", <message>)
    - ("error", <message>)
    """
    results: Dict[str, Tuple[str, str]] = {}
    for name in dataset_names:
        try:
            output_path = process_dataset(
                name=name,
                data_dir=data_dir,
                output_dir=output_dir,
                drop_empty_rows=drop_empty_rows,
                drop_duplicates=drop_duplicates,
            )
            results[name] = ("ok", str(output_path))
        except FileNotFoundError as exc:
            results[name] = ("missing", str(exc))
        except Exception as exc:  # pragma: no cover - defensive catch
            results[name] = ("error", str(exc))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process multiple CSV datasets in one go."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="List of dataset names to process. Default: %(default)s",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Base directory containing the input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data") / "processed",
        type=Path,
        help="Directory to write processed CSVs.",
    )
    parser.add_argument(
        "--keep-empty-rows",
        action="store_true",
        help="Keep rows that are fully empty instead of dropping them.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate rows instead of dropping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = process_all(
        dataset_names=args.datasets,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        drop_empty_rows=not args.keep_empty_rows,
        drop_duplicates=not args.keep_duplicates,
    )
    for name, (status, message) in results.items():
        print(f"[{status.upper()}] {name}: {message}")


if __name__ == "__main__":
    main()
