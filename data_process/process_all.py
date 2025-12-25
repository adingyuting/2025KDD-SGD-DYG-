"""Preprocess non-Bitcoin datasets into `.mat` files for training.

Run from repo root (or from `data_process/`):

    python data_process/process_all.py

Optionally specify datasets (keys below) to limit processing:

    python data_process/process_all.py --datasets wiki_gl digg

Inputs
------
- CSV files located at `../data/<dataset>/<dataset>.csv`
- Each CSV must contain either:
    * 3 columns: src, dst, time
    * 4+ columns: src, dst, label, time (time is always the last column)

Outputs
-------
`<dataset>.mat` saved to the same dataset folder. The preprocessing steps
match those used in `process_bitcoin.py` / `process.py` (optional
symmetrization, edge life, Laplacian normalization).
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from constant import (
    DBLP_TS,
    DIGG_TS,
    LAST_FM_TS,
    PPIN_TS,
    VAL_RATE,
    WIKI_EO_TS,
    WIKI_GL_TS,
    TEST_RATE,
)
from process import pre_process, save_file, split_data

# Keep aligned with process.py defaults
EDGE_LIFE = False
EDGE_LIFE_WINDOW = 10
MAKE_SYMMETRIC = False

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../data")


def dataset_configs() -> Dict[str, Dict]:
    """Return configs for all non-Bitcoin datasets."""

    return {
        # Wiki datasets
        "wiki_gl": {
            "csv_path": os.path.join(DATA_ROOT, "wiki_gl", "wiki_gl.csv"),
            "mat_name": "wiki_gl.mat",
            "ts": WIKI_GL_TS,
        },
        "wiki_eo": {
            "csv_path": os.path.join(DATA_ROOT, "wiki_eo", "wiki_eo.csv"),
            "mat_name": "wiki_eo.mat",
            "ts": WIKI_EO_TS,
        },
        # Social / information
        "digg": {
            "csv_path": os.path.join(DATA_ROOT, "digg", "digg.csv"),
            "mat_name": "digg.mat",
            "ts": DIGG_TS,
        },
        "last_fm": {
            "csv_path": os.path.join(DATA_ROOT, "last_fm", "last_fm.csv"),
            "mat_name": "last_fm.mat",
            "ts": LAST_FM_TS,
        },
        # Academic / biological
        "dblp": {
            "csv_path": os.path.join(DATA_ROOT, "dblp", "dblp.csv"),
            "mat_name": "dblp.mat",
            "ts": DBLP_TS,
        },
        "ppin": {
            "csv_path": os.path.join(DATA_ROOT, "ppin", "ppin.csv"),
            "mat_name": "ppin.mat",
            "ts": PPIN_TS,
        },
    }


def ensure_zero_based(nodes: torch.Tensor) -> torch.Tensor:
    """Shift node ids to zero-based if necessary."""

    min_node = int(nodes.min().item())
    if min_node > 0:
        nodes = nodes - min_node
    return nodes


def infer_labels(data: torch.Tensor, label_col: Optional[int]) -> torch.Tensor:
    """Return edge labels or default ones when no label column is present."""

    if label_col is None:
        return torch.ones(data.size(0), dtype=torch.double)
    return data[:, label_col].to(torch.double)


def build_time_buckets(
    data: torch.Tensor, ts: int, time_dim: int, label_col: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign edges to discrete time buckets and return indices/labels."""

    max_time = float(data[:, time_dim].max())
    min_time = float(data[:, time_dim].min())
    time_delta = max(1, math.floor((max_time - min_time) / ts))

    within_range = data[:, time_dim] <= min_time + time_delta * ts
    data = data[within_range]

    edge_count = data.size(0)
    tensor_idx = torch.zeros([edge_count, 3], dtype=torch.long)
    tensor_labels = torch.zeros(edge_count, dtype=torch.double)

    start = min_time
    for t in range(ts):
        end = start + time_delta
        if t == ts - 1:
            idx = (data[:, time_dim] >= start) & (data[:, time_dim] <= end)
        else:
            idx = (data[:, time_dim] >= start) & (data[:, time_dim] < end)
        start = end

        tensor_idx[idx, 1:3] = data[idx, 0:2].to(torch.long)
        tensor_idx[idx, 0] = t

        labels = infer_labels(data[idx], label_col)
        tensor_labels[idx] = labels

    return tensor_idx, tensor_labels


def preprocess_dataset(dataset_key: str, cfg: Dict):
    print(f"Processing dataset: {dataset_key}")

    raw = np.loadtxt(cfg["csv_path"], delimiter=",")
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)

    data = torch.tensor(raw)
    cols = data.size(1)
    if cols < 3:
        raise ValueError(
            f"Dataset {dataset_key} must have at least 3 columns (src, dst, time). Found {cols}."
        )

    time_dim = cols - 1
    label_col = 2 if cols > 3 else None

    # Normalize node ids to start from zero
    data[:, 0] = ensure_zero_based(data[:, 0])
    data[:, 1] = ensure_zero_based(data[:, 1])

    tensor_idx, tensor_labels = build_time_buckets(
        data, cfg["ts"], time_dim=time_dim, label_col=label_col
    )

    N = int(max(data[:, 0].max(), data[:, 1].max()).item()) + 1
    TS = cfg["ts"]

    A = torch.sparse.DoubleTensor(
        tensor_idx.transpose(1, 0),
        torch.ones(tensor_idx.size(0)),
        torch.Size([TS, N, N]),
    ).coalesce()

    val_samples = int(TS * VAL_RATE)
    test_samples = int(TS * TEST_RATE)
    T = TS - val_samples - test_samples

    A_train = split_data(A, N, T, 0, T)
    A_val = split_data(A, N, T, val_samples, T + val_samples)
    A_test = split_data(A, N, T, val_samples + test_samples, TS)

    train, val, test = pre_process(A_train, A_val, A_test, N, T)

    save_file(
        tensor_idx,
        tensor_labels,
        A,
        A_train,
        A_val,
        A_test,
        train,
        val,
        test,
        os.path.join(DATA_ROOT, dataset_key),
        cfg["mat_name"],
    )

    print(f"Saved {cfg['mat_name']} to {os.path.join(DATA_ROOT, dataset_key)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess non-Bitcoin datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to process (default: all non-Bitcoin datasets)",
    )

    args = parser.parse_args()

    configs = dataset_configs()
    targets: List[str] = args.datasets if args.datasets is not None else list(configs.keys())

    for dataset in targets:
        if dataset not in configs:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(configs.keys())}")
        preprocess_dataset(dataset, configs[dataset])


if __name__ == "__main__":
    main()
