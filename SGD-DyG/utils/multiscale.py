from typing import Iterable, List, Sequence, Tuple

import torch


def build_multiscale_windows(
    adj_sequence: Sequence[torch.Tensor],
    window_size: int,
    decay_lambda: float = 0.8,
    persistence_threshold: int = 2,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Construct short/mid/long adjacency views for each time step.

    Args:
        adj_sequence: list of adjacency tensors ordered by time (history only).
        window_size: number of past slices to include in the window (inclusive).
        decay_lambda: exponential decay factor for the mid view.
        persistence_threshold: minimum count to keep an edge in the long view.

    Returns:
        Tuple of three adjacency lists (short, mid, long) with the same length as adj_sequence.
    """
    if len(adj_sequence) == 0:
        return [], [], []

    short_windows, mid_windows, long_windows = [], [], []
    for t in range(len(adj_sequence)):
        history_start = max(0, t - window_size + 1)
        history = adj_sequence[history_start: t + 1]
        reference = adj_sequence[t]

        short_windows.append(reference.coalesce())
        mid_windows.append(_decay_aggregate(history, decay_lambda, reference))
        long_windows.append(_persistent_aggregate(history, persistence_threshold, reference))

    return short_windows, mid_windows, long_windows


def compute_observability_stats(
    edges: torch.Tensor, adj_sequence: Sequence[torch.Tensor], window_size: int, num_nodes: int
) -> torch.Tensor:
    """
    Compute simple observability statistics for each edge over its history window.

    Args:
        edges: tensor shaped [3, num_edges] with (time, src, dst).
        adj_sequence: adjacency list aligned with the history used for predictions.
        window_size: number of past slices to consider.
        num_nodes: node cardinality (used for sanity checks).

    Returns:
        Tensor of shape [num_edges, 4] containing (cnt_uv, cnt_u, cnt_v, gap_uv).
    """
    device = edges.device
    num_edges = edges.size(1)

    cnt_uv = torch.zeros(num_edges, device=device, dtype=torch.float)
    cnt_u = torch.zeros_like(cnt_uv)
    cnt_v = torch.zeros_like(cnt_uv)
    gap_uv = torch.full((num_edges,), fill_value=window_size + 1, device=device, dtype=torch.float)

    time_to_edge_indices = {}
    for idx, t in enumerate(edges[0].tolist()):
        time_to_edge_indices.setdefault(t, []).append(idx)

    history_cpu = [adj.coalesce().cpu() for adj in adj_sequence]
    for t, edge_indices in time_to_edge_indices.items():
        if t < 0 or t >= len(adj_sequence):
            continue

        history_start = max(0, t - window_size + 1)
        history = history_cpu[history_start: t + 1]
        counts_map, node_counts, last_seen_map = _collect_window_statistics(history, history_start)

        for edge_idx in edge_indices:
            u = int(edges[1, edge_idx])
            v = int(edges[2, edge_idx])
            cnt_uv[edge_idx] = float(counts_map.get((u, v), 0))
            cnt_u[edge_idx] = float(node_counts.get(u, 0))
            cnt_v[edge_idx] = float(node_counts.get(v, 0))

            last_seen = last_seen_map.get((u, v))
            if last_seen is not None:
                gap_uv[edge_idx] = float(t - last_seen)

    return torch.stack((cnt_uv, cnt_u, cnt_v, gap_uv), dim=1)


def _decay_aggregate(history: Iterable[torch.Tensor], decay_lambda: float, reference: torch.Tensor) -> torch.Tensor:
    device, dtype, size = reference.device, reference.dtype, reference.size()
    indices, values = [], []
    for offset, adj in enumerate(reversed(list(history))):
        adj = adj.coalesce()
        if adj._nnz() == 0:
            continue
        indices.append(adj._indices())
        values.append(adj._values() * (decay_lambda ** offset))

    if not values:
        return _empty_sparse(size, device, dtype)

    cat_indices = torch.cat(indices, dim=1)
    cat_values = torch.cat(values)
    aggregated = torch.sparse_coo_tensor(cat_indices, cat_values, size=size, device=device, dtype=dtype)
    return aggregated.coalesce()


def _persistent_aggregate(history: Iterable[torch.Tensor], threshold: int, reference: torch.Tensor) -> torch.Tensor:
    device, dtype, size = reference.device, reference.dtype, reference.size()
    indices, values = [], []
    for adj in history:
        adj = adj.coalesce()
        if adj._nnz() == 0:
            continue
        indices.append(adj._indices())
        values.append(torch.ones_like(adj._values(), dtype=dtype))

    if not values:
        return _empty_sparse(size, device, dtype)

    cat_indices = torch.cat(indices, dim=1)
    cat_values = torch.cat(values)
    aggregated = torch.sparse_coo_tensor(cat_indices, cat_values, size=size, device=device, dtype=dtype).coalesce()

    edge_values = aggregated._values()
    edge_indices = aggregated._indices()
    mask = edge_values >= threshold
    if mask.any():
        return torch.sparse_coo_tensor(edge_indices[:, mask], edge_values[mask], size=size, device=device, dtype=dtype)
    return _empty_sparse(size, device, dtype)


def _collect_window_statistics(
    history: Sequence[torch.Tensor], start_time: int
) -> Tuple[dict, dict, dict]:
    counts_map, node_counts, last_seen_map = {}, {}, {}
    for offset, adj in enumerate(history):
        current_time = start_time + offset
        idx = adj._indices()
        if idx.numel() == 0:
            continue
        rows = idx[0].tolist()
        cols = idx[1].tolist()
        for u, v in zip(rows, cols):
            counts_map[(u, v)] = counts_map.get((u, v), 0) + 1
            node_counts[u] = node_counts.get(u, 0) + 1
            node_counts[v] = node_counts.get(v, 0) + 1
            last_seen_map[(u, v)] = current_time
    return counts_map, node_counts, last_seen_map


def _empty_sparse(size: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    empty_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
    empty_values = torch.tensor([], dtype=dtype, device=device)
    return torch.sparse_coo_tensor(empty_indices, empty_values, size=size, device=device, dtype=dtype)
