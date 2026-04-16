"""Similarity computation utilities.

All public functions return a *sparse* CSR matrix (top-k per row) plus a
:class:`pandas.Index` that maps matrix positions back to entity IDs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

Metric = Literal["cosine", "overlap"]

_DEFAULT_BATCH_SIZE = 2048


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def metadata_similarity(
    df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
    metric: Metric = "cosine",
    top_k: int = 100,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> tuple[sparse.csr_matrix, pd.Index]:
    """Build a sparse top-k similarity matrix from entity metadata.

    Parameters
    ----------
    df : DataFrame
        One row per entity with an ID column and feature columns.
    id_col : str
        Column containing entity IDs.
    feature_cols : list[str]
        Columns to use for similarity computation.
    metric : ``"cosine"`` | ``"overlap"``
        Similarity metric to use.
    top_k : int
        Number of most-similar neighbours to retain per row.
    batch_size : int
        Rows processed per chunk (controls peak memory).
    """
    ids = pd.Index(df[id_col].values)
    features = df[feature_cols]

    if metric == "cosine":
        encoded = _one_hot_encode(features)
        sim = _batched_topk(encoded, encoded, top_k, batch_size, _cosine_chunk)
    elif metric == "overlap":
        codes = _category_codes(features)
        sim = _batched_topk(codes, codes, top_k, batch_size, _overlap_chunk)
    else:
        raise ValueError(f"Unknown metric {metric!r}")

    return sim, ids


def interaction_similarity(
    interactions: pd.DataFrame,
    entity_col: str,
    target_col: str,
    weight_col: str,
    metric: Metric = "cosine",
    top_k: int = 100,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> tuple[sparse.csr_matrix, pd.Index]:
    """Build a sparse top-k similarity matrix from interaction profiles.

    Each entity's profile is a vector of interaction weights across all
    targets.  Two entities are similar when they interact with similar
    targets.

    Parameters
    ----------
    interactions : DataFrame
        Must contain *entity_col*, *target_col*, and *weight_col*.
    entity_col, target_col, weight_col : str
        Column names.
    metric : ``"cosine"`` | ``"overlap"``
        Similarity metric.
    top_k, batch_size : int
        Same semantics as :func:`metadata_similarity`.
    """
    entity_ids = pd.Index(interactions[entity_col].unique())
    target_ids = pd.Index(interactions[target_col].unique())

    row_idx = entity_ids.get_indexer(interactions[entity_col])
    col_idx = target_ids.get_indexer(interactions[target_col])
    weights = interactions[weight_col].values.astype(np.float32)

    profile = sparse.csr_matrix(
        (weights, (row_idx, col_idx)),
        shape=(len(entity_ids), len(target_ids)),
    )

    if metric == "cosine":
        sim = _batched_topk(profile, profile, top_k, batch_size, _cosine_chunk)
    elif metric == "overlap":
        binary = (profile > 0).astype(np.float32)
        sim = _batched_topk(binary, binary, top_k, batch_size, _cosine_chunk)
    else:
        raise ValueError(f"Unknown metric {metric!r}")

    return sim, entity_ids


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _one_hot_encode(features: pd.DataFrame) -> sparse.csr_matrix:
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    return enc.fit_transform(features.astype(str)).tocsr().astype(np.float32)


def _category_codes(features: pd.DataFrame) -> np.ndarray:
    """Return an (n, d) int array of category codes (one col per feature)."""
    return np.column_stack(
        [features[c].astype("category").cat.codes.values for c in features.columns]
    )


# ---------------------------------------------------------------------------
# Batched top-k engine
# ---------------------------------------------------------------------------


def _cosine_chunk(
    batch: np.ndarray | sparse.spmatrix,
    full: np.ndarray | sparse.spmatrix,
) -> np.ndarray:
    """Return dense (batch_size, n) cosine similarities."""
    return cosine_similarity(batch, full).astype(np.float32)


def _overlap_chunk(batch: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Return dense (batch_size, n) overlap similarities.

    Overlap = fraction of columns where values match.
    """
    n_cols = batch.shape[1]
    # batch: (b, d), full: (n, d) -> compare via broadcasting
    # Process in sub-batches to avoid huge (b, n, d) intermediate
    b = batch.shape[0]
    n = full.shape[0]
    out = np.empty((b, n), dtype=np.float32)
    for i in range(b):
        out[i] = np.sum(batch[i] == full, axis=1) / n_cols
    return out


def _batched_topk(
    source: np.ndarray | sparse.spmatrix,
    target: np.ndarray | sparse.spmatrix,
    top_k: int,
    batch_size: int,
    sim_fn,
) -> sparse.csr_matrix:
    """Compute a sparse similarity matrix by processing *source* in chunks.

    For each batch of rows from *source*, compute similarities against all
    rows of *target*, keep only the *top_k* per row, and assemble a CSR
    matrix.
    """
    n = source.shape[0]
    top_k = min(top_k, n)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = source[start:end]

        sims = sim_fn(batch, target)  # (batch_size, n) dense

        # Zero out self-similarity
        for local_i, global_i in enumerate(range(start, end)):
            sims[local_i, global_i] = 0.0

        # Top-k per row
        if top_k < sims.shape[1]:
            top_indices = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        else:
            top_indices = np.broadcast_to(
                np.arange(sims.shape[1]), (sims.shape[0], sims.shape[1])
            ).copy()

        batch_rows = np.repeat(np.arange(start, end), top_k)
        batch_cols = top_indices.ravel()
        batch_vals = sims[np.arange(sims.shape[0])[:, None], top_indices].ravel()

        mask = batch_vals > 0
        rows.append(batch_rows[mask])
        cols.append(batch_cols[mask])
        vals.append(batch_vals[mask])

    all_rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    all_cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    all_vals = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

    return sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))
