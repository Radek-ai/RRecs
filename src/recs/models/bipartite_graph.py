"""Bipartite-graph hop recommender.

Scores products by counting / weighting multi-hop paths in the
customer-product bipartite graph:

    R = sum(w_k * C^((k-1)/2) @ I   for each odd k in hop_weights)

where C = I @ I^T (customer co-purchase matrix).

Supports strict group filtering and soft per-feature penalties on
the intermediate customer / product nodes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from recs.base import Recommender


class BipartiteGraphRec(Recommender):
    """Graph-based recommender using multi-hop bipartite paths.

    Parameters
    ----------
    hop_weights : dict[int, float]
        Mapping of odd hop counts to their blending weights,
        e.g. ``{3: 1.0, 5: 0.3}``.
    aggregation : ``"multiply"`` | ``"count"``
        ``"multiply"`` uses raw interaction weights along each path
        (path score = product of edge weights; implemented via
        :math:`C^{(k-1)/2} I`).
        ``"count"`` binarises the interaction matrix so every path
        contributes exactly 1 (same matrix formulation).
    top_k : int
        Neighbours retained per row in C and P matrices.
    customer_group_cols : list[str] | None
        Columns defining strict customer groups.  Paths cannot
        traverse customers from different groups.
    product_group_cols : list[str] | None
        Columns defining strict product groups.
    customer_penalties : dict[str, float] | float | None
        Soft penalties for crossing customer feature boundaries.
        A dict maps feature columns to penalty factors (0-1).
        A single float applies the same penalty to every metadata
        column.  ``None`` disables penalties.
    product_penalties : dict[str, float] | float | None
        Same as *customer_penalties* but for products.
    cold_start_n : int | None
        If set, customers with **no** interactions get recommendations by
        pretending their basket is the top-*cold_start_n* products by
        **sum of interaction weights** (global).  If *customer_group_cols*
        is set, popularity is computed **within each group** first; if that
        group has no data, **global** popularity is used.  Synthetic basket
        entries are treated like real ones for *exclude_seen*.  Only hop
        keys **1** and **3** from *hop_weights* contribute to the cold-start
        score (5/7-hop terms are skipped for that user).
    """

    def __init__(
        self,
        hop_weights: dict[int, float] | None = None,
        aggregation: Literal["multiply", "count"] = "multiply",
        top_k: int = 100,
        customer_group_cols: list[str] | None = None,
        product_group_cols: list[str] | None = None,
        customer_penalties: dict[str, float] | float | None = None,
        product_penalties: dict[str, float] | float | None = None,
        cold_start_n: int | None = None,
    ) -> None:
        self.hop_weights = hop_weights or {3: 1.0, 5: 0.3}
        self.aggregation = aggregation
        self.top_k = top_k
        self.customer_group_cols = customer_group_cols
        self.product_group_cols = product_group_cols
        self.customer_penalties = customer_penalties
        self.product_penalties = product_penalties
        self.cold_start_n = cold_start_n

        self._customer_idx: pd.Index | None = None
        self._product_idx: pd.Index | None = None
        self._R: sparse.csr_matrix | None = None
        self._interactions: sparse.csr_matrix | None = None
        self._cold_pop_global: np.ndarray | None = None
        self._cold_pop_by_group: dict[str, np.ndarray] | None = None
        self._customer_group_keys: pd.Series | None = None
        self._I_graph: sparse.csr_matrix | None = None

        for k in self.hop_weights:
            if k < 1 or k % 2 == 0:
                raise ValueError(
                    f"hop_weights keys must be positive odd integers, got {k}"
                )

        if aggregation not in ("multiply", "count"):
            raise ValueError(
                f"aggregation must be 'multiply' or 'count', got {aggregation!r}"
            )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> BipartiteGraphRec:
        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)

        n = len(self._customer_idx)
        m = len(self._product_idx)

        # --- Build interaction matrix I (n x m) ---
        row = self._customer_idx.get_indexer(interactions["customer_id"])
        col = self._product_idx.get_indexer(interactions["product_id"])
        weights = interactions["weight"].values.astype(np.float32)

        I = sparse.csr_matrix((weights, (row, col)), shape=(n, m))  # noqa: E741
        self._interactions = I

        if self.cold_start_n is not None and self.cold_start_n > 0:
            pop_g, pop_by_g, gkeys = _cold_popularity_tables(
                interactions,
                customers,
                self._customer_idx,
                self._product_idx,
                self.customer_group_cols,
            )
            self._cold_pop_global = pop_g
            self._cold_pop_by_group = pop_by_g
            self._customer_group_keys = gkeys

        if self.aggregation == "count":
            I = (I > 0).astype(np.float32)  # noqa: E741

        self._I_graph = I

        # --- Co-occurrence matrices ---
        C = _sparse_topk(I @ I.T, self.top_k, zero_diag=True)   # n x n
        P = _sparse_topk(I.T @ I, self.top_k, zero_diag=True)   # m x m

        # --- Strict group filtering ---
        if self.customer_group_cols:
            mask = _group_mask(customers, "customer_id",
                               self.customer_group_cols, self._customer_idx)
            C = C.multiply(mask)

        if self.product_group_cols:
            mask = _group_mask(products, "product_id",
                               self.product_group_cols, self._product_idx)
            P = P.multiply(mask)

        # --- Soft penalties ---
        if self.customer_penalties is not None:
            pen = _penalty_matrix(
                C, customers, "customer_id",
                self.customer_penalties, self._customer_idx,
            )
            C = C.multiply(pen)

        if self.product_penalties is not None:
            pen = _penalty_matrix(
                P, products, "product_id",
                self.product_penalties, self._product_idx,
            )
            P = P.multiply(pen)

        C = C.tocsr()
        P = P.tocsr()

        # --- Multi-hop score accumulation ---
        # k-hop = C^((k-1)/2) @ I  for odd k
        # We compute C powers incrementally.
        max_hop = max(self.hop_weights)
        max_c_power = (max_hop - 1) // 2

        # Precompute needed powers of C
        c_powers: dict[int, sparse.csr_matrix] = {}
        if max_c_power >= 1:
            c_powers[1] = C
        for p in range(2, max_c_power + 1):
            c_powers[p] = _sparse_topk(c_powers[p - 1] @ C, self.top_k)

        self._R = sparse.csr_matrix((n, m), dtype=np.float32)

        for k, w in self.hop_weights.items():
            if w == 0:
                continue
            c_exp = (k - 1) // 2
            if c_exp == 0:
                R_k = I  # 1-hop = direct interactions
            else:
                R_k = c_powers[c_exp] @ I
            self._R = self._R + w * R_k

        return self

    # ------------------------------------------------------------------
    # recommend  (identical logic to WeightedSimilarity)
    # ------------------------------------------------------------------

    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        if self._R is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if customer_ids is None:
            customer_ids = self._customer_idx
        customer_ids = pd.Index(customer_ids)

        idx = self._customer_idx.get_indexer(customer_ids)
        if (idx == -1).any():
            unknown = customer_ids[idx == -1].tolist()
            raise KeyError(f"Unknown customer_ids: {unknown}")

        rows: list[dict] = []
        for pos, cid in zip(idx, customer_ids):
            cold_pack = _maybe_cold_start_scores(self, pos)
            if cold_pack is not None:
                scores, synth_idx = cold_pack
            else:
                scores = np.asarray(self._R[pos].todense()).ravel()
                synth_idx = None

            if exclude_seen:
                seen = self._interactions[pos].nonzero()[1]
                scores[seen] = 0.0
                if synth_idx is not None:
                    scores[synth_idx] = 0.0

            if n < len(scores):
                top = np.argpartition(scores, -n)[-n:]
                top = top[np.argsort(scores[top])[::-1]]
            else:
                top = np.argsort(scores)[::-1][:n]

            for rank, pidx in enumerate(top, 1):
                if scores[pidx] <= 0:
                    break
                rows.append(
                    {
                        "customer_id": cid,
                        "product_id": self._product_idx[pidx],
                        "score": scores[pidx],
                        "rank": rank,
                    }
                )

        return pd.DataFrame(
            rows, columns=["customer_id", "product_id", "score", "rank"]
        )


# ======================================================================
# Helpers
# ======================================================================


def _cold_popularity_tables(
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    customer_idx: pd.Index,
    product_idx: pd.Index,
    customer_group_cols: list[str] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, pd.Series | None]:
    """Per-product sum of weights (global and optional per customer group)."""
    m = len(product_idx)
    pop_global = np.zeros(m, dtype=np.float64)
    col = product_idx.get_indexer(interactions["product_id"])
    w = interactions["weight"].values.astype(np.float64)
    ok = col >= 0
    np.add.at(pop_global, col[ok], w[ok])

    by_group: dict[str, np.ndarray] | None = None
    gkeys: pd.Series | None = None
    if customer_group_cols:
        df_c = customers.set_index("customer_id").reindex(customer_idx)
        gkeys = df_c[customer_group_cols].astype(str).agg("|".join, axis=1)
        merged = interactions.merge(
            customers[["customer_id", *customer_group_cols]],
            on="customer_id",
            how="left",
        )
        keys = merged[customer_group_cols].astype(str).agg("|".join, axis=1)
        by_group = {}
        for gk in keys.dropna().unique():
            sub = merged[keys == gk]
            arr = np.zeros(m, dtype=np.float64)
            c2 = product_idx.get_indexer(sub["product_id"])
            w2 = sub["weight"].values.astype(np.float64)
            ok2 = c2 >= 0
            np.add.at(arr, c2[ok2], w2[ok2])
            by_group[str(gk)] = arr

    return pop_global, by_group, gkeys


def _maybe_cold_start_scores(
    model: BipartiteGraphRec,
    pos: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Scores from synthetic popular basket for interaction-less users; else None."""
    if model.cold_start_n is None or model.cold_start_n <= 0:
        return None
    if model._interactions.getrow(pos).nnz > 0:
        return None
    if model._cold_pop_global is None or model._I_graph is None:
        return None

    m = model._product_idx.size
    n_take = min(model.cold_start_n, m)
    pop = model._cold_pop_global
    if model._customer_group_keys is not None and model._cold_pop_by_group:
        gk = str(model._customer_group_keys.iloc[pos])
        if gk in model._cold_pop_by_group:
            gpop = model._cold_pop_by_group[gk]
            if np.any(gpop > 0):
                pop = gpop

    if np.max(pop) <= 0:
        return None

    top_idx = np.argpartition(pop, -n_take)[-n_take:]
    top_idx = top_idx[np.argsort(pop[top_idx])[::-1]]

    if model.aggregation == "count":
        data = np.ones(len(top_idx), dtype=np.float32)
    else:
        data = pop[top_idx].astype(np.float32)

    rows = np.zeros(len(top_idx), dtype=np.int32)
    s = sparse.csr_matrix((data, (rows, top_idx)), shape=(1, m))
    I_g = model._I_graph
    C_row = s @ I_g.T

    scores = np.zeros(m, dtype=np.float64)
    hw = model.hop_weights
    if 1 in hw and hw[1] != 0:
        scores += float(hw[1]) * s.toarray().ravel()
    if 3 in hw and hw[3] != 0:
        contrib = (C_row @ I_g).toarray().ravel()
        scores += float(hw[3]) * contrib

    return scores.astype(np.float32), top_idx


def _sparse_topk(
    mat: sparse.spmatrix,
    top_k: int,
    zero_diag: bool = False,
) -> sparse.csr_matrix:
    """Keep only the *top_k* largest values per row (sparse)."""
    mat = mat.tocsr()
    n = mat.shape[0]

    if zero_diag and mat.shape[0] == mat.shape[1]:
        mat = mat.copy()
        mat.setdiag(0)
        mat.eliminate_zeros()

    if top_k >= mat.shape[1]:
        return mat

    rows, cols, vals = [], [], []
    for i in range(n):
        row_start, row_end = mat.indptr[i], mat.indptr[i + 1]
        row_data = mat.data[row_start:row_end]
        row_indices = mat.indices[row_start:row_end]

        if len(row_data) <= top_k:
            rows.extend([i] * len(row_data))
            cols.extend(row_indices)
            vals.extend(row_data)
        else:
            top_idx = np.argpartition(row_data, -top_k)[-top_k:]
            rows.extend([i] * top_k)
            cols.extend(row_indices[top_idx])
            vals.extend(row_data[top_idx])

    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=mat.shape, dtype=np.float32
    )


def _group_mask(
    entity_df: pd.DataFrame,
    id_col: str,
    group_cols: list[str],
    idx: pd.Index,
) -> sparse.csr_matrix:
    """Build a sparse {0,1} mask where mask[i,j]=1 iff entities i,j share
    the same composite group key."""
    df = entity_df.set_index(id_col).reindex(idx)
    keys = df[group_cols].astype(str).agg("|".join, axis=1)
    codes = pd.Categorical(keys).codes

    n = len(idx)
    rows, cols = [], []
    # Group entities by code and generate within-group pairs
    group_members: dict[int, list[int]] = {}
    for i, c in enumerate(codes):
        group_members.setdefault(c, []).append(i)

    for members in group_members.values():
        for i in members:
            for j in members:
                if i != j:
                    rows.append(i)
                    cols.append(j)

    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _penalty_matrix(
    reference: sparse.spmatrix,
    entity_df: pd.DataFrame,
    id_col: str,
    penalties: dict[str, float] | float,
    idx: pd.Index,
) -> sparse.csr_matrix:
    """Build a sparse penalty matrix aligned with *reference*'s non-zero
    structure.  Only computes penalties for existing non-zero entries."""
    df = entity_df.set_index(id_col).reindex(idx)

    if isinstance(penalties, (int, float)):
        feature_cols = [c for c in df.columns if c != id_col]
        penalties = {c: float(penalties) for c in feature_cols}

    ref_coo = reference.tocoo()
    pen_values = np.ones(len(ref_coo.data), dtype=np.float32)

    for col, penalty_val in penalties.items():
        if col not in df.columns:
            continue
        col_values = df[col].values
        differs = col_values[ref_coo.row] != col_values[ref_coo.col]
        pen_values[differs] *= penalty_val

    return sparse.csr_matrix(
        (pen_values, (ref_coo.row, ref_coo.col)),
        shape=reference.shape,
    )
