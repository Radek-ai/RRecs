"""Weighted-similarity recommendation model.

Scores are computed as:

    R = (w_cm * CSM + w_ci * ICSM) @ I @ (w_pm * PSM + w_pi * IPSM)

where all matrices are sparse (top-k neighbours only).  Any weight set to
zero skips the corresponding similarity computation entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recs.base import Recommender
from recs.similarity import (
    Metric,
    interaction_similarity,
    metadata_similarity,
)


def _zero_sparse(size: int) -> sparse.csr_matrix:
    return sparse.csr_matrix((size, size), dtype=np.float32)


class WeightedSimilarity(Recommender):
    """Blended metadata + interaction similarity recommender.

    Parameters
    ----------
    w_product_metadata : float
        Weight for the product metadata similarity matrix (PSM).
    w_product_interactions : float
        Weight for the interaction-based product similarity matrix (IPSM).
    w_customer_metadata : float
        Weight for the customer metadata similarity matrix (CSM).
    w_customer_interactions : float
        Weight for the interaction-based customer similarity matrix (ICSM).
    metric : ``"cosine"`` | ``"overlap"``
        Similarity metric.
    top_k : int
        Neighbours retained per row in every similarity matrix.
    """

    def __init__(
        self,
        w_product_metadata: float = 0.25,
        w_product_interactions: float = 0.25,
        w_customer_metadata: float = 0.25,
        w_customer_interactions: float = 0.25,
        metric: Metric = "cosine",
        top_k: int = 100,
    ) -> None:
        self.w_product_metadata = w_product_metadata
        self.w_product_interactions = w_product_interactions
        self.w_customer_metadata = w_customer_metadata
        self.w_customer_interactions = w_customer_interactions
        self.metric = metric
        self.top_k = top_k

        # Populated by fit()
        self._customer_idx: pd.Index | None = None
        self._product_idx: pd.Index | None = None
        self._R: sparse.csr_matrix | None = None
        self._interactions: sparse.csr_matrix | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> WeightedSimilarity:
        cust_meta_cols = [c for c in customers.columns if c != "customer_id"]
        prod_meta_cols = [c for c in products.columns if c != "product_id"]

        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)

        n = len(self._customer_idx)
        m = len(self._product_idx)

        # --- Interaction matrix I (n x m) ---
        row = self._customer_idx.get_indexer(interactions["customer_id"])
        col = self._product_idx.get_indexer(interactions["product_id"])
        weights = interactions["weight"].values.astype(np.float32)
        I = sparse.csr_matrix((weights, (row, col)), shape=(n, m))  # noqa: E741
        self._interactions = I

        # --- Product-side similarities (skip if weight is zero) ---
        if self.w_product_metadata:
            PSM, _ = metadata_similarity(
                products, "product_id", prod_meta_cols,
                metric=self.metric, top_k=self.top_k,
            )
        else:
            PSM = _zero_sparse(m)

        if self.w_product_interactions:
            IPSM, ipsm_idx = interaction_similarity(
                interactions, "product_id", "customer_id", "weight",
                metric=self.metric, top_k=self.top_k,
            )
            IPSM = self._reindex_sparse(IPSM, ipsm_idx, self._product_idx)
        else:
            IPSM = _zero_sparse(m)

        # --- Customer-side similarities (skip if weight is zero) ---
        if self.w_customer_metadata:
            CSM, _ = metadata_similarity(
                customers, "customer_id", cust_meta_cols,
                metric=self.metric, top_k=self.top_k,
            )
        else:
            CSM = _zero_sparse(n)

        if self.w_customer_interactions:
            ICSM, icsm_idx = interaction_similarity(
                interactions, "customer_id", "product_id", "weight",
                metric=self.metric, top_k=self.top_k,
            )
            ICSM = self._reindex_sparse(ICSM, icsm_idx, self._customer_idx)
        else:
            ICSM = _zero_sparse(n)

        # --- Score matrix ---
        # When all weights on a side are zero, use identity (= no blending)
        # so the other side can still contribute.
        if self.w_customer_metadata or self.w_customer_interactions:
            customer_blend = (
                self.w_customer_metadata * CSM
                + self.w_customer_interactions * ICSM
            )
            self._R = customer_blend @ I
        else:
            self._R = I

        if self.w_product_metadata or self.w_product_interactions:
            product_blend = (
                self.w_product_metadata * PSM
                + self.w_product_interactions * IPSM
            )
            self._R = self._R @ product_blend

        return self

    # ------------------------------------------------------------------
    # recommend
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
            scores = np.asarray(self._R[pos].todense()).ravel()

            if exclude_seen:
                seen = self._interactions[pos].nonzero()[1]
                scores[seen] = 0.0

            # Top-n
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

        return pd.DataFrame(rows, columns=["customer_id", "product_id", "score", "rank"])

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reindex_sparse(
        mat: sparse.csr_matrix,
        old_idx: pd.Index,
        new_idx: pd.Index,
    ) -> sparse.csr_matrix:
        """Expand/reorder a square sparse matrix to match *new_idx*."""
        if old_idx.equals(new_idx):
            return mat

        mapping = new_idx.get_indexer(old_idx)
        n = len(new_idx)
        coo = mat.tocoo()

        valid = (mapping[coo.row] >= 0) & (mapping[coo.col] >= 0)
        new_row = mapping[coo.row[valid]]
        new_col = mapping[coo.col[valid]]

        return sparse.csr_matrix(
            (coo.data[valid], (new_row, new_col)), shape=(n, n)
        )
