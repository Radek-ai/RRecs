"""Classic recommendation baselines implementing :class:`recs.base.Recommender`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from recs.base import Recommender


class MostPopularBaseline(Recommender):
    """Recommend globally popular products (excluding already-seen)."""

    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> MostPopularBaseline:
        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)

        self._popular = (
            interactions.groupby("product_id")["customer_id"]
            .nunique()
            .sort_values(ascending=False)
            .index.tolist()
        )
        self._seen = (
            interactions.groupby("customer_id")["product_id"]
            .apply(set)
            .to_dict()
        )
        return self

    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        if customer_ids is None:
            customer_ids = self._customer_idx
        customer_ids = pd.Index(customer_ids)

        rows: list[dict] = []
        for cid in customer_ids:
            seen = self._seen.get(str(cid), set()) if exclude_seen else set()
            recs = [p for p in self._popular if p not in seen][:n]
            for rank, pid in enumerate(recs, 1):
                rows.append(
                    {
                        "customer_id": cid,
                        "product_id": pid,
                        "score": float(n - rank + 1),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(
            rows, columns=["customer_id", "product_id", "score", "rank"]
        )


class UserCFBaseline(Recommender):
    """User-based collaborative filtering (cosine similarity, top neighbours)."""

    def __init__(self, top_neighbors: int = 50) -> None:
        self.top_neighbors = top_neighbors
        self._customer_idx: pd.Index | None = None
        self._product_idx: pd.Index | None = None
        self._ui_matrix: sparse.csr_matrix | None = None
        self._cust_to_idx: dict | None = None
        self._idx_to_prod: dict | None = None
        self._n_items: int = 0

    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> UserCFBaseline:
        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)

        all_customers = sorted(interactions["customer_id"].unique())
        all_products = sorted(interactions["product_id"].unique())
        self._cust_to_idx = {c: i for i, c in enumerate(all_customers)}
        prod_to_idx = {p: i for i, p in enumerate(all_products)}
        self._idx_to_prod = {i: p for p, i in prod_to_idx.items()}
        self._n_items = len(all_products)

        row_ix = interactions["customer_id"].map(self._cust_to_idx).values
        col_ix = interactions["product_id"].map(prod_to_idx).values
        vals = interactions["weight"].values.astype(np.float32)
        n_users = len(all_customers)
        self._ui_matrix = sparse.csr_matrix(
            (vals, (row_ix, col_ix)), shape=(n_users, self._n_items)
        )
        return self

    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        assert self._ui_matrix is not None and self._cust_to_idx is not None

        if customer_ids is None:
            customer_ids = self._customer_idx
        customer_ids = pd.Index(customer_ids)

        n_users = self._ui_matrix.shape[0]
        rows: list[dict] = []

        for cid in customer_ids:
            if str(cid) not in self._cust_to_idx:
                continue
            uidx = self._cust_to_idx[str(cid)]

            sims = cosine_similarity(
                self._ui_matrix[uidx : uidx + 1], self._ui_matrix
            ).ravel()
            sims[uidx] = 0.0

            top_n = min(self.top_neighbors, max(1, len(sims) - 1))
            neigh = np.argpartition(sims, -top_n)[-top_n:]
            scores = np.zeros(self._n_items, dtype=np.float32)
            for ni in neigh:
                if sims[ni] > 0:
                    scores += sims[ni] * np.asarray(
                        self._ui_matrix[ni].todense()
                    ).ravel()

            if exclude_seen:
                seen_idx = self._ui_matrix[uidx].nonzero()[1]
                scores[seen_idx] = 0.0

            top_items = np.argsort(scores)[::-1][:n]
            for rank, pidx in enumerate(top_items, 1):
                if scores[pidx] <= 0:
                    break
                rows.append(
                    {
                        "customer_id": cid,
                        "product_id": self._idx_to_prod[pidx],
                        "score": float(scores[pidx]),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(
            rows, columns=["customer_id", "product_id", "score", "rank"]
        )


class ItemCFBaseline(Recommender):
    """Item-based collaborative filtering (item-item cosine similarity)."""

    def __init__(self) -> None:
        self._customer_idx: pd.Index | None = None
        self._product_idx: pd.Index | None = None
        self._ui_matrix: sparse.csr_matrix | None = None
        self._cust_to_idx: dict | None = None
        self._idx_to_prod: dict | None = None
        self._item_sim: np.ndarray | None = None
        self._n_items: int = 0

    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> ItemCFBaseline:
        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)

        all_customers = sorted(interactions["customer_id"].unique())
        all_products = sorted(interactions["product_id"].unique())
        self._cust_to_idx = {c: i for i, c in enumerate(all_customers)}
        prod_to_idx = {p: i for i, p in enumerate(all_products)}
        self._idx_to_prod = {i: p for p, i in prod_to_idx.items()}
        self._n_items = len(all_products)

        row_ix = interactions["customer_id"].map(self._cust_to_idx).values
        col_ix = interactions["product_id"].map(prod_to_idx).values
        vals = interactions["weight"].values.astype(np.float32)
        self._ui_matrix = sparse.csr_matrix(
            (vals, (row_ix, col_ix)),
            shape=(len(all_customers), self._n_items),
        )

        self._item_sim = cosine_similarity(self._ui_matrix.T)
        np.fill_diagonal(self._item_sim, 0)
        return self

    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        assert self._ui_matrix is not None and self._item_sim is not None

        if customer_ids is None:
            customer_ids = self._customer_idx
        customer_ids = pd.Index(customer_ids)

        rows: list[dict] = []
        for cid in customer_ids:
            if str(cid) not in self._cust_to_idx:
                continue
            uidx = self._cust_to_idx[str(cid)]
            user_items = self._ui_matrix[uidx].nonzero()[1]
            user_weights = np.asarray(
                self._ui_matrix[uidx, user_items].todense()
            ).ravel()

            scores = np.zeros(self._n_items, dtype=np.float32)
            for item_idx, w in zip(user_items, user_weights):
                scores += w * self._item_sim[item_idx]

            if exclude_seen:
                scores[user_items] = 0.0

            top_items = np.argsort(scores)[::-1][:n]
            for rank, pidx in enumerate(top_items, 1):
                if scores[pidx] <= 0:
                    break
                rows.append(
                    {
                        "customer_id": cid,
                        "product_id": self._idx_to_prod[pidx],
                        "score": float(scores[pidx]),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(
            rows, columns=["customer_id", "product_id", "score", "rank"]
        )


class SVDBaseline(Recommender):
    """Matrix factorisation via truncated SVD on the user-item matrix."""

    def __init__(self, n_components: int = 50, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._customer_idx: pd.Index | None = None
        self._product_idx: pd.Index | None = None
        self._ui_matrix: sparse.csr_matrix | None = None
        self._cust_to_idx: dict | None = None
        self._idx_to_prod: dict | None = None
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._n_items: int = 0

    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> SVDBaseline:
        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)

        all_customers = sorted(interactions["customer_id"].unique())
        all_products = sorted(interactions["product_id"].unique())
        self._cust_to_idx = {c: i for i, c in enumerate(all_customers)}
        prod_to_idx = {p: i for i, p in enumerate(all_products)}
        self._idx_to_prod = {i: p for p, i in prod_to_idx.items()}
        self._n_items = len(all_products)

        row_ix = interactions["customer_id"].map(self._cust_to_idx).values
        col_ix = interactions["product_id"].map(prod_to_idx).values
        vals = interactions["weight"].values.astype(np.float32)
        self._ui_matrix = sparse.csr_matrix(
            (vals, (row_ix, col_ix)),
            shape=(len(all_customers), self._n_items),
        )

        n_comp = min(self.n_components, self._ui_matrix.shape[1] - 1, self._ui_matrix.shape[0] - 1)
        n_comp = max(1, n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self._user_factors = svd.fit_transform(self._ui_matrix)
        self._item_factors = svd.components_.T
        return self

    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        assert (
            self._ui_matrix is not None
            and self._user_factors is not None
            and self._item_factors is not None
        )

        if customer_ids is None:
            customer_ids = self._customer_idx
        customer_ids = pd.Index(customer_ids)

        rows: list[dict] = []
        for cid in customer_ids:
            if str(cid) not in self._cust_to_idx:
                continue
            uidx = self._cust_to_idx[str(cid)]
            scores = self._user_factors[uidx] @ self._item_factors.T

            if exclude_seen:
                seen_idx = self._ui_matrix[uidx].nonzero()[1]
                scores[seen_idx] = -np.inf

            top_items = np.argsort(scores)[::-1][:n]
            rank = 0
            for pidx in top_items:
                if scores[pidx] <= -np.inf / 2:
                    break
                rank += 1
                rows.append(
                    {
                        "customer_id": cid,
                        "product_id": self._idx_to_prod[pidx],
                        "score": float(scores[pidx]),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(
            rows, columns=["customer_id", "product_id", "score", "rank"]
        )
