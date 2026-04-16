from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    """Abstract base class for all recommendation models."""

    @abstractmethod
    def fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> Recommender:
        """Build internal state from input tables.

        Parameters
        ----------
        customers : DataFrame
            Must contain a ``customer_id`` column plus metadata columns.
        products : DataFrame
            Must contain a ``product_id`` column plus metadata columns.
        interactions : DataFrame
            Must contain ``customer_id``, ``product_id``, and ``weight`` columns.
        """

    @abstractmethod
    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Return top-*n* product recommendations per customer.

        Parameters
        ----------
        customer_ids
            Subset of customers to score.  ``None`` means all customers
            seen during :meth:`fit`.
        n
            Number of products to return per customer.
        exclude_seen
            If ``True``, products the customer already interacted with are
            removed from the results.  Set to ``False`` to include them.

        Returns
        -------
        DataFrame
            Columns ``customer_id``, ``product_id``, ``score``, ``rank``.
        """
