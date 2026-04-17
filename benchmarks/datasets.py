"""Dataset loaders for benchmarking.

Each loader returns a :class:`BenchmarkData` instance containing the
train/test split and metadata tables ready for model consumption.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass

import pandas as pd


@dataclass
class BenchmarkData:
    """Standardised container returned by every dataset loader."""

    customers: pd.DataFrame       # customer_id + metadata columns
    products: pd.DataFrame        # product_id + metadata columns
    train: pd.DataFrame           # customer_id, product_id, weight
    ground_truth: dict[str, set]  # customer_id -> {test product_ids}
    eval_customers: list[str]     # customers present in both train and test
    name: str = ""                # dataset identifier


def load_online_retail(
    data_dir: str = "benchmarks/data",
    min_interactions: int = 5,
    test_frac: float = 0.2,
) -> BenchmarkData:
    """Download, clean, and split the UCI Online Retail dataset.

    Parameters
    ----------
    data_dir : str
        Directory to cache the downloaded xlsx file.
    min_interactions : int
        Minimum distinct products per customer (and vice-versa) after
        iterative filtering.
    test_frac : float
        Fraction of each customer's interactions (by time) held out for
        testing.
    """
    os.makedirs(data_dir, exist_ok=True)
    xlsx_path = os.path.join(data_dir, "Online Retail.xlsx")

    if not os.path.exists(xlsx_path):
        url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
        zip_path = os.path.join(data_dir, "online_retail.zip")
        print("Downloading UCI Online Retail dataset …")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        os.remove(zip_path)
        print("Done.")

    raw = pd.read_excel(xlsx_path, engine="openpyxl")

    # ---- Cleaning ----
    df = raw.copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["Spend"] = df["Quantity"] * df["UnitPrice"]

    # ---- Aggregate to customer x product ----
    interactions_full = (
        df.groupby(["CustomerID", "StockCode"])
        .agg(weight=("Spend", "sum"), last_date=("InvoiceDate", "max"))
        .reset_index()
    )

    # ---- Iterative min-interaction filter ----
    for _ in range(3):
        cc = interactions_full.groupby("CustomerID")["StockCode"].nunique()
        pc = interactions_full.groupby("StockCode")["CustomerID"].nunique()
        interactions_full = interactions_full[
            interactions_full["CustomerID"].isin(cc[cc >= min_interactions].index)
            & interactions_full["StockCode"].isin(pc[pc >= min_interactions].index)
        ]

    # ---- Metadata tables ----
    kept_custs = interactions_full["CustomerID"].unique()
    kept_prods = interactions_full["StockCode"].unique()

    customers = (
        df[df["CustomerID"].isin(kept_custs)]
        .groupby("CustomerID")["Country"]
        .first()
        .reset_index()
        .rename(columns={"CustomerID": "customer_id", "Country": "country"})
    )

    prod_desc = (
        df[df["StockCode"].isin(kept_prods)]
        .groupby("StockCode")["Description"]
        .first()
        .fillna("UNKNOWN")
        .reset_index()
    )
    prod_desc["category"] = (
        prod_desc["Description"].str.split().str[0].str.lower().fillna("unknown")
    )
    top_cats = prod_desc["category"].value_counts().head(30).index
    prod_desc.loc[~prod_desc["category"].isin(top_cats), "category"] = "other"
    products = prod_desc[["StockCode", "category"]].rename(
        columns={"StockCode": "product_id"}
    )

    # ---- Time-based train / test split ----
    interactions_dated = interactions_full.sort_values("last_date")
    train_rows, test_rows = [], []
    for _, grp in interactions_dated.groupby("CustomerID"):
        n = len(grp)
        split = max(1, int(n * (1 - test_frac)))
        train_rows.append(grp.iloc[:split])
        if split < n:
            test_rows.append(grp.iloc[split:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    train = train_df[["CustomerID", "StockCode", "weight"]].rename(
        columns={"CustomerID": "customer_id", "StockCode": "product_id"}
    )
    ground_truth = (
        test_df.groupby("CustomerID")["StockCode"].apply(set).to_dict()
    )
    eval_customers = sorted(ground_truth.keys())

    print(
        f"Online Retail: {len(customers)} customers, {len(products)} products, "
        f"{len(train)} train, {len(test_df)} test, {len(eval_customers)} eval"
    )

    return BenchmarkData(
        customers=customers,
        products=products,
        train=train,
        ground_truth=ground_truth,
        eval_customers=eval_customers,
        name="online_retail",
    )
