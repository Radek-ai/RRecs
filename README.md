# recs

Modular, extensible recommendation systems library.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import pandas as pd
from recs import WeightedSimilarity

customers = pd.DataFrame({
    "customer_id": ["c0", "c1", "c2"],
    "region":      ["north", "south", "north"],
    "segment":     ["premium", "budget", "mid"],
})

products = pd.DataFrame({
    "product_id": ["p0", "p1", "p2", "p3"],
    "category":   ["electronics", "books", "electronics", "books"],
})

interactions = pd.DataFrame({
    "customer_id": ["c0", "c0", "c1", "c2"],
    "product_id":  ["p0", "p1", "p2", "p3"],
    "weight":      [5.0,  3.0,  4.0,  1.0],
})

model = WeightedSimilarity(
    a=0.25, b=0.25,   # product-side weights (PSM, IPSM)
    c=0.25, d=0.25,   # customer-side weights (CSM, ICSM)
    metric="cosine",   # or "overlap"
    top_k=100,         # neighbours per row in similarity matrices
)

model.fit(customers, products, interactions)
recs = model.recommend(customer_ids=["c0", "c1"], n=5)
print(recs)
#   customer_id product_id     score  rank
# 0          c0         p2  1.234567     1
# 1          c0         p3  0.987654     2
# ...
```

## How it works

Three input tables (customers, products, interactions) produce five matrices:

| Matrix | Shape | Source |
|--------|-------|--------|
| **I**    | n x m | interaction weights |
| **PSM**  | m x m | product metadata similarity |
| **CSM**  | n x n | customer metadata similarity |
| **IPSM** | m x m | product interaction-profile similarity |
| **ICSM** | n x n | customer interaction-profile similarity |

Recommendation scores:

```
R = (c * CSM + d * ICSM)  @  I  @  (a * PSM + b * IPSM)
         n x n              n x m        m x m
                      =  n x m
```

All similarity matrices are **sparse** (top-k neighbours per row), so memory
and compute scale with `k * n` rather than `n^2`.

## Adding new models

Create a class in `src/recs/models/` that inherits from `recs.Recommender` and
implements `fit()` and `recommend()`.
