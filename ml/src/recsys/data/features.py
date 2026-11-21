"""Item side-information aligned to a split's item index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def genre_matrix(items: pd.DataFrame, item_ids: np.ndarray) -> tuple[sp.csr_matrix, list[str]] | None:
    """Multi-hot (n_items x n_genres) matrix, rows ordered like ``item_ids``. None if no genres."""
    if "genres" not in items.columns:
        return None
    per_item = items.set_index("item_id")["genres"].reindex(item_ids)
    vocab = sorted({g for gs in per_item if isinstance(gs, list) for g in gs})
    col = {g: j for j, g in enumerate(vocab)}
    rows, cols = [], []
    for r, gs in enumerate(per_item):
        if isinstance(gs, list):
            for g in gs:
                rows.append(r)
                cols.append(col[g])
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(item_ids), len(vocab)))
    return m, vocab
