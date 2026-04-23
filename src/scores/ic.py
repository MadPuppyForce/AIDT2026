# -*- coding: utf-8 -*-
# Author: Marie Griffon
# Date: 2026

__author__ = "Marie Griffon"

import numpy as np
import pandas as pd
import hashlib
import itertools
from collections import defaultdict
from typing import Dict, Tuple, Optional, Callable

def _context_id_from_tuple(t: Tuple[int, ...]) -> int:
    """
    Stable 64-bit hash for a context tuple of item codes.
    """
    h = hashlib.blake2b(digest_size=8)
    # pack as bytes: join on comma; item codes are ints so cast to str
    h.update((",".join(map(str, t))).encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)

def compute_IC(
    out_df: pd.DataFrame,
    session_col: str = "session_id",
    item_col: str = "item_id",
    return_item_map: bool = True,
    factorize_items: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute item co-occurrence (lift): IC(i, j) = P(e_i ∩ e_j) / (P(e_i) * P(e_j)),
    where probabilities are computed over sessions after de-duplicating items within each session.

    Returns
    -------
    IC_df : DataFrame with columns
      ['item_x','item_y','IC']
    
    item_map_df : optional DataFrame mapping integer codes to original item ids
    """
    # 0) Keep only needed columns and drop exact duplicates
    df = (
        out_df[[session_col, item_col]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    # 1) Build internal integer codes (_icode)
    # By default we factorize item ids -> compact int codes for performance.
    # If your items are already integer codes in `out_df[item_col]`, you can skip factorization.
    if factorize_items:
        df[item_col] = df[item_col].astype("string")
        item_codes, uniques = pd.factorize(df[item_col], sort=False)
        df = df.assign(_icode=item_codes.astype(np.int32))
        # code_to_item maps internal compact codes -> original item ids
        code_to_item = pd.Series(uniques)
    else:
        # Use existing integer codes directly (no re-encoding). These codes may be non-contiguous.
        df[item_col] = pd.to_numeric(df[item_col], errors="raise").astype(np.int64)
        df = df.assign(_icode=df[item_col].astype(np.int64))
        # code_to_item maps codes -> themselves (identity); stored as dict for O(1) lookup
        code_to_item = {int(v): int(v) for v in df["_icode"].unique().tolist()}

    # 2) Deduplicate within-session and make per-session sets of item codes on the fly
    grouped_sessions = df.groupby(session_col)["_icode"]
    n_sessions = grouped_sessions.ngroups
    print(f"Processing {n_sessions:,} sessions...")

    # Counters
    item_session_counts: Dict[int, int] = defaultdict(int)      # x -> |{s : x in s}|
    pair_session_counts: Dict[Tuple[int, int], int] = defaultdict(int)  # (x,y) unordered -> |{s : x,y in s}|

    # 3) Single pass over sessions to accumulate item and pair supports
    for _, session_items in grouped_sessions:
        items_list = sorted(set(session_items.tolist()))  # unique items per session
        if not items_list:
            continue

        # item supports
        for item in items_list:
            item_session_counts[item] += 1

        # pair supports
        if len(items_list) >= 2:
            for item_x, item_y in itertools.combinations(items_list, 2):  # guarantees x < y
                pair_session_counts[(item_x, item_y)] += 1

    print(f"Unique items: {len(item_session_counts):,}")
    # print(f"Co-occurring pairs observed: {len(pair_session_counts):,}")

    # 4) Compute IC = P(e_i ∩ e_j) / (P(e_i) * P(e_j)) with probabilities over sessions
    n_sessions_float = float(n_sessions)
    rows = []
    for (item_x, item_y), count in pair_session_counts.items():
        p_xy = count / n_sessions_float
        p_x = item_session_counts.get(item_x, 0) / n_sessions_float
        p_y = item_session_counts.get(item_y, 0) / n_sessions_float
        denom = p_x * p_y
        if denom <= 0:
            continue
        ic = p_xy / denom
        rows.append((item_x, item_y, ic))

    ic_df = pd.DataFrame(rows, columns=["item_x_code", "item_y_code", "IC"])

    # 5) Map codes back to original item ids (or keep original integer codes)
    if factorize_items:
        def de(code: int) -> str:
            return str(code_to_item.iloc[int(code)])

        if not ic_df.empty:
            ic_df["item_x"] = ic_df["item_x_code"].map(de)
            ic_df["item_y"] = ic_df["item_y_code"].map(de)
            ic_df = ic_df.drop(columns=["item_x_code", "item_y_code"])
            ic_df = ic_df[["item_x", "item_y", "IC"]]
    else:
        if not ic_df.empty:
            ic_df = ic_df.rename(columns={"item_x_code": "item_x", "item_y_code": "item_y"})[
                ["item_x", "item_y", "IC"]
            ]

    # 5b) Relationship summary
    if not ic_df.empty:
        n_subst = int((ic_df["IC"] < 1).sum())
        n_compl = int((ic_df["IC"] > 1).sum())
        n_indep = int((ic_df["IC"] == 1).sum())
        print(f"Computing IC: {n_subst:,} substitutability pairs, {n_compl:,} complementarity pairs, {n_indep:,} independence pairs")

    # 6) Optional mapping table
    item_map_df = None
    if return_item_map:
        if factorize_items:
            item_map_df = pd.DataFrame({
                "item_code": code_to_item.index.astype(np.int32),
                "item_id": code_to_item.values,
            })
        else:
            codes = np.array(sorted(code_to_item.keys()), dtype=np.int64)
            item_map_df = pd.DataFrame({
                "item_code": codes,
                "item_id": codes,
            })

    return ic_df, item_map_df
