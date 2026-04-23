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


def compute_substitutability_and_complementarity(
    out_df: pd.DataFrame,
    session_col: str = "session_id",
    item_col: str = "item_id",
    return_item_map: bool = True,
    factorize_items: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute both substitutability and complementarity scores with strict intersection or association requirement.
    It's possible to calculate both metrics at the level of item_id or other entity by changing item_col (e.g., "class_id").

    Definitions (set-based):
        For each item x, C_x is the SET of unique contexts c = (items in the session except x).
        
        Example: If sessions are {y, x, z}, {y, x, a}, then:
            C_x = {{y, z}, {y, a}}
        
        For a pair (x, y):
            cardi_inter_cx_cy = |C_x ∩ C_y|  (number of shared contexts)
            cardi_union_cx_cy = |C_x ∪ C_y| = |C_x| + |C_y| - |C_x ∩ C_y|
            cardi_assoc_xy    = |A_{x:y}| = |{ c ∈ C_x : y ∈ c }|  (contexts of x where y appears)
            cardi_assoc_yx    = |A_{y:x}| = |{ c ∈ C_y : x ∈ c }|  (contexts of y where x appears)
            
        Substitutability score:
                subst(x,y) = |C_x ∩ C_y| / ( |C_x ∪ C_y| + |A_{x:y}| + |A_{y:x}| )
                
        Association scores:
                assoc(x,y) = |A_{x:y}| / ( |C_x| + |C_x ∩ C_y| )    if |A_{x:y}| > 0
                assoc(y,x) = |A_{y:x}| / ( |C_y| + |C_x ∩ C_y| )    if |A_{y:x}| > 0
                
        Complementarity score (harmonic mean of directional associations):
                compl(x,y) = 2 * assoc(x,y) * assoc(y,x) / ( assoc(x,y) + assoc(y,x) )
                computed only when both directional associations are positive.
                
    This implementation:
      • Optionally factorizes item ids -> compact int codes (factorize_items=True)
      • Streams sessions once to build:
          - item_contexts[x] = set(context_id)
          - context_items[cid] = tuple(sorted item codes in the context)
          - focal_by_context[cid] = set(items x that have this context)
      • Then in a second pass over contexts:
          - cardi_inter[(x,y)]   += 1 for every unordered pair in focal_by_context[cid]
          - cardi_assoc[(x,y)]   += 1 for each x in focal_by_context[cid], y in context_items[cid]

    Parameters
    ----------
    out_df : DataFrame with [session_col, item_col]
    session_col : column name for sessions
    item_col    : column name for items (can be item_id, class_id, etc.)
    return_item_map : if True, also return a mapping DataFrame [item_code, item_id]
    factorize_items : if True (default), factorize item ids into compact codes; if False, assume `out_df[item_col]` already contains integer codes

    Returns
    -------
    substitutability_df : DataFrame with columns
      ['item_x','item_y','substitutability_score']
    
    complementarity_df : DataFrame with columns
      ['item_x','item_y','complementarity_score']
    
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

    # Data structures
    item_contexts: Dict[int, set] = defaultdict(set)        # x -> {cid, ...}
    context_items: Dict[int, Tuple[int, ...]] = dict()      # cid -> context tuple (y1, y2, ...)
    focal_by_context: Dict[int, set] = defaultdict(set)     # cid -> {x1, x2, ...}

    # 3) First pass: build unique contexts and focal_by_context while streaming the sessions
    for _, session_codes in grouped_sessions:
        S = sorted(set(session_codes.tolist()))
        m = len(S)
        if m <= 1:
            continue

        # Multi-item session: each item gets context "all others"
        for idx, x in enumerate(S):
            c_tuple = tuple(S[:idx] + S[idx+1:])
            cid = _context_id_from_tuple(c_tuple)
            if cid not in context_items:
                context_items[cid] = c_tuple
            if cid not in item_contexts[x]:
                item_contexts[x].add(cid)
                focal_by_context[cid].add(x)

    print(f"Built {len(item_contexts):,} C_item sets of contexts")
    print(f"Total unique contexts: {len(context_items):,}")
    
    # 4) Precompute |C_x| for all items
    contexts_cardinals = {x: len(cset) for x, cset in item_contexts.items()}

    # 5) Second pass over contexts to accumulate intersections and associations
    cardi_inter = defaultdict(int)        # (min(x,y), max(x,y)) -> shared_contexts
    cardi_assoc = defaultdict(int)        # (x,y) directed -> count of contexts of x that include y

    print("Computing intersections and associations...")
    for cid, focal_items in focal_by_context.items():
        if not focal_items:
            continue
        focal_list = sorted(focal_items)
        
        # 5a) intersections: items that share this context stored in canonical order
        if len(focal_list) >= 2:
            for x, y in itertools.combinations(focal_list, 2):  # combinations() always gives (x, y) where x < y
                cardi_inter[(x, y)] += 1
        
        # 5b) associations: for each focal x and each y present in the context
        c_members = context_items[cid]
        if c_members:
            for x in focal_list:
                for y in c_members:
                    cardi_assoc[(x, y)] += 1
    print("Computations done.")
    
    # 6) Compute substitutability scores as inter / (union + assoc_xy + assoc_yx) when inter > 0
    rows_subst = []
    for (x, y), cardi_inter_cx_cy in cardi_inter.items():
        if cardi_inter_cx_cy <= 0:
            continue
        cardi_cx = contexts_cardinals.get(x, 0)
        cardi_cy = contexts_cardinals.get(y, 0)
        cardi_union_cx_cy = cardi_cx + cardi_cy - cardi_inter_cx_cy
        cardi_a_xy = cardi_assoc.get((x, y), 0)
        cardi_a_yx = cardi_assoc.get((y, x), 0)
        denom = cardi_union_cx_cy + cardi_a_xy + cardi_a_yx
        if denom == 0:
            continue  # skip degenerate cases
        score = cardi_inter_cx_cy / denom
        rows_subst.append((x, y, score))

    substitutability_df = pd.DataFrame(rows_subst, columns=["item_x_code", "item_y_code", "substitutability_score"])
    
    # 7) Compute directional association scores assoc(x,y) where |A_{x:y}| > 0
    assoc_scores = {}  # (x,y) -> assoc(x,y)
    for (x, y), cardi_a_xy in cardi_assoc.items():
        if cardi_a_xy <= 0:
            continue
        cardi_cx = contexts_cardinals.get(x, 0)
        # intersection is stored with ordered key (min, max)
        inter_key = (x, y) if x < y else (y, x)
        cardi_inter_cx_cy = cardi_inter.get(inter_key, 0)
        denom = cardi_cx + cardi_inter_cx_cy
        if denom == 0:
            continue  # avoid division by zero
        assoc_scores[(x, y)] = cardi_a_xy / denom

    # 8) Compute complementarity scores from directional associations
    compl_scores = {}  # (min(x,y), max(x,y)) -> compl(x,y)
    for (x, y), assoc_xy in assoc_scores.items():
        assoc_yx = assoc_scores.get((y, x))
        if assoc_yx is None or assoc_yx <= 0:
            continue  # need both directions present and positive
        denom = assoc_xy + assoc_yx
        if denom == 0:
            continue
        compl = 2 * assoc_xy * assoc_yx / denom  # harmonic mean of the two directions
        pair = (x, y) if x < y else (y, x)
        compl_scores[pair] = compl

    complementarity_df = pd.DataFrame(
        [(x, y, s) for (x, y), s in compl_scores.items()],
        columns=["item_x_code", "item_y_code", "complementarity_score"],
    )
    print(f"Computing scores: {len(substitutability_df):,} substitutability pairs, {len(complementarity_df):,} complementarity pairs")

    # 9) Map codes back to original item ids (or keep original integer codes)
    if factorize_items:
        def de(code: int) -> str:
            return str(code_to_item.iloc[int(code)])

        if not substitutability_df.empty:
            substitutability_df["item_x"] = substitutability_df["item_x_code"].map(de)
            substitutability_df["item_y"] = substitutability_df["item_y_code"].map(de)
            substitutability_df = substitutability_df.drop(columns=["item_x_code", "item_y_code"])
            substitutability_df = substitutability_df[["item_x", "item_y", "substitutability_score"]]

        if not complementarity_df.empty:
            complementarity_df["item_x"] = complementarity_df["item_x_code"].map(de)
            complementarity_df["item_y"] = complementarity_df["item_y_code"].map(de)
            complementarity_df = complementarity_df.drop(columns=["item_x_code", "item_y_code"])
            complementarity_df = complementarity_df[["item_x", "item_y", "complementarity_score"]]
    else:
        # Items are already integer codes; expose them directly.
        if not substitutability_df.empty:
            substitutability_df = substitutability_df.rename(
                columns={"item_x_code": "item_x", "item_y_code": "item_y"}
            )[["item_x", "item_y", "substitutability_score"]]
        if not complementarity_df.empty:
            complementarity_df = complementarity_df.rename(
                columns={"item_x_code": "item_x", "item_y_code": "item_y"}
            )[["item_x", "item_y", "complementarity_score"]]

    item_map_df = None

    if return_item_map:
        if factorize_items:
            item_map_df = pd.DataFrame({
                "item_code": code_to_item.index.astype(np.int32),
                "item_id": code_to_item.values,
            })
        else:
            # Identity mapping for already-coded items
            codes = np.array(sorted(code_to_item.keys()), dtype=np.int64)
            item_map_df = pd.DataFrame({
                "item_code": codes,
                "item_id": codes,
            })
    
    return substitutability_df, complementarity_df, item_map_df
