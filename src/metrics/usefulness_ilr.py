# -*- coding: utf-8 -*-
# Author: Marie Griffon
# Date: 2026

"""Usefulness and ILR metrics implementation."""

__author__ = "Marie Griffon"

from itertools import combinations
from typing import Dict, Sequence, Tuple

import pandas as pd


def build_relation_maps(
    subst_df: pd.DataFrame, compl_df: pd.DataFrame
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Build lookup maps for substitutability and complementarity scores."""
    
    subst_map = {
        (str(int(row["item_x"])), str(int(row["item_y"]))): float(row["substitutability_score"])
        for _, row in subst_df.iterrows()
    }
    compl_map = {
        (str(int(row["item_x"])), str(int(row["item_y"]))): float(row["complementarity_score"])
        for _, row in compl_df.iterrows()
    }
    return subst_map, compl_map


def max_relation_score(
    item_a: str,
    item_b: str,
    subst_map: Dict[Tuple[str, str], float],
    compl_map: Dict[Tuple[str, str], float],
) -> float:
    """Return the max between subst and compl scores using prebuilt maps."""

    return max(subst_map.get((item_a, item_b), 0.0), compl_map.get((item_a, item_b), 0.0))

def _directional_usefulness(
    source_items: Sequence[str],
    target_items: Sequence[str],
    subst_map: Dict[Tuple[str, str], float],
    compl_map: Dict[Tuple[str, str], float],
) -> float:
    """Average over source items of the best relation to any target item."""

    if not source_items or not target_items:
        return 0.0

    best_scores = []
    for src in source_items:
        best = 0.0
        for tgt in target_items:
            best = max(best, max_relation_score(src, tgt, subst_map=subst_map, compl_map=compl_map))
        best_scores.append(best)
    return sum(best_scores) / float(len(source_items))


def usefulness(
    session_items: Sequence[str],
    recommended_items: Sequence[str],
    subst_map: Dict[Tuple[str, str], float],
    compl_map: Dict[Tuple[str, str], float],
) -> float:
    """Compute the harmonic mean usefulness between session and recommendation sets."""

    if not session_items or not recommended_items:
        return 0.0

    u_qr = _directional_usefulness(session_items, recommended_items, subst_map, compl_map)
    u_rq = _directional_usefulness(recommended_items, session_items, subst_map, compl_map)

    denominator = u_qr + u_rq
    if denominator == 0.0:
        return 0.0

    return (2.0 * u_qr * u_rq) / denominator


def ilr(
    recommended_items: Sequence[str],
    subst_map: Dict[Tuple[str, str], float],
    compl_map: Dict[Tuple[str, str], float],
) -> float:
    """Compute the Intra-List Relationships (ILR) metric for a recommendation list."""

    k = len(recommended_items)
    if k < 2:
        return 0.0

    total = 0.0
    for item_a, item_b in combinations(recommended_items, 2):
        total += max_relation_score(item_a, item_b, subst_map=subst_map, compl_map=compl_map)

    return (2.0 / (k * (k - 1))) * total


def usefulness_mean(
    sessions: Sequence[Sequence[str]],
    recommendations: Sequence[Sequence[str]],
    subst_map: Dict[Tuple[str, str], float],
    compl_map: Dict[Tuple[str, str], float],
) -> float:
    """Compute the mean usefulness over aligned lists of sessions and recommendations."""

    if len(sessions) != len(recommendations):
        raise ValueError("sessions and recommendations must have the same length.")

    scores = []
    for q_items, r_items in zip(sessions, recommendations):
        scores.append(usefulness(q_items, r_items, subst_map=subst_map, compl_map=compl_map))

    return sum(scores) / float(len(scores)) if scores else 0.0


def ilr_mean(
    recommendation_lists: Sequence[Sequence[str]],
    subst_map: Dict[Tuple[str, str], float],
    compl_map: Dict[Tuple[str, str], float],
) -> float:
    """Compute the mean ILR over multiple recommendation lists."""

    scores = [ilr(recs, subst_map=subst_map, compl_map=compl_map) for recs in recommendation_lists]

    return sum(scores) / float(len(scores)) if scores else 0.0
