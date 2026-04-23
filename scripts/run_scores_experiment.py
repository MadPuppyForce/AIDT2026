#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Marie Griffon
# Date: 2026

from __future__ import annotations

__author__ = "Marie Griffon"

from pathlib import Path
from itertools import combinations
from typing import Dict, List, Set, Tuple, Any
import time
from contextlib import contextmanager
from datetime import timedelta

import pandas as pd

from scores.substitution_complementarity_scores import compute_substitutability_and_complementarity
from scores.ic import compute_IC


# ----------------------------
# Config
# ----------------------------
PROCESSED_PATH = Path("data/processed")
DOMAINS = ["electronic", "clothing", "food"]
OUT_PATH = Path("results")

OUT_CSV = OUT_PATH / "scores_precision_recall_summary.csv"

Pair = Tuple[int, int]

# =========================
# Helpers
# =========================
def _fmt_seconds(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[time] {label}: {_fmt_seconds(dt)} ({dt:.3f}s)")
        
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _ensure_cols(df: pd.DataFrame, required: List[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns {missing}. Found={list(df.columns)}")


def _domain_title(domain: str) -> str:
    return {
        "electronic": "Electronic",
        "clothing": "Clothing",
        "food": "Food",
    }.get(domain, domain)


def _pairs_from_bundle_item(df_bundle_item: pd.DataFrame) -> Set[Pair]:
    """
    GT pairs = union over bundles of all 2-combinations of items in that bundle.
    """
    _ensure_cols(df_bundle_item, ["bundle ID", "item ID"], "df_bundle_item_train")

    bundles_items = (
        df_bundle_item.groupby("bundle ID")["item ID"]
        .apply(list)
        .reset_index()
    )
    bundles_items.columns = ["bundle ID", "items"]

    gt: Set[Pair] = set()
    for _, row in bundles_items.iterrows():
        items = row["items"]
        for x, y in combinations(items, 2):
            a, b = (x, y) if x <= y else (y, x)
            gt.add((int(a), int(b)))
    return gt


def _pairs_from_score_df(df_scores: pd.DataFrame) -> Set[Pair]:
    """
    Predicted pairs = set of (item_x, item_y) (normalized order).
    """
    _ensure_cols(df_scores, ["item_x", "item_y"], "df_scores")

    pairs: Set[Pair] = set()
    for _, row in df_scores.iterrows():
        x, y = int(row["item_x"]), int(row["item_y"])
        a, b = (x, y) if x <= y else (y, x)
        pairs.add((a, b))
    return pairs


def _metrics(gt: Set[Pair], pred: Set[Pair]) -> Dict[str, Any]:
    """
    Same core logic as calculate_precison_recall:
    precision = |pred ∩ gt| / |pred|
    recall    = |pred ∩ gt| / |gt|
    """
    inter = gt & pred
    n_gt = len(gt)
    n_pred = len(pred)
    n_correct = len(inter)

    precision = (n_correct / n_pred) if n_pred > 0 else 0.0
    recall = (n_correct / n_gt) if n_gt > 0 else 0.0

    return {
        "#GT pairs": n_gt,
        "#Pred. pairs": n_pred,
        "#Correct. pred. pairs": n_correct,
        "Precision": precision,  # fraction
        "Recall": recall,        # fraction
    }


def run_domain(domain: str) -> List[Dict[str, Any]]:

    with timed(f"domain={domain} TOTAL"):
        print(f"\n==================== DOMAIN: {domain} ====================")
        domain_dir = PROCESSED_PATH / domain

        df_session_item_train = _read_csv(domain_dir / "session_item_train.csv")
        df_bundle_item_train = _read_csv(domain_dir / "bundle_item_train.csv")
        df_bundle_item_test = _read_csv(domain_dir / "bundle_item_test.csv")  # loaded (not used for GT here)

        _ensure_cols(df_session_item_train, ["session ID", "item ID"], "df_session_item_train")
        _ = df_bundle_item_test

        gt_pairs = _pairs_from_bundle_item(df_bundle_item_train)

        print(f"\n[Our scores]")
        with timed(f"domain={domain} compute substitutability+complementarity"):
            subst_df, compl_df, _ = compute_substitutability_and_complementarity(
                df_session_item_train,
                session_col="session ID",
                item_col="item ID",
                return_item_map=False,
                factorize_items=False,
            )
            pred_our = _pairs_from_score_df(subst_df) | _pairs_from_score_df(compl_df)
            print(f"Number of unique predicted pairs: {len(pred_our):,}")


        (OUT_PATH / domain).mkdir(parents=True, exist_ok=True)
        subst_df.to_csv(OUT_PATH / domain / "substitutability_scores.csv", sep=",", index=False)
        compl_df.to_csv(OUT_PATH / domain / "complementarity_scores.csv", sep=",", index=False)


        print(f"\n[IC]")
        with timed(f"domain={domain} compute IC"):
            ic_df, _ = compute_IC(
                df_session_item_train,
                session_col="session ID",
                item_col="item ID",
                return_item_map=False,
                factorize_items=False,
            )

        pred_ic = _pairs_from_score_df(ic_df)

        inter_ic_our = pred_ic & pred_our
        print(f"\n[Intersection between the pairs of items predicted by the IC and by our scores]")
        print(f"Number of pairs in intersection: {len(inter_ic_our):,}")

        dom_name = _domain_title(domain)

        row_ic = {"Domain": dom_name, "Method": "IC"}
        row_ic.update(_metrics(gt_pairs, pred_ic))

        row_our = {"Domain": dom_name, "Method": "Our scores"}
        row_our.update(_metrics(gt_pairs, pred_our))

        return [row_ic, row_our]

def main() -> None:
    with timed("FULL RUN"):
        OUT_PATH.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for domain in DOMAINS:
            rows.extend(run_domain(domain))

        df = pd.DataFrame(
            rows,
            columns=[
                "Domain",
                "Method",
                "#GT pairs",
                "#Pred. pairs",
                "#Correct. pred. pairs",
                "Precision",
                "Recall",
            ],
        )

        df["Precision"] = (df["Precision"] * 100).map(lambda x: f"{x:.2f}%")
        df["Recall"] = (df["Recall"] * 100).map(lambda x: f"{x:.2f}%")
        df.to_csv(OUT_CSV, index=False)

        print("\n" + "=" * 90)
        print("Precision/Recall summary")
        print("=" * 90)
        print(df.to_string(index=False))

        print(f"\nSaved: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()