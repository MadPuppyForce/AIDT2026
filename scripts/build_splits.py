#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Marie Griffon
# Date: 2026

__author__ = "Marie Griffon"

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
RAW_PATH = Path("data/raw")       
OUT_PATH = Path("data/processed")
DOMAINS = ["electronic", "food", "clothing"]

TRAIN_SIZE = 0.8
SEED = 42

# Swap-optimizer hyperparams
MAX_ITER = 1500
CANDIDATE_PAIRS_PER_ITER = 300
WEIGHT_POWER = 1.5
VERBOSE = True


def improve_split_by_swaps(
    df: pd.DataFrame,
    train_sessions: np.ndarray,
    test_sessions: np.ndarray,
    session_col: str = "session ID",
    item_col: str = "item ID",
    random_state: int = 42,
    max_iter: int = 1500,
    candidate_pairs_per_iter: int = 300,
    weight_power: float = 1.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine a train/test session split to maximise item overlap between the two sets.

    Starting from an initial random split, the function iteratively proposes candidate
    session swaps (one train session ↔ one test session) and accepts the swap that
    most reduces the number of items seen exclusively in one split.  Rare items are
    penalised more heavily via an inverse-frequency weight raised to ``weight_power``,
    so the optimiser focuses on covering infrequent items in both sets.

    The split sizes are preserved: every accepted swap exchanges exactly one session
    from each side, keeping the train/test ratio constant.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction table with at least ``session_col`` and ``item_col`` columns.
    train_sessions, test_sessions : np.ndarray
        Initial session assignments produced by a random split.
    session_col, item_col : str
        Column names for sessions and items in *df*.
    random_state : int
        Seed for the random-number generator used to sample candidate pairs.
    max_iter : int
        Maximum number of swap rounds.
    candidate_pairs_per_iter : int
        Number of (train, test) session pairs evaluated per round.
    weight_power : float
        Exponent applied to item frequency when computing per-item weights
        (higher → rare items penalised more).
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    train_sessions, test_sessions : np.ndarray
        Optimised session arrays (same sizes as the inputs).
    """
    rng = np.random.default_rng(random_state)

    session_items = (
        df.groupby(session_col)[item_col]
          .apply(lambda x: set(x.unique()))
          .to_dict()
    )

    item_sessions = (
        df.groupby(item_col)[session_col]
          .apply(lambda x: set(x.unique()))
          .to_dict()
    )
    item_freq = {it: len(sset) for it, sset in item_sessions.items()}
    weights = {it: 1.0 / (max(1, f) ** weight_power) for it, f in item_freq.items()}

    train_set = set(train_sessions.tolist())
    test_set = set(test_sessions.tolist())

    def score_split(train_s: set, test_s: set):
        items_train = set().union(*(session_items.get(s, set()) for s in train_s)) if train_s else set()
        items_test = set().union(*(session_items.get(s, set()) for s in test_s)) if test_s else set()
        only_train = items_train - items_test
        only_test = items_test - items_train
        sc = sum(weights[it] for it in only_train) + sum(weights[it] for it in only_test)
        return sc

    initial_score = score_split(train_set, test_set)
    current_score = initial_score
    if verbose:
        print(f"[swap-opt] initial score (lower=better): {initial_score:.6f}")

    train_list = list(train_set)
    test_list = list(test_set)

    accepted = 0
    for _ in range(max_iter):
        best_delta = 0.0
        best_pair = None

        for _ in range(candidate_pairs_per_iter):
            s_tr = rng.choice(train_list)
            s_te = rng.choice(test_list)

            new_train = train_set.copy()
            new_test = test_set.copy()
            new_train.remove(s_tr); new_train.add(s_te)
            new_test.remove(s_te); new_test.add(s_tr)

            new_score = score_split(new_train, new_test)
            delta = current_score - new_score

            if delta > best_delta:
                best_delta = delta
                best_pair = (s_tr, s_te, new_score)

        if best_pair is None or best_delta <= 1e-12:
            break

        s_tr, s_te, new_score = best_pair
        train_set.remove(s_tr); train_set.add(s_te)
        test_set.remove(s_te); test_set.add(s_tr)

        train_list = list(train_set)
        test_list = list(test_set)

        current_score = new_score
        accepted += 1

        if verbose and (accepted % 25 == 0):
            improvement = (initial_score - current_score) / max(initial_score, 1e-12) * 100
            print(f"[swap-opt] swaps={accepted:4d}  score={current_score:.6f}  improvement={improvement:.1f}%")

    if verbose:
        improvement = (initial_score - current_score) / max(initial_score, 1e-12) * 100
        print(f"[swap-opt] done. accepted={accepted}, final score={current_score:.6f}, improvement={improvement:.1f}%")

    return np.array(list(train_set), dtype=object), np.array(list(test_set), dtype=object)


def compute_and_print_report(
    domain: str,
    df_session_item: pd.DataFrame,
    df_session_bundle: pd.DataFrame,
    df_bundle_item: pd.DataFrame,
    train_sessions_opt: np.ndarray,
    test_sessions_opt: np.ndarray,
) -> Dict[str, Any]:
    # Split DFs
    df_session_item_train = df_session_item[df_session_item["session ID"].isin(train_sessions_opt)].copy()
    df_session_item_test = df_session_item[df_session_item["session ID"].isin(test_sessions_opt)].copy()

    df_session_bundle_train = df_session_bundle[df_session_bundle["session ID"].isin(train_sessions_opt)].copy()
    df_session_bundle_test = df_session_bundle[df_session_bundle["session ID"].isin(test_sessions_opt)].copy()

    train_bundles = df_session_bundle_train["bundle ID"].unique()
    test_bundles = df_session_bundle_test["bundle ID"].unique()

    df_bundle_item_train = df_bundle_item[df_bundle_item["bundle ID"].isin(train_bundles)].copy()
    df_bundle_item_test = df_bundle_item[df_bundle_item["bundle ID"].isin(test_bundles)].copy()

    # Uniques
    unique_sessions = df_session_item["session ID"].unique()
    unique_items = df_session_item["item ID"].unique()

    unique_items_train = df_session_item_train["item ID"].unique()
    unique_items_test = df_session_item_test["item ID"].unique()

    # Densities (safe)
    def safe_density(nnz: int, n_items: int, n_sessions: int) -> float:
        denom = max(1, n_items) * max(1, n_sessions)
        return nnz / denom

    density_session_item_train = safe_density(len(df_session_item_train), len(unique_items_train), len(train_sessions_opt))
    density_session_item_test = safe_density(len(df_session_item_test), len(unique_items_test), len(test_sessions_opt))
    density_session_item = safe_density(len(df_session_item), len(unique_items), len(unique_sessions))

    density_bundle_item_train = safe_density(len(df_bundle_item_train), len(unique_items_train), len(df_session_bundle_train))
    density_bundle_item_test = safe_density(len(df_bundle_item_test), len(unique_items_test), len(df_session_bundle_test))
    density_bundle_item = safe_density(len(df_bundle_item), len(unique_items), len(df_session_bundle))

    # Item coverage
    items_train_set = set(unique_items_train.tolist())
    items_test_set = set(unique_items_test.tolist())
    n_shared = len(items_train_set & items_test_set)
    n_only_train = len(items_train_set - items_test_set)
    n_only_test = len(items_test_set - items_train_set)
    n_items_total = len(unique_items)

    # Bundle size stats
    bundle_counts = df_bundle_item.groupby("bundle ID")["item ID"].count()
    train_bundle_counts = df_bundle_item_train.groupby("bundle ID")["item ID"].count()
    test_bundle_counts = df_bundle_item_test.groupby("bundle ID")["item ID"].count()

    avg_bundle_size = float(bundle_counts.describe().get("mean", np.nan)) if len(bundle_counts) else float("nan")
    avg_train_bundle_size = float(train_bundle_counts.describe().get("mean", np.nan)) if len(train_bundle_counts) else float("nan")
    avg_test_bundle_size = float(test_bundle_counts.describe().get("mean", np.nan)) if len(test_bundle_counts) else float("nan")

    # Leakage
    test_bundle_items = set(df_bundle_item_test["item ID"].unique())
    test_bundle_items_not_in_train = test_bundle_items - items_train_set
    ratio_bundle = len(test_bundle_items_not_in_train) / max(1, len(test_bundle_items))

    test_items_not_in_train = items_test_set - items_train_set
    ratio_items = len(test_items_not_in_train) / max(1, len(items_test_set))

    # ── Comparative table ──────────────────────────────────────────────────────
    n_sessions_total = len(unique_sessions)
    n_train = len(train_sessions_opt)
    n_test = len(test_sessions_opt)
    train_pct = f"{n_train / n_sessions_total * 100:.1f}%"
    test_pct  = f"{n_test  / n_sessions_total * 100:.1f}%"

    W, C = 36, 12

    def _row(label, v_total, v_train, v_test):
        return f"  {label:<{W}} {str(v_total):>{C}} {str(v_train):>{C}} {str(v_test):>{C}}"

    sep = "  " + "-" * (W + 3 * (C + 1))

    print(f"\n{'=' * 70}")
    print(f"  DOMAIN: {domain.upper()}")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<{W}} {'total':>{C}} {'train':>{C}} {'test':>{C}}")
    print(sep)
    print(_row("#Items",                    n_items_total,                  len(unique_items_train),               len(unique_items_test)))
    print(_row("#Sessions",                 n_sessions_total,               f"{n_train} ({train_pct})",            f"{n_test} ({test_pct})"))
    print(_row("#Bundles",                  len(df_session_bundle),         len(df_session_bundle_train),          len(df_session_bundle_test)))
    print(_row("Avg bundle size",           f"{avg_bundle_size:.2f}",       f"{avg_train_bundle_size:.2f}",        f"{avg_test_bundle_size:.2f}"))
    print(sep)
    print(_row("Session-item interactions", len(df_session_item),           len(df_session_item_train),            len(df_session_item_test)))
    print(_row("Bundle-item interactions",  len(df_bundle_item),            len(df_bundle_item_train),             len(df_bundle_item_test)))
    print(_row("Density session-item",      f"{density_session_item*100:.2f}%",       f"{density_session_item_train*100:.2f}%",  f"{density_session_item_test*100:.2f}%"))
    print(_row("Density bundle-item",       f"{density_bundle_item*100:.2f}%",        f"{density_bundle_item_train*100:.2f}%",   f"{density_bundle_item_test*100:.2f}%"))

    # ── Item coverage ──────────────────────────────────────────────────────────
    print(f"\n  --- Item coverage ---")
    print(f"  {'Shared items (train ∩ test)':<{W}} {n_shared:>{C}}  ({n_shared / max(1, n_items_total) * 100:.1f}%)")
    print(f"  {'Only-train items':<{W}} {n_only_train:>{C}}  ({n_only_train / max(1, n_items_total) * 100:.1f}%)")
    print(f"  {'Only-test items':<{W}} {n_only_test:>{C}}  ({n_only_test / max(1, n_items_total) * 100:.1f}%)")

    print(f"\n  --- Test leakage ---")
    print(f"  {'Test bundle items not in train':<{W}} {len(test_bundle_items_not_in_train):>{C}}  ({ratio_bundle * 100:.2f}%)")
    print(f"  {'Test items not in train':<{W}} {len(test_items_not_in_train):>{C}}  ({ratio_items * 100:.2f}%)")

    return {
        "_df_session_item_train": df_session_item_train,
        "_df_session_item_test": df_session_item_test,
        "_df_session_bundle_train": df_session_bundle_train,
        "_df_session_bundle_test": df_session_bundle_test,
        "_df_bundle_item_train": df_bundle_item_train,
        "_df_bundle_item_test": df_bundle_item_test,
        "avg_bundle_size_total": avg_bundle_size,
        "avg_bundle_size_train": avg_train_bundle_size,
        "avg_bundle_size_test": avg_test_bundle_size,
        "n_items_total": n_items_total,
        "n_items_train": len(unique_items_train),
        "n_items_test": len(unique_items_test),
        "n_sessions_total": n_sessions_total,
        "n_sessions_train": n_train,
        "n_sessions_test": n_test,
        "shared_items": n_shared,
        "only_train_items": n_only_train,
        "only_test_items": n_only_test,
        "test_leakage_bundle_pct": ratio_bundle,
        "test_leakage_items_pct": ratio_items,
    }


def save_domain_outputs(
    domain: str,
    train_sessions_opt: np.ndarray,
    test_sessions_opt: np.ndarray,
    report: Dict[str, Any],
) -> None:
    out_domain = OUT_PATH / domain
    out_domain.mkdir(parents=True, exist_ok=True)

    # sessions
    pd.Series(train_sessions_opt, name="session ID").to_csv(out_domain / "train_sessions.csv", index=False)
    pd.Series(test_sessions_opt, name="session ID").to_csv(out_domain / "test_sessions.csv", index=False)

    # dataframes
    report["_df_session_item_train"].to_csv(out_domain / "session_item_train.csv", index=False)
    # report["_df_session_item_test"].to_csv(out_domain / "session_item_test.csv", index=False)
    # report["_df_session_bundle_train"].to_csv(out_domain / "session_bundle_train.csv", index=False)
    # report["_df_session_bundle_test"].to_csv(out_domain / "session_bundle_test.csv", index=False)
    report["_df_bundle_item_train"].to_csv(out_domain / "bundle_item_train.csv", index=False)
    report["_df_bundle_item_test"].to_csv(out_domain / "bundle_item_test.csv", index=False)


def main() -> None:
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    for domain in DOMAINS:
        print(f"\nProcessing domain: {domain} ...")
        domain_path = RAW_PATH / domain
        if not domain_path.exists():
            raise FileNotFoundError(f"Domain folder not found: {domain_path}")

        df_session_item = pd.read_csv(domain_path / "session_item.csv", sep=",")
        df_bundle_item = pd.read_csv(domain_path / "bundle_item.csv", sep=",")
        df_session_bundle = pd.read_csv(domain_path / "session_bundle.csv", sep=",")

        unique_sessions = df_session_item["session ID"].unique()

        train_sessions, test_sessions = train_test_split(
            unique_sessions,
            train_size=TRAIN_SIZE,
            random_state=SEED,
            shuffle=True,
        )
        train_sessions = np.array(train_sessions, dtype=object)
        test_sessions = np.array(test_sessions, dtype=object)

        train_sessions_opt, test_sessions_opt = improve_split_by_swaps(
            df_session_item,
            train_sessions,
            test_sessions,
            session_col="session ID",
            item_col="item ID",
            random_state=SEED,
            max_iter=MAX_ITER,
            candidate_pairs_per_iter=CANDIDATE_PAIRS_PER_ITER,
            weight_power=WEIGHT_POWER,
            verbose=VERBOSE,
        )

        # Sanity checks
        assert len(train_sessions_opt) == len(train_sessions)
        assert len(test_sessions_opt) == len(test_sessions)
        assert set(train_sessions_opt).isdisjoint(set(test_sessions_opt))
        assert len(set(train_sessions_opt) | set(test_sessions_opt)) == len(unique_sessions)

        report = compute_and_print_report(
            domain=domain,
            df_session_item=df_session_item,
            df_session_bundle=df_session_bundle,
            df_bundle_item=df_bundle_item,
            train_sessions_opt=train_sessions_opt,
            test_sessions_opt=test_sessions_opt,
        )
        save_domain_outputs(domain, train_sessions_opt, test_sessions_opt, report)


if __name__ == "__main__":
    main()