#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Marie Griffon
# Date: 2026

"""
Run Experiment #2: ILR + Usefulness robustness under controlled corruption.

What this script does
---------------------
For each domain:
  1) Load trained substitutability/complementarity scores (CSV) and build relation maps.
  2) Load test bundles from data/processed/<domain>/bundle_item_test.csv.
  3) Build a catalogue of all items from data/raw/<domain>/session_item.csv.
  4) ILR: corrupt m positions in the whole bundle (m=0..k), repeat R times, compute ILR.
     Save:
       - results/<domain>/ilr_agg.csv                 (per-bundle aggregation, with rho = m/k)
       - results/<domain>/ilr_agg_rho.csv             (aggregated by rho)
  5) Usefulness: split bundle into input/output (ratio), corrupt m positions in output (m=0..|O|),
     repeat R times, compute Usefulness.
     Save:
       - results/<domain>/usefulness_agg.csv          (per-bundle aggregation, rho_out = m/|O|)
       - results/<domain>/usefulness_agg_rho.csv      (aggregated by rho_out)

Global (across domains):
  - Concatenate per-domain *agg.csv and compute global curves vs rho and rho_out.
  - Save:
       - results/global_ilr_agg_rho.csv
       - results/global_usefulness_agg_rho.csv
  - Save figures (PNG + PDF) in results/.
"""

from __future__ import annotations

__author__ = "Marie Griffon"

from pathlib import Path
from typing import List, Dict, Any, Tuple
import math
import time
from contextlib import contextmanager
from datetime import timedelta


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from metrics.usefulness_ilr import build_relation_maps, ilr, usefulness


# =========================
# CONFIG
# =========================
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
RESULTS_PATH = Path("results")

DOMAINS = ["electronic", "clothing", "food"]

# Repetitions per (bundle, m)
R = 100
SEED = 42

# ILR: only evaluate bundles with size k in [K_MIN, K_MAX]
K_MIN = 2
K_MAX = 10

# Usefulness: input/output split ratio (input = floor(ratio*k), output = rest)
INPUT_RATIO = 0.6

# Optional: save long (per-repeat) raw samples (can be huge)
SAVE_LONG_TABLES = False


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
        
def _read_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=sep)


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


def _load_scores(domain: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subst_path = RESULTS_PATH / domain / "substitutability_scores.csv"
    compl_path = RESULTS_PATH / domain / "complementarity_scores.csv"

    subst_df = _read_csv(subst_path)
    compl_df = _read_csv(compl_path)

    # Minimal checks: item_x/item_y must exist. Score column is not enforced here.
    _ensure_cols(subst_df, ["item_x", "item_y"], f"subst_df ({subst_path})")
    _ensure_cols(compl_df, ["item_x", "item_y"], f"compl_df ({compl_path})")

    return subst_df, compl_df


def _load_catalogue_items(domain: str) -> np.ndarray:
    domain_path = RAW_PATH / domain
    df_session_item = _read_csv(domain_path / "session_item.csv", sep=",")
    _ensure_cols(df_session_item, ["item ID"], "df_session_item (raw)")
    return df_session_item["item ID"].unique()


def _load_test_bundles(domain: str) -> pd.DataFrame:
    domain_dir = PROCESSED_PATH / domain
    df_bundle_item_test = _read_csv(domain_dir / "bundle_item_test.csv", sep=",")
    _ensure_cols(df_bundle_item_test, ["bundle ID", "item ID"], "df_bundle_item_test")

    test_bundles = (
        df_bundle_item_test.groupby("bundle ID")["item ID"]
        .apply(list)
        .reset_index()
        .rename(columns={"item ID": "items"})
    )
    test_bundles["bundle_size"] = test_bundles["items"].apply(len)
    return test_bundles


def corrupt_bundle(items: List[Any], m: int, catalogue: np.ndarray, rng: np.random.Generator) -> List[Any]:
    """Replace m positions of a bundle by intruders drawn without replacement from catalogue \ items."""
    items_list = list(items)
    k = len(items_list)
    if m < 0 or m > k:
        raise ValueError("m must be in [0, k].")
    if m == 0:
        return items_list.copy()

    positions = rng.choice(k, size=m, replace=False)
    available = np.array([x for x in catalogue if x not in items_list], dtype=object)
    if len(available) < m:
        raise ValueError("Catalogue too small to draw m distinct intruders.")

    replacements = rng.choice(available, size=m, replace=False)
    corrupted = items_list.copy()
    for pos, rep in zip(positions, replacements):
        corrupted[pos] = rep

    assert len(corrupted) == k
    assert len(set(corrupted)) == len(corrupted)
    return corrupted


def generate_ilr_runs_for_k_range(
    df_bundles: pd.DataFrame,
    k_values: List[int],
    catalogue: np.ndarray,
    subst_map: Dict,
    compl_map: Dict,
    R: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for k in k_values:
        subset = df_bundles[df_bundles["bundle_size"] == k]
        if subset.empty:
            continue

        m_values = list(range(k + 1))
        for _, row in subset.iterrows():
            items = row["items"]
            bundle_id = row["bundle ID"]
            for m in m_values:
                for r in range(R):
                    corrupted = corrupt_bundle(items, m, catalogue, rng)
                    score = ilr([str(x) for x in corrupted], subst_map=subst_map, compl_map=compl_map)
                    records.append(
                        {
                            "bundle_id": bundle_id,
                            "bundle_size": k,
                            "m_corruption": m,
                            "repeat_id": r + 1,
                            "ilr_value": score,
                        }
                    )

    if not records:
        raise RuntimeError("No ILR samples generated (check test bundles / k range).")

    return pd.DataFrame(records)


def split_bundle(items: List[Any], ratio: float = 0.6) -> Tuple[List[Any], List[Any]]:
    """Return (I, O) split in-order with input_size=floor(ratio*k), both sides >= 1."""
    items = list(items)
    k = len(items)
    input_size = max(1, int(math.floor(ratio * k)))
    output_size = k - input_size
    if input_size < 1 or output_size < 1:
        raise ValueError(f"Invalid split sizes: input={input_size}, output={output_size}, k={k}")
    return items[:input_size], items[input_size:]


def corrupt_output(
    inputs: List[Any],
    outputs: List[Any],
    m: int,
    catalogue: np.ndarray,
    rng: np.random.Generator,
) -> List[Any]:
    """Corrupt m positions of O by injecting intruders outside (I ∪ O)."""
    inputs = list(inputs)
    outputs = list(outputs)
    o = len(outputs)

    if m < 0 or m > o:
        raise ValueError("m must be in [0, |O|].")
    if m == 0:
        return outputs.copy()

    positions = rng.choice(o, size=m, replace=False)
    forbidden = set(inputs) | set(outputs)
    candidates = np.array([x for x in catalogue if x not in forbidden], dtype=object)
    if len(candidates) < m:
        raise ValueError("Catalogue too small to draw m distinct intruders.")

    replacements = rng.choice(candidates, size=m, replace=False)
    corrupted = outputs.copy()
    for pos, rep in zip(positions, replacements):
        corrupted[pos] = rep

    assert len(corrupted) == o
    assert not (set(corrupted) & set(inputs))
    assert len(set(corrupted)) == len(corrupted)
    return corrupted


def run_domain(domain: str) -> None:
    with timed(f"domain={domain} TOTAL"):
        print(f"\n==================== DOMAIN: {domain} ====================")
        out_dir = RESULTS_PATH / domain
        out_dir.mkdir(parents=True, exist_ok=True)
    
        with timed(f"domain={domain} load scores + maps"):
            subst_df, compl_df = _load_scores(domain)
            subst_map, compl_map = build_relation_maps(subst_df, compl_df)
    
        with timed(f"domain={domain} load catalogue"):
            catalogue_items = _load_catalogue_items(domain)
    
        with timed(f"domain={domain} load test bundles"):
            test_bundles = _load_test_bundles(domain)
        available_k = sorted(test_bundles["bundle_size"].unique())
        k_values = [k for k in range(K_MIN, K_MAX + 1) if k in available_k]
    
        if not k_values:
            raise RuntimeError(
                f"No bundle sizes in [{K_MIN},{K_MAX}] for domain={domain}. "
                f"Available sizes={available_k[:25]}{'...' if len(available_k)>25 else ''}"
            )
    
        rng = np.random.default_rng(SEED)
    
        # =========================================================
        # ILR
        # =========================================================
        with timed(f"domain={domain} ILR"):
            print(f"[ILR] k_values={k_values}, R={R}")
            df_ilr = generate_ilr_runs_for_k_range(
                df_bundles=test_bundles,
                k_values=k_values,
                catalogue=catalogue_items,
                subst_map=subst_map,
                compl_map=compl_map,
                R=R,
                rng=rng,
            )
        
            if SAVE_LONG_TABLES:
                df_ilr.to_csv(out_dir / "ilr_long.csv", index=False)
        
            agg_ilr = (
                df_ilr.groupby(["bundle_id", "bundle_size", "m_corruption"])
                .agg(
                    mean_ilr=("ilr_value", "mean"),
                    std_ilr=("ilr_value", "std"),
                    repeats=("ilr_value", "count"),
                )
                .reset_index()
            )
            agg_ilr["rho"] = agg_ilr["m_corruption"] / agg_ilr["bundle_size"]
            agg_ilr["std_ilr"] = agg_ilr["std_ilr"].fillna(0.0)
            agg_ilr = agg_ilr.sort_values("rho").reset_index(drop=True)
            agg_ilr.to_csv(out_dir / "ilr_agg.csv", index=False)
        
            ilr_rho = (
                agg_ilr.groupby("rho")
                .agg(
                    n_bundles=("bundle_id", "nunique"),
                    mean_ilr=("mean_ilr", "mean"),
                    std_ilr=("mean_ilr", "std"),
                )
                .reset_index()
            )
            ilr_rho["std_ilr"] = ilr_rho["std_ilr"].fillna(0.0)
            ilr_rho["stderr"] = ilr_rho["std_ilr"] / np.sqrt(ilr_rho["n_bundles"].replace(0, np.nan).to_numpy())
            ilr_rho["ci_low"] = ilr_rho["mean_ilr"] - 1.96 * ilr_rho["stderr"]
            ilr_rho["ci_high"] = ilr_rho["mean_ilr"] + 1.96 * ilr_rho["stderr"]
            ilr_rho = ilr_rho.sort_values("rho").reset_index(drop=True)
            # ilr_rho.to_csv(out_dir / "ilr_agg_rho.csv", index=False)
        
            # Plot ILR (per-domain)
            plt.rcParams["ps.fonttype"] = 42
            plt.rcParams["pdf.fonttype"] = 42
        
            plt.figure(figsize=(8, 5))
            x = ilr_rho["rho"].to_numpy()
            y = ilr_rho["mean_ilr"].to_numpy()
            ci_low = ilr_rho["ci_low"].to_numpy()
            ci_high = ilr_rho["ci_high"].to_numpy()
            plt.plot(x, y, marker="o", label=f"{_domain_title(domain)} mean ILR")
            plt.fill_between(x, ci_low, ci_high, alpha=0.2, label="95% CI")
            plt.xlabel(r"Relative corruption  $\rho = m/k$")
            plt.ylabel(r"Average ILR $\mu_{\rho}$")
            plt.xlim(-0.02, 1.02)
            plt.ylim(bottom=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "ilr_vs_corruption.png")
            plt.savefig(out_dir / "ilr_vs_corruption.pdf")
            plt.close()
        
            # ILR summary stats (per-domain)
            if len(ilr_rho) >= 2:
                sp_ilr = spearmanr(ilr_rho["rho"], ilr_rho["mean_ilr"]).correlation
            else:
                sp_ilr = np.nan
            mu_rho0 = ilr_rho.loc[np.isclose(ilr_rho["rho"], 0.0), "mean_ilr"].mean()
            mu_rho1 = ilr_rho.loc[np.isclose(ilr_rho["rho"], 1.0), "mean_ilr"].mean()
            drop_ilr = mu_rho0 - mu_rho1 if not (np.isnan(mu_rho0) or np.isnan(mu_rho1)) else np.nan
        
            print("[ILR] Degradation indicators")
            print(f"  - Spearman(rho, mean_ilr): {sp_ilr:.4f}")
            print(f"  - Drop total (rho=0 -> rho=1): {drop_ilr:.4f}")
        
        # =========================================================
        # Usefulness
        # =========================================================
        with timed(f"domain={domain} Usefulness"):
            print(f"[Usefulness] INPUT_RATIO={INPUT_RATIO}, R={R}")
        
            records: List[Dict[str, Any]] = []
        
            for _, row in test_bundles.iterrows():
                items = row["items"]
                k = len(items)
                inputs, outputs = split_bundle(items, ratio=INPUT_RATIO)
                o = len(outputs)   
                for m in range(o + 1):
                    for r in range(1, R + 1):
                        corrupted_o = corrupt_output(inputs, outputs, m, catalogue_items, rng)
                        u_val = usefulness(
                            [str(x) for x in inputs],
                            [str(x) for x in corrupted_o],
                            subst_map=subst_map,
                            compl_map=compl_map,
                        )
                        records.append(
                            {
                                "bundle_id": row["bundle ID"],
                                "bundle_size": k,
                                "input_size": len(inputs),
                                "output_size": o,
                                "m_corruption": m,
                                "repeat_id": r,
                                "usefulness_value": u_val,
                            }
                        )
        
            if not records:
                raise RuntimeError("No Usefulness samples generated.")
        
            df_use = pd.DataFrame(records)
            if SAVE_LONG_TABLES:
                df_use.to_csv(out_dir / "usefulness_long.csv", index=False)
        
            agg_use = (
                df_use.groupby(["bundle_id", "m_corruption", "output_size"])
                .agg(
                    mean_usefulness=("usefulness_value", "mean"),
                    std_usefulness=("usefulness_value", "std"),
                    repeats=("usefulness_value", "count"),
                )
                .reset_index()
            )
            agg_use["rho_out"] = agg_use["m_corruption"] / agg_use["output_size"]
            agg_use["std_usefulness"] = agg_use["std_usefulness"].fillna(0.0)
            agg_use = agg_use.sort_values("rho_out").reset_index(drop=True)
            agg_use.to_csv(out_dir / "usefulness_agg.csv", index=False)
        
            use_rho = (
                agg_use.groupby("rho_out")
                .agg(
                    n_bundles=("bundle_id", "nunique"),
                    mean_usefulness=("mean_usefulness", "mean"),
                    std_usefulness=("mean_usefulness", "std"),
                )
                .reset_index()
            )
            use_rho["std_usefulness"] = use_rho["std_usefulness"].fillna(0.0)
            use_rho["stderr"] = use_rho["std_usefulness"] / np.sqrt(use_rho["n_bundles"].replace(0, np.nan).to_numpy())
            use_rho["ci_low"] = use_rho["mean_usefulness"] - 1.96 * use_rho["stderr"]
            use_rho["ci_high"] = use_rho["mean_usefulness"] + 1.96 * use_rho["stderr"]
            use_rho = use_rho.sort_values("rho_out").reset_index(drop=True)
            # use_rho.to_csv(out_dir / "usefulness_agg_rho.csv", index=False)
        
            # Plot Usefulness (per-domain)
            plt.figure(figsize=(8, 5))
            x = use_rho["rho_out"].to_numpy()
            y = use_rho["mean_usefulness"].to_numpy()
            ci_low = use_rho["ci_low"].to_numpy()
            ci_high = use_rho["ci_high"].to_numpy()
            plt.plot(x, y, marker="o", label=f"{_domain_title(domain)} mean Usefulness")
            plt.fill_between(x, ci_low, ci_high, alpha=0.2, label="95% CI")
            plt.xlabel(r"Relative corruption  $\rho_o = m/|O|$")
            plt.ylabel(r"Average Usefulness $\mu_{\rho_o}$")
            plt.xlim(-0.02, 1.02)
            plt.ylim(bottom=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "usefulness_vs_corruption.png")
            plt.savefig(out_dir / "usefulness_vs_corruption.pdf")
            plt.close()
        
            # Usefulness summary stats (per-domain)
            if len(use_rho) >= 2:
                sp_use = spearmanr(use_rho["rho_out"], use_rho["mean_usefulness"]).correlation
            else:
                sp_use = np.nan
            mu_rho0 = use_rho.loc[np.isclose(use_rho["rho_out"], 0.0), "mean_usefulness"].mean()
            mu_rho1 = use_rho.loc[np.isclose(use_rho["rho_out"], 1.0), "mean_usefulness"].mean()
            drop_use = mu_rho0 - mu_rho1 if not (np.isnan(mu_rho0) or np.isnan(mu_rho1)) else np.nan
        
            print("[Usefulness] Degradation indicators")
            print(f"  - Spearman(rho_out, mean_usefulness): {sp_use:.4f}")
            print(f"  - Drop total (rho_out=0 -> rho_out=1): {drop_use:.4f}")


def aggregate_global() -> None:
    """
    Load per-domain aggregates and produce global rho curves.
    """
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # ---- Usefulness
    use_frames = []
    ilr_frames = []
    missing = []

    print(f"\n==================== GLOBAL ====================")

    for domain in DOMAINS:
        base_dir = RESULTS_PATH / domain

        use_path = base_dir / "usefulness_agg.csv"
        ilr_path = base_dir / "ilr_agg.csv"

        if use_path.is_file():
            df = pd.read_csv(use_path)
            df["subset"] = domain
            df["bundle_id_subset"] = df["subset"].astype(str) + "_" + df["bundle_id"].astype(str)
            use_frames.append(df)
        else:
            missing.append(str(use_path))

        if ilr_path.is_file():
            df = pd.read_csv(ilr_path)
            df["subset"] = domain
            df["bundle_id_subset"] = df["subset"].astype(str) + "_" + df["bundle_id"].astype(str)
            ilr_frames.append(df)
        else:
            missing.append(str(ilr_path))

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(" -", f)

    # ---- Global Usefulness curve
    if use_frames:
        df_use = pd.concat(use_frames, ignore_index=True)
        g = (
            df_use.groupby("rho_out")
            .agg(
                n_bundles=("bundle_id_subset", "nunique"),
                mean_usefulness=("mean_usefulness", "mean"),
                std_usefulness=("mean_usefulness", "std"),
            )
            .reset_index()
        )
        g["std_usefulness"] = g["std_usefulness"].fillna(0.0)
        g["stderr"] = g["std_usefulness"] / np.sqrt(g["n_bundles"].replace(0, np.nan).to_numpy())
        g["ci_low"] = g["mean_usefulness"] - 1.96 * g["stderr"]
        g["ci_high"] = g["mean_usefulness"] + 1.96 * g["stderr"]
        g = g.sort_values("rho_out").reset_index(drop=True)
        # g.to_csv(RESULTS_PATH / "global_usefulness_agg_rho.csv", index=False)

        plt.figure(figsize=(8, 5))
        x = g["rho_out"].to_numpy()
        y = g["mean_usefulness"].to_numpy()
        plt.plot(x, y, marker="o", label="Mean Usefulness")
        plt.fill_between(x, g["ci_low"].to_numpy(), g["ci_high"].to_numpy(), alpha=0.2, label="95% CI")
        plt.xlabel(r"Relative corruption  $\rho_o = m/|O|$")
        plt.ylabel(r"Average Usefulness $\mu_{\rho_o}$")
        plt.xlim(-0.02, 1.02)
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_PATH / "global_usefulness_vs_corruption.png")
        plt.savefig(RESULTS_PATH / "global_usefulness_vs_corruption.pdf")
        plt.close()

        if len(g) >= 2:
            sp = spearmanr(g["rho_out"], g["mean_usefulness"]).correlation
        else:
            sp = np.nan
        mu0 = g.loc[np.isclose(g["rho_out"], 0.0), "mean_usefulness"].mean()
        mu1 = g.loc[np.isclose(g["rho_out"], 1.0), "mean_usefulness"].mean()
        drop = mu0 - mu1 if not (np.isnan(mu0) or np.isnan(mu1)) else np.nan

        print("\n[Usefulness]")
        print(f"  - Spearman(rho_out, mean_usefulness): {sp:.4f}")
        print(f"  - Drop total (rho_out=0 -> rho_out=1): {drop:.4f}")
    else:
        print("\nNo usefulness_agg.csv loaded.")

    # ---- Global ILR curve
    if ilr_frames:
        df_ilr = pd.concat(ilr_frames, ignore_index=True)
        g = (
            df_ilr.groupby("rho")
            .agg(
                n_bundles=("bundle_id_subset", "nunique"),
                mean_ilr=("mean_ilr", "mean"),
                std_ilr=("mean_ilr", "std"),
            )
            .reset_index()
        )
        g["std_ilr"] = g["std_ilr"].fillna(0.0)
        g["stderr"] = g["std_ilr"] / np.sqrt(g["n_bundles"].replace(0, np.nan).to_numpy())
        g["ci_low"] = g["mean_ilr"] - 1.96 * g["stderr"]
        g["ci_high"] = g["mean_ilr"] + 1.96 * g["stderr"]
        g = g.sort_values("rho").reset_index(drop=True)
        # g.to_csv(RESULTS_PATH / "global_ilr_agg_rho.csv", index=False)

        plt.figure(figsize=(8, 5))
        x = g["rho"].to_numpy()
        y = g["mean_ilr"].to_numpy()
        plt.plot(x, y, marker="o", label="Mean ILR")
        plt.fill_between(x, g["ci_low"].to_numpy(), g["ci_high"].to_numpy(), alpha=0.2, label="95% CI")
        plt.xlabel(r"Relative corruption  $\rho = m/k$")
        plt.ylabel(r"Average ILR $\mu_{\rho}$")
        plt.xlim(-0.02, 1.02)
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_PATH / "global_ilr_vs_corruption.png")
        plt.savefig(RESULTS_PATH / "global_ilr_vs_corruption.pdf")
        plt.close()

        if len(g) >= 2:
            sp = spearmanr(g["rho"], g["mean_ilr"]).correlation
        else:
            sp = np.nan
        mu0 = g.loc[np.isclose(g["rho"], 0.0), "mean_ilr"].mean()
        mu1 = g.loc[np.isclose(g["rho"], 1.0), "mean_ilr"].mean()
        drop = mu0 - mu1 if not (np.isnan(mu0) or np.isnan(mu1)) else np.nan

        print("\n[ILR]")
        print(f"  - Spearman(rho, mean_ilr): {sp:.4f}")
        print(f"  - Drop total (rho=0 -> rho=1): {drop:.4f}")
    else:
        print("\nNo ilr_agg.csv loaded.")


def main() -> None:
    with timed("FULL RUN"):
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)

        for domain in DOMAINS:
            run_domain(domain)

        with timed("GLOBAL AGGREGATION"):
            aggregate_global()

        print("\nDone.")

if __name__ == "__main__":
    main()