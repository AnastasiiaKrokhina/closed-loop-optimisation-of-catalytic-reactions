#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_REPO_ID = "SandboxAQ/aqcat25"

COMMON_ELEMENTS = [
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "Li", "Na", "K", "Mg", "Ca", "Al", "Si", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "Se", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Te", "W", "Ir", "Pt", "Au", "Pb",
]

METAL_ELEMENTS = set([
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb", "Bi",
])

SOLVENT_LIBRARY = pd.DataFrame([
    {
        "solvent": "water",
        "polarity": 1.00,
        "viscosity_cP": 0.89,
        "dielectric": 78.4,
        "hbond_strength": 1.00,
    },
    {
        "solvent": "methanol",
        "polarity": 0.76,
        "viscosity_cP": 0.54,
        "dielectric": 32.6,
        "hbond_strength": 0.80,
    },
    {
        "solvent": "ethanol",
        "polarity": 0.65,
        "viscosity_cP": 1.07,
        "dielectric": 24.6,
        "hbond_strength": 0.70,
    },
    {
        "solvent": "acetonitrile",
        "polarity": 0.46,
        "viscosity_cP": 0.37,
        "dielectric": 37.5,
        "hbond_strength": 0.20,
    },
    {
        "solvent": "dmf",
        "polarity": 0.39,
        "viscosity_cP": 0.92,
        "dielectric": 36.7,
        "hbond_strength": 0.30,
    },
    {
        "solvent": "toluene",
        "polarity": 0.10,
        "viscosity_cP": 0.59,
        "dielectric": 2.4,
        "hbond_strength": 0.00,
    },
])


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Data loading and sampling
# -----------------------------

def load_aqcat_streaming_sample(
    repo_id: str,
    split: str,
    max_rows: int,
    seed: int,
    energy_min: Optional[float] = None,
    energy_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Stream AQCat25 metadata and keep a bounded random-ish sample.

    This avoids loading the full dataset into memory.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install datasets first: pip install datasets"
        ) from exc

    print(f"Streaming split={split!r} from {repo_id!r} ...")

    ds = load_dataset(repo_id, split=split, streaming=True)

    rng = random.Random(seed)
    reservoir: List[dict] = []
    seen = 0

    for row in tqdm(ds, desc="Streaming AQCat25 metadata"):
        energy = safe_float(row.get("adsorption_energy"))
        if np.isnan(energy):
            continue
        if energy_min is not None and energy < energy_min:
            continue
        if energy_max is not None and energy > energy_max:
            continue

        seen += 1
        if len(reservoir) < max_rows:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_rows:
                reservoir[j] = row

    if not reservoir:
        raise RuntimeError("No rows were sampled. Try relaxing filters or using another split.")

    df = pd.DataFrame(reservoir)
    print(f"Sampled {len(df):,} rows from {seen:,} eligible streamed rows.")
    return df


def stratified_sample_existing_parquet(
    parquet_path: str | Path,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """
    Optional mode: if you already have a local metadata parquet, sample it stratified.
    """
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["adsorption_energy"])

    df["energy_bin"] = pd.cut(
        df["adsorption_energy"],
        bins=[-20, -4, -3, -2, -1, 0, 1, 20],
        labels=["ultra_strong", "very_strong", "strong", "medium", "weak", "positive", "high_positive"],
    )

    group_cols = ["adsorbate", "is_spin_off", "energy_bin"]
    sampled_parts = []
    target_frac = min(1.0, max_rows / max(len(df), 1))

    for _, group in df.groupby(group_cols, dropna=False):
        n = max(1, int(round(len(group) * target_frac)))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=seed))

    sampled = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed)

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)

    sampled = sampled.drop(columns=["energy_bin"], errors="ignore")
    print(f"Loaded {len(df):,} rows and kept {len(sampled):,} stratified rows.")
    return sampled


# -----------------------------
# Feature engineering
# -----------------------------

def parse_elements_from_string(text: str) -> Dict[str, int]:
    """
    Very simple formula-like parser.

    Works reasonably for strings such as:
        *CO
        *NH2N(CH3)2
        mp-1216478_001_2_False

    It is intentionally simple and portfolio-friendly.
    For real chemistry, replace with RDKit/pymatgen-based parsing.
    """
    if not isinstance(text, str):
        return {}

    cleaned = text.replace("*", "")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", cleaned)
    counts: Dict[str, int] = {}
    for elem, num in tokens:
        if elem in COMMON_ELEMENTS:
            counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def adsorbate_features(adsorbate: str) -> Dict[str, float]:
    counts = parse_elements_from_string(adsorbate)
    total_atoms = sum(counts.values())

    hetero = sum(v for k, v in counts.items() if k not in {"C", "H"})
    heavy = sum(v for k, v in counts.items() if k != "H")

    features = {
        "ads_total_atoms_proxy": total_atoms,
        "ads_heavy_atoms_proxy": heavy,
        "ads_hetero_atoms_proxy": hetero,
        "ads_C_count": counts.get("C", 0),
        "ads_H_count": counts.get("H", 0),
        "ads_N_count": counts.get("N", 0),
        "ads_O_count": counts.get("O", 0),
        "ads_S_count": counts.get("S", 0),
        "ads_contains_N": float(counts.get("N", 0) > 0),
        "ads_contains_O": float(counts.get("O", 0) > 0),
        "ads_contains_S": float(counts.get("S", 0) > 0),
        "ads_hetero_fraction_proxy": hetero / total_atoms if total_atoms else 0.0,
    }
    return features


def slab_features(slab_id: str) -> Dict[str, float]:
    counts = parse_elements_from_string(slab_id)
    elems = set(counts)
    metals = elems.intersection(METAL_ELEMENTS)

    # Try to extract Miller/facet-like numbers if present.
    # Example: mp-1216478_001_2_False -> contains "001".
    facet_match = re.search(r"_(\d{3})_", str(slab_id))
    facet = facet_match.group(1) if facet_match else "unknown"

    features = {
        "slab_num_elements_proxy": len(elems),
        "slab_num_metal_elements_proxy": len(metals),
        "slab_is_metal_rich_proxy": float(len(metals) >= max(1, len(elems) / 2)) if elems else 0.0,
        "slab_contains_Fe": float("Fe" in elems),
        "slab_contains_Co": float("Co" in elems),
        "slab_contains_Ni": float("Ni" in elems),
        "slab_contains_Cu": float("Cu" in elems),
        "slab_contains_Pt": float("Pt" in elems),
        "slab_contains_Pd": float("Pd" in elems),
        "slab_contains_Ru": float("Ru" in elems),
        "slab_contains_Rh": float("Rh" in elems),
        "slab_facet_code": facet,
    }
    return features


def add_environment_descriptors(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Add MD-inspired reaction-condition descriptors.

    These simulate coarse-grained descriptors one could obtain from short MD:
    - diffusion proxy
    - viscosity proxy
    - aggregation proxy
    - catalyst-environment contact proxy
    - ligand/solvent effects

    In a real project, replace this block with actual MD analysis outputs.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    solvents = SOLVENT_LIBRARY.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    temperature_K = rng.uniform(293.15, 393.15, size=n)
    residence_time_s = np.exp(rng.uniform(np.log(5), np.log(600), size=n))
    ligand_concentration_mM = np.exp(rng.uniform(np.log(0.1), np.log(100), size=n))
    catalyst_loading_mol_percent = rng.uniform(0.1, 10.0, size=n)
    flow_rate_uL_min = np.exp(rng.uniform(np.log(1), np.log(500), size=n))

    viscosity_proxy = solvents["viscosity_cP"].values * (293.15 / temperature_K) ** 0.7
    diffusion_proxy = temperature_K / np.maximum(viscosity_proxy, 1e-6)
    diffusion_proxy = diffusion_proxy / np.nanmedian(diffusion_proxy)

    # Aggregation becomes more likely with low polarity, higher concentration, and high catalyst loading.
    aggregation_proxy = (
        (1.0 - solvents["polarity"].values)
        * np.log1p(ligand_concentration_mM)
        * np.sqrt(catalyst_loading_mol_percent)
    )
    aggregation_proxy = aggregation_proxy / np.nanpercentile(aggregation_proxy, 95)
    aggregation_proxy = np.clip(aggregation_proxy, 0, 2)

    # Contact proxy: rough simulated measure of how much the environment contacts the catalyst surface.
    contact_proxy = (
        0.4 * solvents["polarity"].values
        + 0.3 * solvents["hbond_strength"].values
        + 0.2 * np.log1p(residence_time_s) / np.log1p(600)
        + 0.1 * np.log1p(catalyst_loading_mol_percent) / np.log1p(10)
    )

    out = df.copy()
    out["solvent"] = solvents["solvent"].values
    out["solvent_polarity"] = solvents["polarity"].values
    out["solvent_dielectric"] = solvents["dielectric"].values
    out["solvent_hbond_strength"] = solvents["hbond_strength"].values
    out["temperature_K"] = temperature_K
    out["residence_time_s"] = residence_time_s
    out["ligand_concentration_mM"] = ligand_concentration_mM
    out["catalyst_loading_mol_percent"] = catalyst_loading_mol_percent
    out["flow_rate_uL_min"] = flow_rate_uL_min
    out["viscosity_proxy"] = viscosity_proxy
    out["diffusion_proxy"] = diffusion_proxy
    out["aggregation_proxy"] = aggregation_proxy
    out["catalyst_environment_contact_proxy"] = contact_proxy

    return out


def build_feature_table(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = add_environment_descriptors(df, seed=seed)

    feature_rows: List[Dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building descriptors"):
        features: Dict[str, object] = {}

        features.update(adsorbate_features(str(row.get("adsorbate", ""))))
        features.update(slab_features(str(row.get("slab_id", ""))))

        features["total_energy"] = safe_float(row.get("total_energy"))
        features["fmax"] = safe_float(row.get("fmax"))
        features["mag"] = safe_float(row.get("mag"), default=0.0)
        features["is_spin_off"] = bool(row.get("is_spin_off", False))
        features["is_rerun"] = bool(row.get("is_rerun", False))
        features["is_md"] = bool(row.get("is_md", False))
        features["fid"] = safe_float(row.get("fid"), default=0.0)

        # Reaction/environment descriptors
        for col in [
            "solvent", "solvent_polarity", "solvent_dielectric", "solvent_hbond_strength",
            "temperature_K", "residence_time_s", "ligand_concentration_mM",
            "catalyst_loading_mol_percent", "flow_rate_uL_min", "viscosity_proxy",
            "diffusion_proxy", "aggregation_proxy", "catalyst_environment_contact_proxy",
        ]:
            features[col] = row[col]

        # Keep identifiers for interpretation, not model target.
        features["frame_id"] = row.get("frame_id", None)
        features["adsorbate"] = row.get("adsorbate", None)
        features["slab_id"] = row.get("slab_id", None)

        feature_rows.append(features)

    X = pd.DataFrame(feature_rows)
    y = df["adsorption_energy"].astype(float)


    # removing bool-type columns from the dataset
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X, y


# -----------------------------
# Modelling
# -----------------------------

def make_model(X: pd.DataFrame, seed: int) -> Pipeline:
    identifier_cols = ["frame_id", "adsorbate", "slab_id"]
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]

    categorical_cols = [
        c for c in X.columns
        if c not in identifier_cols + bool_cols
        and not pd.api.types.is_numeric_dtype(X[c])
    ]

    numeric_cols = [
        c for c in X.columns
        if c not in categorical_cols + bool_cols + identifier_cols
        and pd.api.types.is_numeric_dtype(X[c])
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            # ("bool", Pipeline([
            #     ("imputer", SimpleImputer(strategy="most_frequent")),
            # ]), bool_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    # RandomForest gives a simple uncertainty proxy using variance across trees.
    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf),
    ])

    return model


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    seed: int,
) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = make_model(X_train, seed=seed)
    print("Training surrogate model...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "mae_eV": float(mean_absolute_error(y_test, pred)),
        "rmse_eV": float(math.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model metrics:")
    print(json.dumps(metrics, indent=2))

    parity = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": pred,
        "frame_id": X_test["frame_id"].values,
        "adsorbate": X_test["adsorbate"].values,
        "slab_id": X_test["slab_id"].values,
    })

    parity.to_csv(output_dir / "test_predictions.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(parity["y_true"], parity["y_pred"], s=10, alpha=0.5)
    low = min(parity["y_true"].min(), parity["y_pred"].min())
    high = max(parity["y_true"].max(), parity["y_pred"].max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("DFT adsorption energy / eV")
    plt.ylabel("Predicted adsorption energy / eV")
    plt.title("Surrogate model parity plot")
    plt.tight_layout()
    plt.savefig(output_dir / "parity_plot.png", dpi=200)
    plt.close()

    joblib.dump(model, output_dir / "model.joblib")
    return model, metrics, parity


# -----------------------------
# Uncertainty and Bayesian optimisation
# -----------------------------

def random_forest_predict_uncertainty(model: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict mean and uncertainty from tree ensemble spread.
    """
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["regressor"]

    X_transformed = preprocessor.transform(X)
    tree_preds = np.vstack([tree.predict(X_transformed) for tree in rf.estimators_])

    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean, std


def expected_improvement_minimization(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected improvement for minimisation.

    Lower adsorption energy is treated as better here.
    If your chemical objective is different, modify this function.
    """
    improvement = best_y - mu - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 0] = 0.0
    return ei


def make_candidate_conditions(
    base_X: pd.DataFrame,
    n_candidates: int,
    seed: int,
) -> pd.DataFrame:
    """
    Create virtual candidates by taking existing catalyst/adsorbate entries
    and varying reaction/environment descriptors.
    """
    rng = np.random.default_rng(seed)

    base = base_X.sample(
        n=n_candidates,
        replace=True,
        random_state=seed,
    ).reset_index(drop=True)

    solvents = SOLVENT_LIBRARY.sample(
        n=n_candidates,
        replace=True,
        random_state=seed + 13,
    ).reset_index(drop=True)

    temperature_K = rng.uniform(293.15, 423.15, size=n_candidates)
    residence_time_s = np.exp(rng.uniform(np.log(2), np.log(1200), size=n_candidates))
    ligand_concentration_mM = np.exp(rng.uniform(np.log(0.05), np.log(200), size=n_candidates))
    catalyst_loading_mol_percent = rng.uniform(0.05, 15.0, size=n_candidates)
    flow_rate_uL_min = np.exp(rng.uniform(np.log(0.5), np.log(1000), size=n_candidates))

    viscosity_proxy = solvents["viscosity_cP"].values * (293.15 / temperature_K) ** 0.7
    diffusion_proxy = temperature_K / np.maximum(viscosity_proxy, 1e-6)
    diffusion_proxy = diffusion_proxy / np.nanmedian(diffusion_proxy)

    aggregation_proxy = (
        (1.0 - solvents["polarity"].values)
        * np.log1p(ligand_concentration_mM)
        * np.sqrt(catalyst_loading_mol_percent)
    )
    aggregation_proxy = aggregation_proxy / np.nanpercentile(aggregation_proxy, 95)
    aggregation_proxy = np.clip(aggregation_proxy, 0, 2)

    contact_proxy = (
        0.4 * solvents["polarity"].values
        + 0.3 * solvents["hbond_strength"].values
        + 0.2 * np.log1p(residence_time_s) / np.log1p(1200)
        + 0.1 * np.log1p(catalyst_loading_mol_percent) / np.log1p(15)
    )

    candidates = base.copy()
    candidates["solvent"] = solvents["solvent"].values
    candidates["solvent_polarity"] = solvents["polarity"].values
    candidates["solvent_dielectric"] = solvents["dielectric"].values
    candidates["solvent_hbond_strength"] = solvents["hbond_strength"].values
    candidates["temperature_K"] = temperature_K
    candidates["residence_time_s"] = residence_time_s
    candidates["ligand_concentration_mM"] = ligand_concentration_mM
    candidates["catalyst_loading_mol_percent"] = catalyst_loading_mol_percent
    candidates["flow_rate_uL_min"] = flow_rate_uL_min
    candidates["viscosity_proxy"] = viscosity_proxy
    candidates["diffusion_proxy"] = diffusion_proxy
    candidates["aggregation_proxy"] = aggregation_proxy
    candidates["catalyst_environment_contact_proxy"] = contact_proxy

    return candidates


def reaction_objective_score(
    predicted_adsorption_energy: np.ndarray,
    diffusion_proxy: np.ndarray,
    aggregation_proxy: np.ndarray,
    contact_proxy: np.ndarray,
    target_energy: float = -2.0,
) -> np.ndarray:
    """
    Composite simulated catalytic objective.

    This is not claiming that lower adsorption energy is always better.
    Many catalytic reactions require intermediate binding: not too weak and not too strong.

    Here we optimize toward target_energy, while penalizing poor diffusion and aggregation.
    Lower score is better.
    """
    binding_term = np.abs(predicted_adsorption_energy - target_energy)
    diffusion_penalty = 0.3 / np.maximum(diffusion_proxy, 1e-3)
    aggregation_penalty = 0.5 * aggregation_proxy
    contact_bonus = -0.2 * contact_proxy

    return binding_term + diffusion_penalty + aggregation_penalty + contact_bonus


def closed_loop_optimisation(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    n_candidates: int,
    top_k: int,
    seed: int,
    target_energy: float,
) -> pd.DataFrame:
    print("Generating virtual reaction-condition candidates...")
    candidates = make_candidate_conditions(X, n_candidates=n_candidates, seed=seed)

    mu, sigma = random_forest_predict_uncertainty(model, candidates)

    score = reaction_objective_score(
        predicted_adsorption_energy=mu,
        diffusion_proxy=candidates["diffusion_proxy"].values.astype(float),
        aggregation_proxy=candidates["aggregation_proxy"].values.astype(float),
        contact_proxy=candidates["catalyst_environment_contact_proxy"].values.astype(float),
        target_energy=target_energy,
    )

    # EI on the composite score is approximated by treating score as mean-like.
    # Since uncertainty is available for adsorption-energy prediction only,
    # this is a pragmatic laptop-scale proxy.
    best_score = np.min(
        reaction_objective_score(
            predicted_adsorption_energy=y.values,
            diffusion_proxy=X["diffusion_proxy"].values.astype(float),
            aggregation_proxy=X["aggregation_proxy"].values.astype(float),
            contact_proxy=X["catalyst_environment_contact_proxy"].values.astype(float),
            target_energy=target_energy,
        )
    )

    ei = expected_improvement_minimization(
        mu=score,
        sigma=sigma,
        best_y=best_score,
        xi=0.01,
    )

    suggestions = candidates.copy()
    suggestions["predicted_adsorption_energy_eV"] = mu
    suggestions["uncertainty_eV"] = sigma
    suggestions["objective_score_lower_is_better"] = score
    suggestions["expected_improvement"] = ei

    suggestions = suggestions.sort_values(
        by=["expected_improvement", "objective_score_lower_is_better"],
        ascending=[False, True],
    ).head(top_k)

    keep_cols = [
        "frame_id", "adsorbate", "slab_id", "solvent", "temperature_K",
        "residence_time_s", "ligand_concentration_mM", "catalyst_loading_mol_percent",
        "flow_rate_uL_min", "diffusion_proxy", "viscosity_proxy", "aggregation_proxy",
        "catalyst_environment_contact_proxy", "predicted_adsorption_energy_eV",
        "uncertainty_eV", "objective_score_lower_is_better", "expected_improvement",
    ]

    suggestions[keep_cols].to_csv(output_dir / "optimisation_suggestions.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(score, ei, s=10, alpha=0.5)
    plt.xlabel("Composite objective score, lower is better")
    plt.ylabel("Expected improvement")
    plt.title("Bayesian optimisation candidate ranking")
    plt.tight_layout()
    plt.savefig(output_dir / "acquisition_plot.png", dpi=200)
    plt.close()

    return suggestions[keep_cols]


# -----------------------------
# Main CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laptop-scale AQCat25 MD-informed closed-loop optimisation demo"
    )

    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--split", type=str, default="val_id")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="aqcat25_bo_output")

    parser.add_argument(
        "--local-parquet",
        type=str,
        default=None,
        help="Optional local metadata parquet. If provided, sampling is done locally instead of streaming from HF.",
    )

    parser.add_argument("--energy-min", type=float, default=None)
    parser.add_argument("--energy-max", type=float, default=None)
    parser.add_argument("--n-candidates", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--target-energy",
        type=float,
        default=-2.0,
        help="Target adsorption energy for Sabatier-like objective. Default: -2.0 eV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_output_dir(args.output_dir)

    if args.local_parquet:
        sampled_df = stratified_sample_existing_parquet(
            parquet_path=args.local_parquet,
            max_rows=args.max_rows,
            seed=args.seed,
        )
    else:
        sampled_df = load_aqcat_streaming_sample(
            repo_id=args.repo_id,
            split=args.split,
            max_rows=args.max_rows,
            seed=args.seed,
            energy_min=args.energy_min,
            energy_max=args.energy_max,
        )

    sampled_path = output_dir / "sampled_metadata.parquet"
    sampled_df.to_parquet(sampled_path, index=False)
    print(f"Saved sampled metadata to {sampled_path}")

    X, y = build_feature_table(sampled_df, seed=args.seed)
    features = X.copy()
    features["adsorption_energy"] = y.values
    features_path = output_dir / "features.parquet"
    features.to_parquet(features_path, index=False)
    print(f"Saved features to {features_path}")

    model, metrics, parity = train_and_evaluate(
        X=X,
        y=y,
        output_dir=output_dir,
        seed=args.seed,
    )

    suggestions = closed_loop_optimisation(
        model=model,
        X=X,
        y=y,
        output_dir=output_dir,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        seed=args.seed,
        target_energy=args.target_energy,
    )

    print("\nTop optimisation suggestions:")
    print(suggestions.head(args.top_k).to_string(index=False))

    print("\nDone. Results saved in:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
