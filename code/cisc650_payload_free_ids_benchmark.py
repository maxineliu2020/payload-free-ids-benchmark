#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CISC 650 Computer Networks - Nova Southeastern University (NSU)

Project:
    Explainable Machine Learning for Payload-Free Intrusion Detection:
    A Reproducible Benchmark for Cybersecurity Education

Author:
    Yuanyuan Liu (Maxine)

Course:
    CISC 650 Computer Networks
    College of Computing, Artificial Intelligence, and Cybersecurity
    Nova Southeastern University

Instructor / Co-author:
    Wei Li, Ph.D.

Purpose:
    This script supports the conference-paper experiment package by providing a
    reproducible, well-documented benchmark for payload-free intrusion detection.
    It can run a controlled synthetic IDS stress test and, when CICIDS2017 CSV
    files are supplied locally, a flow-level benchmark using public IDS data.

Important scope note:
    The script does not inspect packet payloads and does not claim to process
    authentic encrypted healthcare packet captures. It works with flow-level
    metadata features that are appropriate for privacy-aware cybersecurity
    education and reproducible experimentation.

Typical usage:
    1) Synthetic stress test only:
        python cisc650_payload_free_ids_benchmark.py --synthetic-only

    2) CICIDS2017 benchmark with CSV files stored in ./data:
        python cisc650_payload_free_ids_benchmark.py --data-dir ./data

Outputs:
    - CSV metrics tables
    - ROC / precision-recall plots
    - feature-group ablation results
    - robustness results under feature perturbation
    - optional SHAP summary plot when the SHAP package is available
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Centralized configuration for the NSU CISC 650 IDS experiments.

    The defaults reproduce the controlled stress-test setting used in the paper.
    Values can be changed from the command line when instructors want students
    to explore a more difficult or easier classification setting.
    """

    random_state: int = 42
    synthetic_samples: int = 40_000
    synthetic_features: int = 30
    synthetic_informative: int = 10
    synthetic_redundant: int = 10
    synthetic_class_sep: float = 0.45
    synthetic_label_noise: float = 0.05
    benign_attack_ratio: Tuple[float, float] = (0.8, 0.2)
    retained_synthetic_features: int = 25
    cicids_sample_size: int = 10_300
    test_size: float = 0.15
    validation_size: float = 0.15


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure readable command-line logging for experiment progress."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def ensure_output_dir(path: Path) -> Path:
    """Create the output directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_metric_auc(metric_name: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC or PR-AUC safely.

    Some small classroom experiments may accidentally create a test split with
    only one class. In that case, ROC-AUC is undefined. Returning NaN is better
    than crashing, because students can inspect the reason in the output table.
    """

    try:
        if metric_name == "roc":
            return float(roc_auc_score(y_true, y_score))
        if metric_name == "pr":
            return float(average_precision_score(y_true, y_score))
        raise ValueError(f"Unsupported AUC metric: {metric_name}")
    except ValueError:
        return float("nan")


def write_metrics_table(rows: List[Dict[str, float]], output_path: Path) -> pd.DataFrame:
    """Save model metrics to CSV and return the corresponding DataFrame."""

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logging.info("Saved metrics table: %s", output_path)
    return df


# ---------------------------------------------------------------------------
# Data generation and preprocessing
# ---------------------------------------------------------------------------

def generate_synthetic_ids_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Generate a controlled synthetic IDS dataset.

    The synthetic dataset is intentionally harder than many public IDS samples.
    It contains nonlinear structure, redundant features, class imbalance, and
    label noise. This makes it useful for teaching why a simple accuracy number
    is insufficient and why nonlinear models often outperform linear baselines.
    """

    logging.info("Generating synthetic IDS dataset for CISC 650 experiment.")

    x_data, y_data = make_classification(
        n_samples=config.synthetic_samples,
        n_features=config.synthetic_features,
        n_informative=config.synthetic_informative,
        n_redundant=config.synthetic_redundant,
        n_classes=2,
        weights=list(config.benign_attack_ratio),
        flip_y=config.synthetic_label_noise,
        class_sep=config.synthetic_class_sep,
        random_state=config.random_state,
    )

    columns: List[str] = []
    for idx in range(config.synthetic_features):
        if idx < config.synthetic_informative:
            columns.append(f"feat_info_{idx + 1}")
        elif idx < config.synthetic_informative + config.synthetic_redundant:
            columns.append(f"feat_redundant_{idx + 1 - config.synthetic_informative}")
        else:
            columns.append(
                f"feat_noise_{idx + 1 - config.synthetic_informative - config.synthetic_redundant}"
            )

    df = pd.DataFrame(x_data, columns=columns)

    # Make synthetic values resemble non-negative flow measurements. This is not
    # a claim of realism; it simply makes the synthetic table easier to interpret
    # in a networking lab.
    for col in columns:
        df[col] = np.expm1(df[col] * 0.5)

    df = df.clip(lower=0.0)
    df["Label"] = y_data.astype(int)
    return df


def find_cicids_csv_files(data_dir: Path) -> List[Path]:
    """Return candidate CICIDS2017 CSV files from a local directory."""

    if not data_dir.exists():
        return []

    candidates = sorted(
        path for path in data_dir.glob("*.csv")
        if "ISCX" in path.name or "WorkingHours" in path.name or "CIC" in path.name
    )
    return candidates


def load_cicids_dataset(data_dir: Path, max_rows_per_file: Optional[int] = None) -> pd.DataFrame:
    """Load CICIDS2017 CSV files from a local folder.

    The function is intentionally conservative. It loads only CSV files found in
    the user-supplied directory and does not download data from the Internet.
    Large public IDS CSV files can be memory-intensive, so instructors may set
    max_rows_per_file for a shorter classroom run.
    """

    csv_files = find_cicids_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CICIDS-style CSV files found in {data_dir}. "
            "Use --synthetic-only or place CICIDS2017 CSV files in the data directory."
        )

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        logging.info("Loading CICIDS CSV: %s", csv_path.name)
        frame = pd.read_csv(csv_path, nrows=max_rows_per_file, low_memory=False)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    logging.info("Loaded CICIDS-style dataset with %d rows and %d columns.", *combined.shape)
    return combined


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing spaces from CICIDS column names."""

    renamed = {col: str(col).strip() for col in df.columns}
    return df.rename(columns=renamed)


def map_binary_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.Series:
    """Map CICIDS labels to binary values: benign=0, attack=1."""

    if label_col not in df.columns:
        raise KeyError(f"Expected label column '{label_col}' not found.")

    labels = df[label_col].astype(str).str.strip().str.upper()
    return (labels != "BENIGN").astype(int)


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are identifiers, timestamps, or obvious leakage risks.

    Removing these columns helps students focus on transferable flow behavior.
    It does not eliminate all dataset artifacts, which is why SHAP analysis and
    feature ablation are still important.
    """

    drop_keywords = [
        "flow id",
        "src ip",
        "source ip",
        "dst ip",
        "destination ip",
        "timestamp",
        "simillarhttp",
    ]

    cols_to_drop = []
    for col in df.columns:
        lowered = str(col).strip().lower()
        if any(keyword in lowered for keyword in drop_keywords):
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop, errors="ignore")


def prepare_feature_matrix(df: pd.DataFrame, label_col: str = "Label") -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare numeric features and binary labels for model training."""

    df = normalize_column_names(df)
    y = map_binary_labels(df, label_col=label_col) if label_col in df.columns else df[label_col]
    x = df.drop(columns=[label_col], errors="ignore")
    x = remove_identifier_columns(x)

    # Keep numeric columns only. This avoids silently one-hot encoding strings
    # that may function as identifiers.
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0.0)

    return x, y.astype(int)


def split_and_scale(
    x: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Create stratified train/validation/test splits and standardize features."""

    # The two-step split creates train/validation/test partitions while keeping
    # class proportions stable.
    temp_size = config.validation_size + config.test_size
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=temp_size,
        stratify=y,
        random_state=config.random_state,
    )

    relative_test_size = config.test_size / temp_size
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=config.random_state,
    )

    feature_names = list(x_train.columns)
    scaler = StandardScaler().fit(x_train)

    x_train_scaled = np.nan_to_num(scaler.transform(x_train), nan=0.0, posinf=0.0, neginf=0.0)
    x_val_scaled = np.nan_to_num(scaler.transform(x_val), nan=0.0, posinf=0.0, neginf=0.0)
    x_test_scaled = np.nan_to_num(scaler.transform(x_test), nan=0.0, posinf=0.0, neginf=0.0)

    return (
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        feature_names,
    )


# ---------------------------------------------------------------------------
# Modeling and evaluation
# ---------------------------------------------------------------------------

def build_models(config: ExperimentConfig) -> Dict[str, object]:
    """Create the baseline and tree-based models used in the benchmark."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=600,
            class_weight="balanced",
            random_state=config.random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.15,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=config.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model: object, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a trained classifier using standard IDS metrics."""

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": safe_metric_auc("roc", y_test, y_prob),
        "PR-AUC": safe_metric_auc("pr", y_test, y_prob),
    }


def train_and_evaluate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Train all benchmark models and return metrics."""

    models = build_models(config)
    rows: List[Dict[str, float]] = {}

    metric_rows: List[Dict[str, float]] = []
    for name, model in models.items():
        logging.info("Training model: %s", name)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = evaluate_model(name, model, x_test, y_test)
        metrics["TrainSeconds"] = elapsed
        metric_rows.append(metrics)

    return models, pd.DataFrame(metric_rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    models: Dict[str, object],
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC and precision-recall curves.

    The plot uses line styles in addition to color so that it remains readable
    when printed in grayscale, following ACM accessibility guidance.
    """

    line_styles = ["-", "--", "-."]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_value = safe_metric_auc("roc", y_test, y_prob)
        plt.plot(fpr, tpr, linestyle=line_styles[idx % len(line_styles)], label=f"{name} (AUC={auc_value:.3f})")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = safe_metric_auc("pr", y_test, y_prob)
        plt.plot(recall, precision, linestyle=line_styles[idx % len(line_styles)], label=f"{name} (PR-AUC={pr_auc:.3f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved ROC/PR figure: %s", output_path)


def plot_metric_bar(df: pd.DataFrame, label_col: str, metric_col: str, output_path: Path) -> None:
    """Create a compact bar chart for a metrics table."""

    plt.figure(figsize=(8, 4))
    plt.bar(df[label_col].astype(str), df[metric_col].astype(float))
    plt.xlabel(label_col)
    plt.ylabel(metric_col)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved bar chart: %s", output_path)


# ---------------------------------------------------------------------------
# Feature ablation and robustness
# ---------------------------------------------------------------------------

def select_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """Group flow features into interpretable categories.

    The grouping is intentionally heuristic because public datasets use a wide
    range of feature names. The goal is educational interpretation, not a claim
    that these are the only valid network-feature categories.
    """

    groups: Dict[str, List[int]] = {
        "size_based": [],
        "timing_based": [],
        "rate_based": [],
        "flag_window_header": [],
    }

    for idx, name in enumerate(feature_names):
        lower = name.lower()
        if any(token in lower for token in ["len", "byte", "packet length", "tot", "subflow"]):
            groups["size_based"].append(idx)
        if any(token in lower for token in ["iat", "duration", "idle", "active"]):
            groups["timing_based"].append(idx)
        if any(token in lower for token in ["rate", "/s", "per second"]):
            groups["rate_based"].append(idx)
        if any(token in lower for token in ["flag", "win", "header", "port"]):
            groups["flag_window_header"].append(idx)

    # Remove empty groups so smaller synthetic experiments do not fail.
    return {name: indices for name, indices in groups.items() if indices}


def run_feature_ablation(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate XGBoost on feature groups for interpretability."""

    groups = select_feature_groups(feature_names)
    groups["all_features"] = list(range(x_train.shape[1]))

    rows: List[Dict[str, float]] = []
    for group_name, indices in groups.items():
        model = build_models(config)["XGBoost"]
        model.fit(x_train[:, indices], y_train)
        metrics = evaluate_model(group_name, model, x_test[:, indices], y_test)
        metrics["FeatureCount"] = len(indices)
        rows.append(metrics)

    return pd.DataFrame(rows)


def run_robustness_test(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    noise_levels: Iterable[float] = (0.0, 0.01, 0.02, 0.05, 0.10),
) -> pd.DataFrame:
    """Evaluate a trained model under multiplicative feature noise."""

    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, float]] = []

    for level in noise_levels:
        if level == 0:
            perturbed = x_test.copy()
        else:
            noise = rng.normal(loc=1.0, scale=level, size=x_test.shape)
            perturbed = x_test * noise

        metrics = evaluate_model(f"{level:.0%}", model, perturbed, y_test)
        metrics["NoiseLevel"] = level
        rows.append(metrics)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def run_optional_shap_summary(
    model: object,
    x_test: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    max_rows: int = 500,
) -> None:
    """Generate a SHAP summary plot when the SHAP package is installed."""

    try:
        import shap  # Imported lazily because SHAP can be slow to import.
    except ImportError:
        logging.warning("SHAP is not installed; skipping SHAP summary plot.")
        return

    sample = x_test[: min(max_rows, x_test.shape[0])]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # XGBoost binary classifiers usually return a matrix. Random Forest may
    # return a list. The following handles both forms.
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[-1]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(
        shap_values_to_plot,
        sample,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved SHAP summary figure: %s", output_path)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_synthetic_experiment(config: ExperimentConfig, output_dir: Path) -> None:
    """Run the controlled synthetic stress-test experiment."""

    df = generate_synthetic_ids_dataset(config)
    df.to_csv(output_dir / "synthetic_ids.csv", index=False)

    x = df.drop(columns=["Label"])
    y = df["Label"]

    selected_columns = list(x.columns)[: config.retained_synthetic_features]
    x = x[selected_columns]

    x_train, _x_val, x_test, y_train, _y_val, y_test, feature_names = split_and_scale(x, y, config)
    models, metrics_df = train_and_evaluate_models(x_train, x_test, y_train, y_test, config)
    metrics_df.to_csv(output_dir / "synthetic_model_metrics.csv", index=False)

    plot_roc_pr_curves(models, x_test, y_test, output_dir / "synthetic_roc_pr_curves.png")

    # SHAP is run on Random Forest for the synthetic setting because it is fast
    # and stable for classroom demonstration.
    run_optional_shap_summary(models["Random Forest"], x_test, feature_names, output_dir / "synthetic_shap_summary.png")


def run_cicids_experiment(config: ExperimentConfig, data_dir: Path, output_dir: Path, max_rows_per_file: Optional[int]) -> None:
    """Run the CICIDS2017 benchmark if public CSV files are available locally."""

    df = load_cicids_dataset(data_dir, max_rows_per_file=max_rows_per_file)
    x, y = prepare_feature_matrix(df)

    # Stratified sampling keeps the classroom run manageable. If the requested
    # sample is larger than the dataset, the whole dataset is used.
    if len(x) > config.cicids_sample_size:
        x_sample, _, y_sample, _ = train_test_split(
            x,
            y,
            train_size=config.cicids_sample_size,
            stratify=y,
            random_state=config.random_state,
        )
    else:
        x_sample, y_sample = x, y

    x_train, _x_val, x_test, y_train, _y_val, y_test, feature_names = split_and_scale(x_sample, y_sample, config)
    models, metrics_df = train_and_evaluate_models(x_train, x_test, y_train, y_test, config)
    metrics_df.to_csv(output_dir / "cicids_model_metrics.csv", index=False)

    plot_roc_pr_curves(models, x_test, y_test, output_dir / "cicids_roc_pr_curves.png")

    xgb_model = models["XGBoost"]
    ablation_df = run_feature_ablation(x_train, x_test, y_train, y_test, feature_names, config)
    ablation_df.to_csv(output_dir / "cicids_feature_ablation.csv", index=False)

    robustness_df = run_robustness_test(xgb_model, x_test, y_test, config.random_state)
    robustness_df.to_csv(output_dir / "cicids_robustness.csv", index=False)

    plot_metric_bar(ablation_df, "Model", "F1", output_dir / "cicids_feature_ablation_f1.png")
    plot_metric_bar(robustness_df, "Model", "F1", output_dir / "cicids_robustness_f1.png")

    run_optional_shap_summary(xgb_model, x_test, feature_names, output_dir / "cicids_shap_summary.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CISC 650 experiment script."""

    parser = argparse.ArgumentParser(
        description="NSU CISC 650 payload-free IDS benchmark for cybersecurity education."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing optional CICIDS2017 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where result tables and figures will be saved.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only the synthetic stress-test experiment.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row limit per CICIDS CSV file for classroom-scale runs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the NSU CISC 650 reproducible IDS benchmark."""

    configure_logging()
    args = parse_args()

    config = ExperimentConfig(random_state=args.random_state)
    output_dir = ensure_output_dir(args.output_dir)

    logging.info("Starting NSU CISC 650 payload-free IDS benchmark.")
    logging.info("Author: Yuanyuan Liu (Maxine)")
    logging.info("Course: CISC 650 Computer Networks, Nova Southeastern University")

    run_synthetic_experiment(config, output_dir)

    if args.synthetic_only:
        logging.info("Synthetic-only mode selected; CICIDS benchmark skipped.")
        return

    try:
        run_cicids_experiment(config, args.data_dir, output_dir, args.max_rows_per_file)
    except FileNotFoundError as exc:
        logging.warning("%s", exc)
        logging.warning("CICIDS benchmark skipped. Synthetic results were still produced.")

    logging.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
