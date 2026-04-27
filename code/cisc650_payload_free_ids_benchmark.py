#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CISC 650 Computer Networks - Nova Southeastern University (NSU)

Project:
    Explainable Machine Learning for Payload-Free Intrusion Detection:
    A Modular and Reproducible Benchmark for Cybersecurity Education

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
    modular, reproducible, and well-documented benchmark for payload-free
    intrusion detection. It is designed for cybersecurity education: instructors
    and learners can swap datasets, preprocessing choices, classifiers, metrics,
    and visual outputs without rewriting the entire experiment.

Professor-feedback update in this version:
    - Adds additional supervised classifiers beyond the original three models.
    - Adds optional unsupervised baselines using PCA reconstruction error and
      K-Means distance-to-cluster-center scoring.
    - Makes the experiment pipeline more explicitly modular: input data,
      preprocessing, feature handling, model execution, metrics, plotting, and
      explainability are separated into reusable functions.
    - Supports user-supplied flow-level CSV data through a configurable label
      column, in addition to the synthetic stress test and CICIDS-style CSVs.

Important scope note:
    This script does not inspect packet payloads and does not claim to process
    authentic encrypted healthcare packet captures. It works with flow-level
    metadata features appropriate for privacy-aware cybersecurity education and
    reproducible experimentation. CICIDS2017 is not treated as a healthcare
    dataset.

Typical usage:
    1) Synthetic stress test only:
        python code/cisc650_payload_free_ids_benchmark.py --dataset synthetic --output-dir outputs

    2) Synthetic stress test with optional unsupervised baselines:
        python code/cisc650_payload_free_ids_benchmark.py --dataset synthetic --include-unsupervised --output-dir outputs

    3) CICIDS-style benchmark with CSV files stored in ./data:
        python code/cisc650_payload_free_ids_benchmark.py --dataset cicids --data-dir data --output-dir outputs

    4) User-supplied flow-level CSV file:
        python code/cisc650_payload_free_ids_benchmark.py --dataset user --input-csv data/my_flows.csv --label-column Label --output-dir outputs

Outputs:
    - CSV metric tables
    - ROC / precision-recall plots
    - feature-group ablation results when feature names support grouping
    - robustness results under feature perturbation
    - optional SHAP summary plot when the SHAP package is installed
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost is optional for lightweight classroom runs.
    XGBClassifier = None


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Centralized configuration for the NSU CISC 650 IDS experiments.

    The default values reproduce a controlled classroom-scale benchmark. They
    can be changed from the command line when instructors or students want to
    explore an easier, harder, smaller, or larger classification setting.
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
    mlp_max_iter: int = 250
    kmeans_clusters: int = 8
    pca_components: int = 10


@dataclass(frozen=True)
class ModelSpec:
    """Metadata wrapper for a model used in the benchmark.

    The metadata is useful for education because it lets the output table show
    not only performance, but also the learning family behind each method.
    """

    name: str
    estimator: object
    family: str
    paradigm: str


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure readable command-line logging for experiment progress."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def ensure_output_dir(path: Path) -> Path:
    """Create the output directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_auc(metric_name: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC or PR-AUC safely.

    In very small classroom experiments, a split can occasionally contain only
    one class. In that case ROC-AUC is undefined. Returning NaN is more useful
    than crashing because the output table still explains the failed condition.
    """

    try:
        if metric_name == "roc":
            return float(roc_auc_score(y_true, y_score))
        if metric_name == "pr":
            return float(average_precision_score(y_true, y_score))
        raise ValueError(f"Unsupported AUC metric: {metric_name}")
    except ValueError:
        return float("nan")


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame and log the destination."""

    df.to_csv(output_path, index=False)
    logging.info("Saved table: %s", output_path)


# ---------------------------------------------------------------------------
# Data generation and loading modules
# ---------------------------------------------------------------------------

def generate_synthetic_ids_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Generate a controlled synthetic IDS dataset.

    The synthetic dataset is intentionally harder than many toy examples. It
    includes nonlinear structure, redundant features, class imbalance, and label
    noise. This makes it useful for teaching why accuracy alone is insufficient
    and why different model families behave differently.
    """

    logging.info("Generating synthetic IDS dataset.")
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

    # Convert synthetic values into non-negative, skewed measurements so the
    # table is easier to interpret as flow-like data in a networking lab. This
    # is not a claim that these are real traffic flows.
    for col in columns:
        df[col] = np.expm1(df[col] * 0.5)
    df = df.clip(lower=0.0)
    df["Label"] = y_data.astype(int)
    return df


def find_cicids_csv_files(data_dir: Path) -> List[Path]:
    """Return candidate CICIDS-style CSV files from a local directory."""

    if not data_dir.exists():
        return []
    return sorted(
        path for path in data_dir.glob("*.csv")
        if "ISCX" in path.name or "WorkingHours" in path.name or "CIC" in path.name
    )


def load_cicids_dataset(data_dir: Path, max_rows_per_file: Optional[int] = None) -> pd.DataFrame:
    """Load CICIDS-style CSV files from a local folder.

    This function intentionally does not download data. Large public IDS files
    should be downloaded by the user and placed in the data directory. This keeps
    the GitHub repository small and avoids redistributing large datasets.
    """

    csv_files = find_cicids_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CICIDS-style CSV files found in {data_dir}. "
            "Use --dataset synthetic or place CICIDS CSV files in the data directory."
        )

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        logging.info("Loading CICIDS CSV: %s", csv_path.name)
        frames.append(pd.read_csv(csv_path, nrows=max_rows_per_file, low_memory=False))

    combined = pd.concat(frames, ignore_index=True)
    logging.info("Loaded CICIDS-style dataset with %d rows and %d columns.", *combined.shape)
    return combined


def load_user_csv(input_csv: Path) -> pd.DataFrame:
    """Load a user-supplied flow-level CSV file.

    This is the main extension point requested in the professor feedback: users
    can integrate their own data as long as the file has numeric flow-level
    features and a binary or string label column.
    """

    if not input_csv.exists():
        raise FileNotFoundError(f"User CSV file not found: {input_csv}")
    logging.info("Loading user CSV: %s", input_csv)
    return pd.read_csv(input_csv, low_memory=False)


# ---------------------------------------------------------------------------
# Preprocessing and feature modules
# ---------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names."""

    return df.rename(columns={col: str(col).strip() for col in df.columns})


def map_binary_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.Series:
    """Map labels to binary values: benign/normal=0 and attack/anomaly=1."""

    if label_col not in df.columns:
        raise KeyError(f"Expected label column '{label_col}' not found.")

    raw = df[label_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return (numeric.fillna(0).astype(float) > 0).astype(int)

    labels = raw.astype(str).str.strip().str.upper()
    benign_values = {"BENIGN", "NORMAL", "0", "FALSE", "LEGITIMATE"}
    return (~labels.isin(benign_values)).astype(int)


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers, timestamps, and obvious leakage-risk columns."""

    drop_keywords = [
        "flow id", "src ip", "source ip", "dst ip", "destination ip",
        "timestamp", "simillarhttp", "source port", "destination port",
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
    y = map_binary_labels(df, label_col=label_col)
    x = df.drop(columns=[label_col], errors="ignore")
    x = remove_identifier_columns(x)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, y.astype(int)


def split_and_scale(
    x: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Create stratified train/validation/test splits and standardize features."""

    temp_size = config.validation_size + config.test_size
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=temp_size, stratify=y, random_state=config.random_state
    )

    relative_test_size = config.test_size / temp_size
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=config.random_state
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
# Model modules
# ---------------------------------------------------------------------------

def build_supervised_models(config: ExperimentConfig, include_xgboost: bool = True) -> List[ModelSpec]:
    """Create supervised classifiers used in the modular benchmark.

    The list intentionally covers different learning assumptions so that the
    benchmark can be used as a teaching tool rather than only a leaderboard.
    """

    models = [
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(max_iter=800, class_weight="balanced", random_state=config.random_state),
            "linear",
            "supervised",
        ),
        ModelSpec(
            "KNN",
            KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "distance-based",
            "supervised",
        ),
        ModelSpec(
            "Naive Bayes",
            GaussianNB(),
            "probabilistic",
            "supervised",
        ),
        ModelSpec(
            "Linear SVM",
            LinearSVC(class_weight="balanced", random_state=config.random_state, max_iter=6000),
            "margin-based",
            "supervised",
        ),
        ModelSpec(
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                early_stopping=True,
                max_iter=config.mlp_max_iter,
                random_state=config.random_state,
            ),
            "neural network",
            "supervised",
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                class_weight="balanced",
                random_state=config.random_state,
                n_jobs=-1,
            ),
            "bagging ensemble",
            "supervised",
        ),
    ]

    if include_xgboost:
        if XGBClassifier is None:
            logging.warning("XGBoost is not installed; skipping the XGBoost classifier.")
        else:
            models.append(
                ModelSpec(
                    "XGBoost",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=5,
                        learning_rate=0.15,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        tree_method="hist",
                        random_state=config.random_state,
                        eval_metric="logloss",
                        n_jobs=1,
                    ),
                    "boosting ensemble",
                    "supervised",
                )
            )

    return models


class PCAAnomalyDetector:
    """Simple PCA reconstruction-error anomaly detector.

    Higher reconstruction error is treated as a stronger anomaly score. This is
    included as an educational unsupervised baseline rather than as an optimized
    production IDS method.
    """

    def __init__(self, n_components: int, random_state: int) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca: Optional[PCA] = None

    def fit(self, x_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "PCAAnomalyDetector":
        n_components = min(self.n_components, x_train.shape[1], max(1, x_train.shape[0] - 1))
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca.fit(x_train)
        return self

    def score_anomaly(self, x_data: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCAAnomalyDetector must be fitted before scoring.")
        reconstructed = self.pca.inverse_transform(self.pca.transform(x_data))
        return np.mean((x_data - reconstructed) ** 2, axis=1)


class KMeansAnomalyDetector:
    """K-Means distance-based anomaly detector.

    The detector can use only benign training samples when labels are available.
    This mirrors a common anomaly-detection teaching setup: learn normal behavior
    and then score deviations from it.
    """

    def __init__(self, n_clusters: int, random_state: int) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None

    def fit(self, x_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "KMeansAnomalyDetector":
        if y_train is not None and np.any(y_train == 0):
            training_data = x_train[y_train == 0]
        else:
            training_data = x_train
        n_clusters = min(self.n_clusters, max(1, training_data.shape[0]))
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans.fit(training_data)
        return self

    def score_anomaly(self, x_data: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("KMeansAnomalyDetector must be fitted before scoring.")
        distances = self.kmeans.transform(x_data)
        return np.min(distances, axis=1)


def build_unsupervised_models(config: ExperimentConfig) -> List[ModelSpec]:
    """Create optional unsupervised baselines for teaching comparison."""

    return [
        ModelSpec(
            "PCA Reconstruction",
            PCAAnomalyDetector(n_components=config.pca_components, random_state=config.random_state),
            "dimensionality-reduction anomaly score",
            "unsupervised",
        ),
        ModelSpec(
            "K-Means Distance",
            KMeansAnomalyDetector(n_clusters=config.kmeans_clusters, random_state=config.random_state),
            "clustering anomaly score",
            "unsupervised",
        ),
    ]


def build_model_registry(config: ExperimentConfig, include_unsupervised: bool = False, include_xgboost: bool = True) -> List[ModelSpec]:
    """Return the complete model registry for one benchmark run."""

    registry = build_supervised_models(config, include_xgboost=include_xgboost)
    if include_unsupervised:
        registry.extend(build_unsupervised_models(config))
    return registry


# ---------------------------------------------------------------------------
# Evaluation modules
# ---------------------------------------------------------------------------

def get_model_scores(model: object, x_data: np.ndarray) -> np.ndarray:
    """Return continuous attack/anomaly scores for ROC and PR curves."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_data)
    if hasattr(model, "score_anomaly"):
        return model.score_anomaly(x_data)
    raise TypeError(f"Model type does not expose scoring method: {type(model)}")


def choose_threshold_from_validation(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    """Select a threshold that maximizes validation F1.

    This is used for unsupervised/anomaly-score models and any model that does
    not expose a direct binary classifier decision.
    """

    candidate_thresholds = np.quantile(val_scores, np.linspace(0.05, 0.95, 91))
    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        preds = (val_scores >= threshold).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def predict_binary(model: object, x_data: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Return binary predictions for supervised or score-based models."""

    if threshold is not None:
        return (get_model_scores(model, x_data) >= threshold).astype(int)
    if hasattr(model, "predict"):
        return model.predict(x_data).astype(int)
    return (get_model_scores(model, x_data) >= 0.5).astype(int)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute standard IDS evaluation metrics."""

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": safe_auc("roc", y_true, y_score),
        "PR-AUC": safe_auc("pr", y_true, y_score),
    }


def train_and_evaluate_models(
    model_specs: Sequence[ModelSpec],
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Train all selected models and return the fitted models and metrics."""

    fitted_models: Dict[str, object] = {}
    rows: List[Dict[str, float]] = []

    for spec in model_specs:
        logging.info("Training model: %s", spec.name)
        start_train = time.perf_counter()

        if spec.paradigm == "unsupervised" and hasattr(spec.estimator, "fit"):
            spec.estimator.fit(x_train, y_train)
        else:
            spec.estimator.fit(x_train, y_train)

        train_seconds = time.perf_counter() - start_train

        threshold: Optional[float] = None
        if spec.paradigm == "unsupervised" or not hasattr(spec.estimator, "predict"):
            val_scores = get_model_scores(spec.estimator, x_val)
            threshold = choose_threshold_from_validation(y_val, val_scores)

        start_score = time.perf_counter()
        y_score = get_model_scores(spec.estimator, x_test)
        y_pred = predict_binary(spec.estimator, x_test, threshold=threshold)
        score_seconds = time.perf_counter() - start_score

        metrics = evaluate_predictions(y_test, y_pred, y_score)
        metrics.update({
            "Model": spec.name,
            "Family": spec.family,
            "Paradigm": spec.paradigm,
            "Threshold": threshold if threshold is not None else np.nan,
            "TrainSeconds": train_seconds,
            "ScoreSeconds": score_seconds,
        })
        rows.append(metrics)
        fitted_models[spec.name] = spec.estimator

    metrics_df = pd.DataFrame(rows)
    ordered_cols = [
        "Model", "Family", "Paradigm", "Accuracy", "Precision", "Recall", "F1",
        "ROC-AUC", "PR-AUC", "Threshold", "TrainSeconds", "ScoreSeconds",
    ]
    return fitted_models, metrics_df[ordered_cols]


# ---------------------------------------------------------------------------
# Visualization and explainability modules
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    models: Dict[str, object],
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC and precision-recall curves for all benchmark models."""

    line_styles = ["-", "--", "-.", ":"]
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    for idx, (name, model) in enumerate(models.items()):
        y_score = get_model_scores(model, x_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_value = safe_auc("roc", y_test, y_score)
        plt.plot(fpr, tpr, linestyle=line_styles[idx % len(line_styles)], linewidth=1.4,
                 label=f"{name} (AUC={auc_value:.3f})")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=7)

    plt.subplot(1, 2, 2)
    for idx, (name, model) in enumerate(models.items()):
        y_score = get_model_scores(model, x_test)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = safe_auc("pr", y_test, y_score)
        plt.plot(recall, precision, linestyle=line_styles[idx % len(line_styles)], linewidth=1.4,
                 label=f"{name} (PR-AUC={pr_auc:.3f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved ROC/PR figure: %s", output_path)


def plot_metric_bar(df: pd.DataFrame, label_col: str, metric_col: str, output_path: Path) -> None:
    """Create a compact bar chart for a metrics table."""

    plt.figure(figsize=(9, 4))
    plt.bar(df[label_col].astype(str), df[metric_col].astype(float))
    plt.xlabel(label_col)
    plt.ylabel(metric_col)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved bar chart: %s", output_path)


def run_optional_shap_summary(
    model: object,
    x_test: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    max_rows: int = 500,
) -> None:
    """Generate a SHAP summary plot when SHAP is installed.

    SHAP is computationally heavier than the core benchmark. The function is
    optional so the repository remains usable in lightweight classroom settings.
    """

    try:
        import shap
    except ImportError:
        logging.warning("SHAP is not installed; skipping SHAP summary plot.")
        return

    sample = x_test[: min(max_rows, x_test.shape[0])]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[-1]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(shap_values_to_plot, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved SHAP summary figure: %s", output_path)


# ---------------------------------------------------------------------------
# Feature ablation and robustness modules
# ---------------------------------------------------------------------------

def select_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """Group features into interpretable categories when possible."""

    groups: Dict[str, List[int]] = {
        "size_based": [],
        "timing_based": [],
        "rate_based": [],
        "flag_window_header": [],
        "synthetic_informative": [],
        "synthetic_redundant": [],
        "synthetic_noise": [],
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
        if lower.startswith("feat_info"):
            groups["synthetic_informative"].append(idx)
        if lower.startswith("feat_redundant"):
            groups["synthetic_redundant"].append(idx)
        if lower.startswith("feat_noise"):
            groups["synthetic_noise"].append(idx)

    return {name: indices for name, indices in groups.items() if indices}


def run_feature_ablation(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate XGBoost over feature groups for educational interpretation."""

    groups = select_feature_groups(feature_names)
    groups["all_features"] = list(range(x_train.shape[1]))
    rows: List[Dict[str, float]] = []

    for group_name, indices in groups.items():
        if XGBClassifier is not None:
            model = XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.15,
                tree_method="hist",
                random_state=config.random_state,
                eval_metric="logloss",
                n_jobs=1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=120,
                max_depth=10,
                class_weight="balanced",
                random_state=config.random_state,
                n_jobs=-1,
            )
        model.fit(x_train[:, indices], y_train)
        y_score = get_model_scores(model, x_test[:, indices])
        y_pred = model.predict(x_test[:, indices])
        metrics = evaluate_predictions(y_test, y_pred, y_score)
        metrics.update({"FeatureGroup": group_name, "FeatureCount": len(indices)})
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
        y_score = get_model_scores(model, perturbed)
        y_pred = predict_binary(model, perturbed)
        metrics = evaluate_predictions(y_test, y_pred, y_score)
        metrics.update({"NoiseLevel": level, "Model": f"{level:.0%} noise"})
        rows.append(metrics)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_benchmark(
    df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    label_col: str,
    run_name: str,
    include_unsupervised: bool,
    include_xgboost: bool,
    run_shap: bool,
) -> None:
    """Run the full modular benchmark on a prepared DataFrame."""

    x, y = prepare_feature_matrix(df, label_col=label_col)

    # Keep synthetic feature count aligned with the original project design.
    if run_name == "synthetic" and x.shape[1] > config.retained_synthetic_features:
        x = x.iloc[:, : config.retained_synthetic_features]

    x_train, x_val, x_test, y_train, y_val, y_test, feature_names = split_and_scale(x, y, config)
    model_specs = build_model_registry(config, include_unsupervised=include_unsupervised, include_xgboost=include_xgboost)
    fitted_models, metrics_df = train_and_evaluate_models(
        model_specs, x_train, x_val, x_test, y_train, y_val, y_test
    )

    save_dataframe(metrics_df, output_dir / f"{run_name}_model_metrics.csv")
    plot_roc_pr_curves(fitted_models, x_test, y_test, output_dir / f"{run_name}_roc_pr_curves.png")
    plot_metric_bar(metrics_df, "Model", "F1", output_dir / f"{run_name}_model_f1.png")

    # Feature ablation and robustness are run using XGBoost to keep these
    # secondary analyses focused and computationally manageable.
    ablation_df = run_feature_ablation(x_train, x_val, x_test, y_train, y_val, y_test, feature_names, config)
    save_dataframe(ablation_df, output_dir / f"{run_name}_feature_ablation.csv")
    plot_metric_bar(ablation_df, "FeatureGroup", "F1", output_dir / f"{run_name}_feature_ablation_f1.png")

    secondary_model_name = "XGBoost" if "XGBoost" in fitted_models else "Random Forest"
    if secondary_model_name in fitted_models:
        secondary_model = fitted_models[secondary_model_name]
        robustness_df = run_robustness_test(secondary_model, x_test, y_test, config.random_state)
        save_dataframe(robustness_df, output_dir / f"{run_name}_robustness.csv")
        plot_metric_bar(robustness_df, "Model", "F1", output_dir / f"{run_name}_robustness_f1.png")

        if run_shap and secondary_model_name in {"XGBoost", "Random Forest"}:
            run_optional_shap_summary(
                secondary_model, x_test, feature_names, output_dir / f"{run_name}_shap_summary.png"
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CISC 650 experiment script."""

    parser = argparse.ArgumentParser(
        description="NSU CISC 650 modular payload-free IDS benchmark for cybersecurity education."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "cicids", "user"],
        default="synthetic",
        help="Dataset module to run.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CICIDS-style CSV files.")
    parser.add_argument("--input-csv", type=Path, default=None, help="User-supplied flow-level CSV file.")
    parser.add_argument("--label-column", type=str, default="Label", help="Name of the label column in the input data.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for result tables and figures.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row limit per CICIDS CSV file.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--synthetic-samples", type=int, default=40_000, help="Number of synthetic samples.")
    parser.add_argument("--include-unsupervised", action="store_true", help="Include PCA and K-Means anomaly baselines.")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost for lightweight environments or quick classroom runs.")
    parser.add_argument("--run-shap", action="store_true", help="Generate SHAP summary plots when SHAP is installed.")
    return parser.parse_args()


def main() -> None:
    """Entry point for the NSU CISC 650 modular IDS benchmark."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig(random_state=args.random_state, synthetic_samples=args.synthetic_samples)
    output_dir = ensure_output_dir(args.output_dir)

    logging.info("Starting NSU CISC 650 modular payload-free IDS benchmark.")
    logging.info("Author: Yuanyuan Liu (Maxine)")
    logging.info("Course: CISC 650 Computer Networks, Nova Southeastern University")

    if args.dataset == "synthetic":
        df = generate_synthetic_ids_dataset(config)
        df.to_csv(output_dir / "synthetic_ids.csv", index=False)
        run_benchmark(df, config, output_dir, "Label", "synthetic", args.include_unsupervised, not args.skip_xgboost, args.run_shap)
    elif args.dataset == "cicids":
        df = load_cicids_dataset(args.data_dir, max_rows_per_file=args.max_rows_per_file)
        run_benchmark(df, config, output_dir, "Label", "cicids", args.include_unsupervised, not args.skip_xgboost, args.run_shap)
    elif args.dataset == "user":
        if args.input_csv is None:
            raise ValueError("--input-csv is required when --dataset user is selected.")
        df = load_user_csv(args.input_csv)
        run_benchmark(df, config, output_dir, args.label_column, "user", args.include_unsupervised, not args.skip_xgboost, args.run_shap)
    else:
        raise ValueError(f"Unsupported dataset option: {args.dataset}")

    logging.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
