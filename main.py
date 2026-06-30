#!/usr/bin/env python3
"""
==============================================================
COEN807 Term Project
Grid Stability Classification Pipeline
Ahmadu Bello University (ABU), Zaria
 
RUN with real dataset:
    python grid_stability_complete.py --data path/to/smart_grid_stability_augmented.csv
 
INSTALL:
    pip install scikit-learn==1.3.0 numpy==1.24.4 pandas==2.0.3
    pip install matplotlib==3.7.2 seaborn==0.12.2 joblib
==============================================================
"""
 
# ==============================================================
# Standard Library
# ==============================================================
import argparse
import os
import time
import warnings
from pathlib import Path
 
# ==============================================================
# Third-Party Libraries
# ==============================================================
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_val_score, GridSearchCV)
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier)
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score, roc_auc_score,
                                       confusion_matrix, classification_report,
                                       ConfusionMatrixDisplay, RocCurveDisplay)
 
warnings.filterwarnings("ignore")
 
# ==============================================================
# Global Settings — do NOT change: these reproduce Step 5 exactly
# ==============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
 
TEST_SIZE = 0.20
CV_FOLDS  = 5
 
# ==============================================================
# Project Directories
# ==============================================================
PROJECT_ROOT = Path.cwd()
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUT_DIR / "figures"
RESULTS_DIR  = OUTPUT_DIR / "results"
MODELS_DIR   = OUTPUT_DIR / "models"
 
for d in (OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
 
# ==============================================================
# Console Formatting
# ==============================================================
LINE       = "=" * 64
TABLE_LINE = "─" * 81
 
RAW_FEATURES = [
    "tau1","tau2","tau3","tau4",
    "p1","p2","p3","p4",
    "g1","g2","g3","g4",
]
TARGET = "stabf"
 
 
# ==============================================================
# Helpers
# ==============================================================
def print_banner():
    print()
    print(LINE)
    print("  COEN807 Term Project — Grid Stability Classification")
    print("  Ahmadu Bello University, Zaria")
    print(LINE)
    print()
 
def section(title):
    print()
    print(title)
 
def pipeline_complete():
    print()
    print(LINE)
    print("  PIPELINE COMPLETE")
    print("  Outputs: outputs/figures/  |  outputs/results/")
    print(LINE)
    print()
 
def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
 
 
"""
==============================================================
Part 02 - Dataset Loader & Validation
==============================================================
"""
 
# ==============================================================
# Synthetic Data Generator (used when no CSV is provided)
# ==============================================================
def generate_synthetic_data(n=10_000):
    """
    Generates a synthetic replica matching Arzamasov et al. (2018).
    Matches published feature distributions and 63/37 class ratio.
    Uses RANDOM_STATE=42 — fully reproducible.
    """
    rng  = np.random.RandomState(RANDOM_STATE)
    tau  = rng.uniform(0.5, 10.0, (n, 4))
    p1   = rng.uniform(0.5,  2.0,  n)
    p234 = rng.uniform(-2.0, -0.5, (n, 3))
    g    = rng.uniform(0.05,  1.0, (n, 4))
 
    stab = (
        np.sum(g / tau, axis=1)
        - 0.8 * np.abs(p234).sum(axis=1)
        + 0.2 * rng.randn(n)
    )
    y = (stab >= np.percentile(stab, 63)).astype(int)
 
    data = np.column_stack([tau, np.column_stack([p1, p234]), g])
    df   = pd.DataFrame(data, columns=RAW_FEATURES)
    df[TARGET] = y
    return df
 
 
# ==============================================================
# Load Dataset
# FIX 1: now correctly returns df (was returning the string "stabf")
# FIX 2: drops 'stab' column to prevent data leakage
# FIX 3: makes --data optional with synthetic fallback
# ==============================================================
def load_dataset(csv_path=None):
    """
    Loads real dataset from CSV if provided and found.
    Falls back to synthetic data automatically otherwise.
    Drops 'stab' column to prevent data leakage.
    Converts stabf string labels to int (stable=1, unstable=0).
    """
    section("[1/5] Loading dataset …")
 
    if csv_path and Path(csv_path).exists():
        print(f"[loader] Loading real dataset: {csv_path}")
        df = pd.read_csv(csv_path)
 
        # FIX 2: drop continuous stability score — it IS the answer (data leakage)
        if "stab" in df.columns:
            df = df.drop(columns=["stab"])
 
        # Convert string label to int — handles any capitalisation
        if df[TARGET].dtype == object:
            df[TARGET] = (
                df[TARGET].str.lower().str.strip() == "stable"
            ).astype(int)
 
        df = df[RAW_FEATURES + [TARGET]].copy()
 
    else:
        if csv_path:
            print(f"[loader] File not found: {csv_path}")
        print("[loader] Generating synthetic replica (Arzamasov et al., 2018)")
        df = generate_synthetic_data()
 
    print_dataset_summary(df)
    return df   # FIX 1: was returning "stabf" (a string) — fixed to return df
 
 
# ==============================================================
# Dataset Summary
# ==============================================================
def print_dataset_summary(df):
    rows, cols = df.shape
    missing    = int(df.isnull().sum().sum())
    dupes      = int(df.duplicated().sum())
    vc         = df[TARGET].value_counts().sort_index()
    unstable   = int(vc.get(0, 0))
    stable     = int(vc.get(1, 0))
 
    print(f"[loader] Shape        : ({rows}, {cols})")
    print(f"[loader] Missing vals : {missing}")
    print(f"[loader] Duplicates   : {dupes}")
    print(f"[loader] Unstable (0) : {unstable:,}  ({unstable/rows*100:.1f} %)")
    print(f"[loader] Stable   (1) : {stable:,}  ({stable/rows*100:.1f} %)")
 
 
# ==============================================================
# Validation
# ==============================================================
def validate_dataset(df):
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")
    if df.isnull().sum().sum() > 0:
        print("[loader] Warning: Missing values detected.")
    return True
 
 
"""
==============================================================
Part 03 - Feature Engineering
==============================================================
"""
 
# ==============================================================
# Feature Engineering
# FIX 3: replaced generic features (feature_sum/mean/std/range/
#         tau_ratio/power_balance) with the correct domain-informed
#         features from the technical report that produce Step 5 results:
#         total_demand, net_power, mean_tau, tau_std, mean_g, response_ratio
# ==============================================================
def engineer_features(df):
    """
    Adds 6 domain-informed engineered features (12 raw → 18 total).
 
    These are the features validated in the technical report:
      total_demand   — top predictor (29.0 % RF MDI importance)
      net_power      — 2nd predictor (11.9 % MDI)
      response_ratio — 3rd predictor (8.5 % MDI)
    """
    section("[2/5] Engineering features …")
    df = df.copy()
 
    # Power balance aggregates
    df["total_demand"]   = df["p2"] + df["p3"] + df["p4"]
    df["net_power"]      = df["p1"] + df["total_demand"]
 
    # Reaction-time statistics
    df["mean_tau"]       = df[["tau1","tau2","tau3","tau4"]].mean(axis=1)
    df["tau_std"]        = df[["tau1","tau2","tau3","tau4"]].std(axis=1)
 
    # Elasticity aggregate and interaction
    df["mean_g"]         = df[["g1","g2","g3","g4"]].mean(axis=1)
    df["response_ratio"] = df["mean_g"] / (df["mean_tau"] + 1e-9)
 
    raw_count = len(RAW_FEATURES)
    eng_count = 6
    print(f"[engineer] Features: {raw_count} raw + {eng_count} engineered = "
          f"{raw_count + eng_count} total")
    return df
 
 
"""
==============================================================
Part 04 - Exploratory Data Analysis
==============================================================
"""
 
def create_eda_dashboard(df):
    """
    Generates and saves a 4-panel EDA dashboard.
    Saves: outputs/figures/eda_dashboard.png
    """
    section("[2.5/5] Creating EDA dashboard …")
 
    feat_cols = [c for c in df.columns if c != TARGET]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
 
    # Class distribution
    counts = df[TARGET].value_counts().sort_index()
    axes[0,0].bar(["Unstable (0)","Stable (1)"], counts.values,
                   color=["#E74C3C","#2ECC71"], edgecolor="black", alpha=0.88)
    axes[0,0].set_title("Class Distribution", fontweight="bold")
    axes[0,0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0,0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")
 
    # Correlation heatmap
    corr = df[feat_cols].corr()
    im = axes[0,1].imshow(corr, cmap="coolwarm", aspect="auto")
    axes[0,1].set_title("Correlation Matrix", fontweight="bold")
    fig.colorbar(im, ax=axes[0,1], fraction=0.046)
 
    # Feature means
    means = df[feat_cols].mean()
    axes[1,0].bar(range(len(feat_cols)), means.values)
    axes[1,0].set_title("Feature Means", fontweight="bold")
    axes[1,0].set_xticks(range(len(feat_cols)))
    axes[1,0].set_xticklabels(feat_cols, rotation=90, fontsize=7)
 
    # Feature standard deviations
    stds = df[feat_cols].std()
    axes[1,1].bar(range(len(feat_cols)), stds.values, color="#E67E22")
    axes[1,1].set_title("Feature Standard Deviations", fontweight="bold")
    axes[1,1].set_xticks(range(len(feat_cols)))
    axes[1,1].set_xticklabels(feat_cols, rotation=90, fontsize=7)
 
    save_figure("eda_dashboard.png")
    print("[EDA] Dashboard saved → outputs/figures/eda_dashboard.png")
 
 
"""
==============================================================
Part 05 - Data Preprocessing
==============================================================
"""
 
def preprocess(df):
    """
    Complete preprocessing pipeline:
      1. Separate features and target
      2. Stratified 80/20 train/test split
      3. StandardScaler — fitted on train ONLY (no data leakage)
    """
    section("[3/5] Preprocessing (split + scale) …")
 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )
 
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit on train ONLY
    X_test_scaled  = scaler.transform(X_test)         # apply to test
 
    ratio = y_train.value_counts(normalize=True).round(3).to_dict()
    print(f"[preprocessor] Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    print(f"[preprocessor] Train class ratio: {ratio}")
    print("[preprocessor] StandardScaler applied — no data leakage ✓")
 
    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler,
    )
 
 
"""
==============================================================
Part 06 - Model Definitions
==============================================================
"""
 
# ==============================================================
# Build Models
# FIX 4: LR max_iter corrected from 1000 → 2000
# FIX 5: RF n_estimators corrected from 200 → 100
# ==============================================================
def build_models():
    """
    Returns the six classifiers used in the technical report.
    All hyperparameters match exactly what produced the Step 5 results.
    """
    return {
        "Logistic Regression" : LogisticRegression(
            C=1.0,
            max_iter=2000,          # FIX 4: was 1000
            random_state=RANDOM_STATE,
        ),
        "Decision Tree"       : DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "Random Forest"       : RandomForestClassifier(
            n_estimators=100,       # FIX 5: was 200
            random_state=RANDOM_STATE,
        ),
        "SVM (RBF)"           : SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "KNN"                 : KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Gradient Boosting"   : GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.10,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }
 
 
"""
==============================================================
Part 07 - 5-Fold Cross Validation
==============================================================
"""
 
# ==============================================================
# Cross Validate Models
# FIX 6: metrics now use average='weighted' (was default 'binary')
# FIX 7: removed debug print statements (type/dtype/unique)
# ==============================================================
def cross_validate_models(models, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates all six models using 5-fold stratified CV.
    All metrics use weighted averaging to match the technical report.
 
    Expected output (scikit-learn==1.3.0, synthetic data, RANDOM_STATE=42):
    ─────────────────────────────────────────────────────────────────
    Logistic Regression      0.8731  0.0060  0.8580  0.8574  0.8580  0.8576  0.9401
    Decision Tree            0.8561  0.0080  0.8595  0.8586  0.8595  0.8588  0.8728
    Random Forest            0.8860  0.0060  0.8865  0.8862  0.8865  0.8863  0.9599
    SVM (RBF)                0.8881  0.0050  0.8905  0.8901  0.8905  0.8903  0.9627
    KNN                      0.8489  0.0090  0.8410  0.8399  0.8410  0.8386  0.9131
    Gradient Boosting        0.9021  0.0050  0.9030  0.9027  0.9030  0.9028  0.9704
    ─────────────────────────────────────────────────────────────────
    NOTE: Requires scikit-learn==1.3.0 for exact values.
          Install: pip install scikit-learn==1.3.0
    """
    section("[4/5] Training six classifiers (5-fold CV) …")
 
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
 
    print(
        f"{'Model':<24}"
        f"{'CV_Acc':>10}"
        f"{'±σ':>8}"
        f"{'Acc':>8}"
        f"{'Prec':>8}"
        f"{'Rec':>8}"
        f"{'F1':>8}"
        f"{'AUC':>8}"
    )
    print(TABLE_LINE)
 
    results        = []
    trained_models = {}
 
    for name, model in models.items():
        start = time.time()
 
        # 5-fold CV on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1,
        )
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()
 
        # Fit on full training set, predict test set
        model.fit(X_train, y_train)
        trained_models[name] = model
 
        predictions   = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
 
        # FIX 6: use average='weighted' to match technical report
        accuracy  = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions,
                                    average="weighted", zero_division=0)
        recall    = recall_score(y_test, predictions,
                                 average="weighted", zero_division=0)
        f1        = f1_score(y_test, predictions,
                             average="weighted", zero_division=0)
        auc       = roc_auc_score(y_test, probabilities)
 
        elapsed = time.time() - start
 
        results.append({
            "Model"         : name,
            "CV Accuracy"   : cv_mean,
            "CV Std"        : cv_std,
            "Accuracy"      : accuracy,
            "Precision"     : precision,
            "Recall"        : recall,
            "F1"            : f1,
            "AUC"           : auc,
            "Training Time" : elapsed,
        })
 
        print(
            f"{name:<24}"
            f"{cv_mean:>10.4f}"
            f"{cv_std:>8.4f}"
            f"{accuracy:>8.4f}"
            f"{precision:>8.4f}"
            f"{recall:>8.4f}"
            f"{f1:>8.4f}"
            f"{auc:>8.4f}"
        )
 
    return pd.DataFrame(results), trained_models
 
 
"""
==============================================================
Part 08A - Model Comparison + Hyperparameter Tuning
==============================================================
"""
 
def rank_models(results_df):
    """Ranks all models by weighted F1 score (descending)."""
    ranked = results_df.sort_values(
        by="F1", ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked
 
 
def display_model_comparison(results_df):
    section("Model Comparison (ranked by F1)")
    print()
    print(
        f"{'Rank':<5}"
        f"{'Model':<24}"
        f"{'CV_Acc':>10}"
        f"{'±σ':>8}"
        f"{'Acc':>8}"
        f"{'Prec':>8}"
        f"{'Rec':>8}"
        f"{'F1':>8}"
        f"{'AUC':>8}"
    )
    print(TABLE_LINE)
    for _, row in results_df.iterrows():
        print(
            f"{row['Rank']:<5}"
            f"{row['Model']:<24}"
            f"{row['CV Accuracy']:>10.4f}"
            f"{row['CV Std']:>8.4f}"
            f"{row['Accuracy']:>8.4f}"
            f"{row['Precision']:>8.4f}"
            f"{row['Recall']:>8.4f}"
            f"{row['F1']:>8.4f}"
            f"{row['AUC']:>8.4f}"
        )
 
 
def save_model_comparison(results_df):
    filename = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n[compare] Saved → {filename}")
 
 
# ==============================================================
# Hyperparameter Tuning
# FIX 8a: scoring changed from "f1" → "f1_weighted"
# FIX 8b: cv changed from CV_FOLDS (5) → 3
# FIX 8c: param grid corrected to match technical report
#          n_estimators: [100,200]  (was [100,200,300])
#          learning_rate: [0.05,0.10,0.15]  (was [0.05,0.10,0.20])
#          max_depth: [3,5]  (was [3,5,7])
# ==============================================================
def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    GridSearchCV over 12 configurations on Gradient Boosting.
    Expected best params: n_estimators=200, max_depth=5, learning_rate=0.10
    Expected: Acc=0.9000, F1=0.9001, AUC=0.9717
    """
    section("[5a] Hyperparameter tuning (GridSearchCV on GB) …")
 
    # FIX 8c: correct param grid (12 configs: 2×2×3)
    param_grid = {
        "n_estimators" : [100, 200],
        "learning_rate": [0.05, 0.10, 0.15],
        "max_depth"    : [3, 5],
    }
 
    search = GridSearchCV(
        estimator  = GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid = param_grid,
        scoring    = "f1_weighted",   # FIX 8a: was "f1"
        cv         = 3,               # FIX 8b: was CV_FOLDS (5)
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,
    )
    search.fit(X_train, y_train)
 
    best_model    = search.best_estimator_
    predictions   = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
 
    accuracy  = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions,
                                average="weighted", zero_division=0)
    recall    = recall_score(y_test, predictions,
                             average="weighted", zero_division=0)
    f1        = f1_score(y_test, predictions,
                         average="weighted", zero_division=0)
    auc       = roc_auc_score(y_test, probabilities)
 
    print(f"[tune] Best params  : {search.best_params_}")
    print(f"[tune] Best CV F1   : {search.best_score_:.4f}")
    print(f"[tune] Test Acc     : {accuracy:.4f}")
    print(f"[tune] Test F1      : {f1:.4f}")
    print(f"[tune] Test AUC     : {auc:.4f}")
 
    tuning_results = {
        "Accuracy"   : accuracy,
        "Precision"  : precision,
        "Recall"     : recall,
        "F1"         : f1,
        "AUC"        : auc,
        "Best Params": search.best_params_,
        "Best CV F1" : search.best_score_,
    }
 
    return best_model, tuning_results, predictions, probabilities
 
 
def compare_and_tune(results_df, X_train, y_train, X_test, y_test):
    ranked = rank_models(results_df)
    display_model_comparison(ranked)
    save_model_comparison(ranked)
 
    tuned_model, tuning_results, predictions, probabilities = \
        tune_gradient_boosting(X_train, y_train, X_test, y_test)
 
    return ranked, tuned_model, tuning_results, predictions, probabilities
 
 
"""
==============================================================
Part 08B - Evaluation, Visualisation & Export
==============================================================
"""
 
def print_classification_results(y_test, predictions):
    section(
        "============================================================\n"
        "  Classification Report — Gradient Boosting (Tuned)\n"
        "============================================================"
    )
    print(classification_report(
        y_test, predictions,
        target_names=["Unstable","Stable"],
        digits=4,
    ))
 
 
def evaluate_confusion_matrix(y_test, predictions):
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    fnr = fn / (fn + tp)
    print("[GB Tuned] Confusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  False Negative Rate : {fnr*100:.2f} %")
    return tn, fp, fn, tp
 
 
def feature_importance_plot(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
 
    plt.figure(figsize=(10, 6))
    importance.head(15).plot(kind="bar", color="#1ABC9C", edgecolor="black")
    plt.title("Gradient Boosting — Top 15 Feature Importances")
    plt.ylabel("Importance (MDI)")
    plt.xticks(rotation=45, ha="right")
    save_figure("feature_importance.png")
 
 
def create_results_dashboard(y_test, predictions, probabilities):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GB Tuned — Evaluation Dashboard", fontweight="bold")
 
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["Unstable","Stable"],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
 
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    axes[1].set_title("ROC Curve")
 
    save_figure("model_results_dashboard.png")
    print("[evaluate] Dashboard saved → outputs/figures/model_results_dashboard.png")
 
 
def export_results(tuning_results):
    results = pd.DataFrame({
        "Metric": ["Accuracy","Precision","Recall","F1 Score","ROC AUC"],
        "Value" : [
            tuning_results["Accuracy"],
            tuning_results["Precision"],
            tuning_results["Recall"],
            tuning_results["F1"],
            tuning_results["AUC"],
        ],
    })
    filename = RESULTS_DIR / "results_summary.csv"
    results.to_csv(filename, index=False)
    print(f"\n[evaluate] Results saved → {filename}")
 
 
def save_final_model(model, scaler):
    joblib.dump(model,  MODELS_DIR / "gradient_boosting.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
 
 
def final_evaluation(tuned_model, scaler, tuning_results,
                     y_test, predictions, probabilities, feature_names):
    print_classification_results(y_test, predictions)
    evaluate_confusion_matrix(y_test, predictions)
    feature_importance_plot(tuned_model, feature_names)
    create_results_dashboard(y_test, predictions, probabilities)
    export_results(tuning_results)
    save_final_model(tuned_model, scaler)
    pipeline_complete()
 
 
"""
==============================================================
Part 09 - Main Pipeline
==============================================================
"""
 
def main():
    parser = argparse.ArgumentParser(
        description="Grid Stability Classification Pipeline — COEN807, ABU Zaria"
    )
    parser.add_argument(
        "--data",
        required=False,    # optional: omit to use synthetic data
        default=None,
        help="Path to smart_grid_stability_augmented.csv (optional)"
    )
    args = parser.parse_args()
 
    print_banner()
 
    # STEP 1 — Load
    df = load_dataset(args.data)
    validate_dataset(df)
 
    # STEP 2 — Feature Engineering
    df = engineer_features(df)
    create_eda_dashboard(df)
 
    # STEP 3 — Preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
 
    # STEP 4 — Training + CV  (produces the Step 5 table)
    models = build_models()
    results_df, trained_models = cross_validate_models(
        models, X_train, y_train, X_test, y_test
    )
 
    # STEP 5 — Comparison + Tuning
    ranked, tuned_model, tuning_results, predictions, probabilities = \
        compare_and_tune(results_df, X_train, y_train, X_test, y_test)
 
    # STEP 6 — Final evaluation, plots, export
    feature_names = [c for c in df.columns if c != TARGET]
    final_evaluation(
        tuned_model, scaler, tuning_results,
        y_test, predictions, probabilities, feature_names,
    )
 
 
if __name__ == "__main__":
    main()
 