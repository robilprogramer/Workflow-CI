"""
modelling.py — MLflow Project Entry Point
==========================================
Versi modelling.py yang disesuaikan untuk MLflow Project.
Menerima hyperparameter via argparse agar bisa dikontrol dari MLProject.

Penggunaan manual:
    python modelling.py --n_estimators 200 --max_depth 10

Penggunaan via MLflow Project:
    mlflow run . -P n_estimators=200 -P max_depth=10
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Gold-Price-CI"
TARGET_COL      = "USD (PM)"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "gold_preprocessing", "gold_train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "gold_preprocessing", "gold_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def parse_args():
    """Parse hyperparameter dari command line / MLProject."""
    parser = argparse.ArgumentParser(description="Gold Price Prediction - MLflow Project")
    parser.add_argument("--n_estimators",      type=int, default=200)
    parser.add_argument("--max_depth",         type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf",  type=int, default=1)
    parser.add_argument("--random_state",      type=int, default=42)
    return parser.parse_args()


def load_dataset():
    """Memuat dataset train dan test."""
    print("📂 Memuat dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
    y_train = train_df[TARGET_COL]
    X_test  = test_df[X_train.columns]
    y_test  = test_df[TARGET_COL]

    print(f"  Train : {X_train.shape}")
    print(f"  Test  : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def create_feature_importance_plot(model, feature_names, save_path):
    """Plot feature importance dan simpan sebagai PNG."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_names[::-1], importances[indices[::-1]],
            color='steelblue', edgecolor='white')
    ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Feature importance plot: {save_path}")


def create_residual_plot(y_test, y_pred, save_path):
    """Plot residual analysis dan simpan sebagai PNG."""
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    axes[0].set_title('Actual vs Predicted', fontweight='bold')
    axes[0].set_xlabel('Actual USD (PM)')
    axes[0].set_ylabel('Predicted USD (PM)')
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=60, color='salmon', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Distribusi Residual', fontweight='bold')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frekuensi')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Residual Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Residual plot: {save_path}")


def train(args):
    """Pipeline training lengkap dengan manual MLflow logging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_dataset()

    # ── Cek apakah sudah ada active run dari MLflow Project
    # Jika ya → pakai run yang ada, jangan buat baru
    active_run = mlflow.active_run()
    if active_run:
        print(f"  Menggunakan active run dari MLflow Project: {active_run.info.run_id}")
        run_context = mlflow.start_run(run_id=active_run.info.run_id, nested=True)
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_context = mlflow.start_run(run_name="RandomForest-CI")

    with run_context:

        print("\n🚀 Training model...")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ── Metrics
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f"\n📊 Hasil Evaluasi:")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R²   : {r2:.4f}")
        print(f"  MAPE : {mape:.4f}")

        # ── Manual Logging: Parameters
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
        mlflow.log_param("random_state",      args.random_state)
        mlflow.log_param("train_size",        len(X_train))
        mlflow.log_param("test_size",         len(X_test))
        mlflow.log_param("n_features",        X_train.shape[1])

        # ── Manual Logging: Metrics
        mlflow.log_metric("mae",      mae)
        mlflow.log_metric("mse",      mse)
        mlflow.log_metric("rmse",     rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape",     mape)

        # ── Manual Logging: Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(3)
        )

        # ── Log Artefak Tambahan
        print("\n🎨 Membuat artefak tambahan...")
        feat_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        res_path  = os.path.join(OUTPUT_DIR, "residual_plot.png")
        rep_path  = os.path.join(OUTPUT_DIR, "evaluation_report.json")

        create_feature_importance_plot(model, list(X_train.columns), feat_path)
        create_residual_plot(y_test, y_pred, res_path)

        report = {
            "model": "RandomForestRegressor",
            "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape},
            "params": vars(args)
        }
        with open(rep_path, 'w') as f:
            json.dump(report, f, indent=4)

        mlflow.log_artifact(feat_path, artifact_path="plots")
        mlflow.log_artifact(res_path,  artifact_path="plots")
        mlflow.log_artifact(rep_path,  artifact_path="reports")

        # ── Tags
        mlflow.set_tag("dataset",   "Gold Price LBMA 2001-2023")
        mlflow.set_tag("target",    "USD (PM)")
        mlflow.set_tag("developer", "Robil")
        mlflow.set_tag("stage",     "CI")

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Run ID     : {run_id}")
        print(f"✅ Experiment : {EXPERIMENT_NAME}")

    return model, run_id


if __name__ == "__main__":
    args = parse_args()
    model, run_id = train(args)
    print("\n✅ Training selesai!")
