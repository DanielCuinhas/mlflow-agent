"""
Tabular classification baseline with agent-friendly MLflow logging.

Run:
    python train_tabular.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Check http://localhost:5000 for the MLflow UI.
"""
import hashlib
import subprocess

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5
DATA_VERSION = "breast_cancer_v1"  # bump whenever your data or features change
PRIMARY_METRIC = "cv_f1_mean"

# MLflow saves sklearn models with skops, which only allows known-safe types.
# These are models we trained ourselves, so trusting these specific types is fine.
# If a new "untrusted types" error names another type, review it and add it here.
TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tabular-classification-baseline")


def load_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    # Lock away a final test set. It is never evaluated or logged here, so neither
    # you nor the agent can overfit to it. Touch it once, at the very end.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    return X_dev, y_dev


def data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def feature_importances(model, names):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(est.coef_).ravel()
    else:
        return None
    pairs = sorted(zip(names, map(float, vals)), key=lambda p: -p[1])
    return dict(pairs[:20])  # top 20 keeps it compact for the agent


CONFIGS = [
    ("logreg", "linear", LogisticRegression(C=1.0, max_iter=2000), {"C": 1.0}),
    ("logreg", "linear", LogisticRegression(C=0.1, max_iter=2000), {"C": 0.1}),
    ("rf", "tree_ensemble",
     RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
     {"n_estimators": 200, "max_depth": 6}),
    ("rf", "tree_ensemble",
     RandomForestClassifier(n_estimators=400, max_depth=None, random_state=SEED),
     {"n_estimators": 400, "max_depth": "None"}),
    ("hgb", "gbm",
     HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, random_state=SEED),
     {"learning_rate": 0.05, "max_depth": 3}),
]


def main():
    X, y = load_data()
    dhash = data_hash(X, y)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}

    for name, family, estimator, hparams in CONFIGS:
        # Scale only for linear models; trees don't need it
        model = make_pipeline(StandardScaler(), estimator) if family == "linear" else estimator
        run_name = f"{name}-" + "-".join(f"{k}{v}" for k, v in hparams.items())

        with mlflow.start_run(run_name=run_name):
            # --- tags: how the agent groups and compares runs ---
            mlflow.set_tags({
                "model_family": family,
                "model_name": name,
                "data_version": DATA_VERSION,
                "data_hash": dhash,
                "git_commit": git_commit(),
                "task": "binary_classification",
                "primary_metric": PRIMARY_METRIC,
            })

            # --- params: everything needed to reproduce ---
            mlflow.log_params({
                **hparams, "seed": SEED, "cv_folds": N_FOLDS,
                "n_rows": len(X), "n_features": X.shape[1],
            })

            # --- metrics: mean AND std so noise is visible ---
            res = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                 return_train_score=True, n_jobs=-1)
            metrics = {}
            for m in scoring:
                val, trn = res[f"test_{m}"], res[f"train_{m}"]
                metrics[f"cv_{m}_mean"] = float(val.mean())
                metrics[f"cv_{m}_std"] = float(val.std())
                metrics[f"train_{m}_mean"] = float(trn.mean())
                metrics[f"gap_{m}"] = float(trn.mean() - val.mean())  # overfitting signal
            metrics["fit_time_mean_s"] = float(res["fit_time"].mean())
            mlflow.log_metrics(metrics)

            # --- artifacts: structured JSON, not just plots ---
            oof = cross_val_predict(model, X, y, cv=cv)
            mlflow.log_dict(
                classification_report(y, oof, output_dict=True), "diagnostics/classification_report.json"
            )
            mlflow.log_dict(
                {"labels": [0, 1], "matrix": confusion_matrix(y, oof).tolist()},
                "diagnostics/confusion_matrix.json",
            )
            model.fit(X, y)
            fi = feature_importances(model, X.columns)
            if fi:
                mlflow.log_dict(fi, "diagnostics/feature_importances.json")
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES)

            print(f"{run_name:45s} {PRIMARY_METRIC}={metrics[PRIMARY_METRIC]:.4f} "
                  f"(±{metrics['cv_f1_std']:.4f})  gap={metrics['gap_f1']:.4f}")


if __name__ == "__main__":
    main()