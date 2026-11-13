import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from src.config import DATA_PATH, MODEL_PATH, TARGET_COL, FRAUD_THRESHOLD, BASE_DIR
from src.preprocessing import engineer_features

METRICS_OUT_PATH = os.path.join(BASE_DIR, "models", "metrics_report.json")


def evaluate_model(threshold: float = FRAUD_THRESHOLD):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python -m src.data_generator` first."
        )

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_columns = artifact["features"]

    df = pd.read_csv(DATA_PATH)
    X = engineer_features(df, feature_columns=feature_columns)
    y_true = df[TARGET_COL].values

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": threshold,
        "total_samples": int(len(y_true)),
        "actual_frauds": int(y_true.sum()),
        "predicted_frauds": int(y_pred.sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
    }

    print("=== Fraud Detection Model Metrics ===")
    print(f"Threshold          : {metrics['threshold']}")
    print(f"Total Transactions : {metrics['total_samples']}")
    print(f"Actual Frauds      : {metrics['actual_frauds']}")
    print(f"Predicted Frauds   : {metrics['predicted_frauds']}")
    print("-" * 38)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"F1-Score           : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC             : {metrics['pr_auc']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print("-" * 38)
    print("Confusion Matrix:")
    print(f"  TN: {tn}  |  FP: {fp}")
    print(f"  FN: {fn}  |  TP: {tp}")
    print("-" * 38)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Fraud (1)"]))

    # Threshold sensitivity check
    print("=== Threshold Sensitivity Analysis ===")
    for t in [0.30, 0.40, 0.50, 0.60, 0.75]:
        preds_t = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds_t, zero_division=0)
        r = recall_score(y_true, preds_t, zero_division=0)
        f1 = f1_score(y_true, preds_t, zero_division=0)
        print(f"Threshold {t:.2f} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

    os.makedirs(os.path.dirname(METRICS_OUT_PATH), exist_ok=True)
    with open(METRICS_OUT_PATH, "w") as out_f:
        json.dump(metrics, out_f, indent=4)
    print(f"\nSaved JSON metrics report -> {METRICS_OUT_PATH}")

    return metrics


if __name__ == "__main__":
    evaluate_model()