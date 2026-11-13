import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.config import DATA_PATH, MODEL_PATH, TARGET_COL, RANDOM_STATE
from src.preprocessing import engineer_features

def train_model():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    X = engineer_features(df)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print("Training Random Forest Classifier (balanced class weights)...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)
    print(f"\nModel artifact saved successfully -> {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
