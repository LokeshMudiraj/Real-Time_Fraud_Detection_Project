import os
import logging
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import MODEL_PATH, LOG_PATH, FRAUD_THRESHOLD
from src.preprocessing import engineer_features

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(
    title="Enterprise Real-Time Fraud Detection API",
    description="Low-latency scoring service for mobile & banking transactions.",
    version="1.0.0"
)

model = None
feature_columns = []

@app.on_event("startup")
def load_model():
    global model, feature_columns
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
            model = artifact["model"]
            feature_columns = artifact["features"]
        logging.info("Model loaded successfully into memory.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN99001")
    user_id: int = Field(..., example=1042)
    amount: float = Field(..., gt=0, example=18500.00)
    merchant_id: int = Field(..., example=2015)
    location: str = Field(..., example="Mumbai")
    device_type: str = Field(..., example="Android")
    transaction_type: str = Field(..., example="transfer")
    location_deviation: int = Field(0, ge=0, le=1, example=1)
    mins_since_last_txn: float = Field(60.0, ge=0, example=2.5)
    hour: int = Field(..., ge=0, le=23, example=2)
    dayofweek: int = Field(..., ge=0, le=6, example=6)

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(txn: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Run `python -m src.train` first.")
        
    df_raw = pd.DataFrame([txn.model_dump()])
    X_processed = engineer_features(df_raw, feature_columns=feature_columns)
    
    prob = float(model.predict_proba(X_processed)[0][1])
    is_fraud = prob >= FRAUD_THRESHOLD
    
    risk_level = "HIGH" if prob >= 0.75 else ("MEDIUM" if prob >= 0.40 else "LOW")
    
    logging.info(f"TXN={txn.transaction_id} | USER={txn.user_id} | AMT={txn.amount} | PROB={prob:.4f} | FRAUD={is_fraud}")
    
    return PredictionResponse(
        transaction_id=txn.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        risk_level=risk_level
    )
