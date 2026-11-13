import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_mobile_payments.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "predictions.log")

CATEGORICAL_COLS = ["location", "device_type", "transaction_type"]
NUMERIC_COLS = ["amount", "user_id", "merchant_id", "location_deviation", "mins_since_last_txn", "hour", "dayofweek", "is_night"]
TARGET_COL = "is_fraud"

FRAUD_THRESHOLD = 0.50
RANDOM_STATE = 42
