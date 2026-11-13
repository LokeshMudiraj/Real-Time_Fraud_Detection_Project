import pandas as pd
from src.config import CATEGORICAL_COLS

def engineer_features(df: pd.DataFrame, feature_columns: list = None):
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
    df['is_night'] = df['hour'].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)
    
    drop_cols = [c for c in ['transaction_id', 'timestamp', 'is_fraud'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    X = pd.get_dummies(X, columns=[c for c in CATEGORICAL_COLS if c in X.columns], drop_first=True)
    
    if feature_columns is not None:
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        
    return X
