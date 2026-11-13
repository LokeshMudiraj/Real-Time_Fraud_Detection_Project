import pandas as pd
import numpy as np
import random
from src.config import DATA_PATH, RANDOM_STATE

def generate_data(n=10000, fraud_ratio=0.025):
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    
    user_ids = range(1000, 1200)
    merchant_ids = range(2000, 2100)
    locations = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad']
    device_types = ['Android', 'iOS', 'Web']
    transaction_types = ['transfer', 'top-up', 'bill', 'purchase']
    
    start_ts = pd.Timestamp.now() - pd.Timedelta(days=60)
    random_seconds = np.random.randint(0, 60 * 24 * 3600, size=n)
    timestamps = sorted([start_ts + pd.Timedelta(seconds=int(s)) for s in random_seconds])

    data = []
    user_last_loc, user_last_ts = {}, {}

    for i in range(n):
        uid = random.choice(user_ids)
        ts = timestamps[i]
        loc = random.choice(locations)
        is_fraud = 1 if random.random() < fraud_ratio else 0
        
        amount = round(random.uniform(10, 10000), 2)
        if is_fraud:
            amount *= random.uniform(1.8, 5.0)
            
        loc_changed = 1 if (uid in user_last_loc and user_last_loc[uid] != loc) else 0
        time_diff_mins = (ts - user_last_ts[uid]).total_seconds() / 60.0 if uid in user_last_ts else 9999.0
        
        if is_fraud and random.random() < 0.7:
            loc_changed = 1
            time_diff_mins = round(random.uniform(0.5, 15.0), 2)

        user_last_loc[uid] = loc
        user_last_ts[uid] = ts

        data.append({
            'transaction_id': f"TXN{i:06d}",
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': uid,
            'amount': round(amount, 2),
            'merchant_id': random.choice(merchant_ids),
            'location': loc,
            'device_type': random.choice(device_types),
            'transaction_type': random.choice(transaction_types),
            'location_deviation': loc_changed,
            'mins_since_last_txn': round(time_diff_mins, 2),
            'is_fraud': is_fraud
        })
        
    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {n} synthetic transactions -> {DATA_PATH}")

if __name__ == "__main__":
    generate_data()
