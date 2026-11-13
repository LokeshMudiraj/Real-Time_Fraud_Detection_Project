# Enterprise Real-Time Fraud Detection System

A modular, business-grade machine learning pipeline, FastAPI inference service, and Streamlit operations dashboard for real-time banking and mobile payment fraud detection.

## Directory Structure

```text
fraud_detection_project/
├── data/               # Raw and synthetic transaction datasets
├── src/                # Core ML pipeline & REST API
│   ├── config.py       # Centralized paths, thresholds, and hyperparameters
│   ├── data_generator.py # Synthetic transaction & behavioral velocity generator
│   ├── preprocessing.py  # Shared feature engineering (training & inference parity)
│   ├── train.py        # Model training, evaluation, and serialization
│   └── api.py          # FastAPI REST service with audit logging & risk scoring
├── dashboard/          # Interactive web UI
│   └── streamlit_app.py # Streamlit risk console & fraud analytics dashboard
├── models/             # Serialized ML model artifacts (.pkl)
├── logs/               # Prediction audit logs
├── tests/              # Automated API & schema unit tests
└── requirements.txt    # Pinned Python dependencies
```

## Setup & Execution

1. **Create a Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python -m src.train
   ```

3. **Start the FastAPI Inference Service (Terminal 1):**
   ```bash
   uvicorn src.api:app --reload
   ```

4. **Launch the Streamlit Risk Console (Terminal 2):**
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```

5. **Run Automated Tests:**
   ```bash
   pytest
   ```
