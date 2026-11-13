import os
import sys
import pickle
import logging

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import DATA_PATH, MODEL_PATH, FRAUD_THRESHOLD
from src.preprocessing import engineer_features

st.set_page_config(
    page_title="Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

INDIGO = "#4F46E5"     # primary: general measures
ROSE = "#E11D48"       # risk: fraud measures
EMERALD = "#059669"    # safe: legitimate measures
AMBER = "#D97706"      # medium risk
SLATE = "#64748B"      # secondary text
GRID = "rgba(148, 163, 184, 0.20)"

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 14px;
        padding: 26px 30px;
        margin-bottom: 1.25rem;
        color: #F8FAFC;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Section headings */
    .section-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin: 1.75rem 0 0.75rem 0;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-left: 4px solid #4F46E5;
        padding: 16px 20px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 14px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] .sidebar-brand {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 0;
    }
    section[data-testid="stSidebar"] .sidebar-caption {
        font-size: 12px;
        color: #64748B;
        margin-bottom: 1rem;
    }

    .footer {
        color: #64748B;
        font-size: 12px;
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. DATA AND MODEL LOADING
# ============================================================

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    required = ["is_fraud", "amount", "transaction_type", "location"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["is_fraud", "amount"])
    df["is_fraud"] = df["is_fraud"].astype(int)

    return df


@st.cache_resource
def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ============================================================
# 4. DATA PREPARATION HELPERS
# ============================================================

def group_rate(frame, col):
    """Total count, fraud count and fraud rate (%) per category."""
    grouped = (
        frame.assign(_key=frame[col].astype(str))
        .groupby("_key")
        .agg(total=("is_fraud", "size"), fraud=("is_fraud", "sum"))
        .reset_index()
        .rename(columns={"_key": col})
    )
    grouped["rate"] = grouped["fraud"] / grouped["total"] * 100
    return grouped


def amount_band_frame(frame):
    """Fraud rate by transaction size band (INR)."""
    bins = [0, 5_000, 25_000, 100_000, float("inf")]
    labels = ["Under ₹5K", "₹5K–25K", "₹25K–1L", "₹1L+"]
    banded = frame.assign(
        band=pd.cut(frame["amount"], bins=bins, labels=labels, right=False).astype(str)
    )
    return group_rate(banded, "band"), labels


def hourly_frame(frame):
    """Fraud rate for each hour of the day (0–23)."""
    data = frame.assign(hour=pd.to_numeric(frame["hour"], errors="coerce")).dropna(subset=["hour"])
    data = data.assign(hour=data["hour"].astype(int))
    hourly = (
        data.groupby("hour")
        .agg(total=("is_fraud", "size"), fraud=("is_fraud", "sum"))
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    hourly["rate"] = (hourly["fraud"] / hourly["total"]).where(hourly["total"] > 0, 0) * 100
    return hourly


def daily_frame(frame, date_col):
    """Fraud rate for each calendar day."""
    parsed = frame.assign(_date=pd.to_datetime(frame[date_col], errors="coerce")).dropna(subset=["_date"])
    if parsed.empty:
        return parsed
    daily = (
        parsed.groupby(parsed["_date"].dt.normalize())
        .agg(total=("is_fraud", "size"), fraud=("is_fraud", "sum"))
        .reset_index()
        .rename(columns={"_date": "date"})
    )
    daily["rate"] = daily["fraud"] / daily["total"] * 100
    return daily

# ============================================================
# 5. CHART BUILDERS
# ============================================================

def style_chart(chart, title, height):
    """One consistent look for every chart."""
    return (
        chart.properties(
            height=height,
            title=alt.TitleParams(
                text=title,
                anchor="start",
                fontSize=14,
                fontWeight=600,
                offset=14,
            ),
        )
        .configure_view(stroke=None)
        .configure_axis(
            grid=True,
            gridColor=GRID,
            domain=False,
            tickSize=0,
            labelColor=SLATE,
            titleColor=SLATE,
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight=500,
            labelPadding=6,
        )
    )


def ranked_bar(data, category, value, title, value_title, value_format, color, height=300):
    """Horizontal bars sorted by value, with the value printed at the end of each bar."""
    base = alt.Chart(data)

    bars = base.mark_bar(color=color, cornerRadiusEnd=4, size=18).encode(
        x=alt.X(f"{value}:Q", title=value_title, axis=alt.Axis(format=value_format)),
        y=alt.Y(f"{category}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Category"),
            alt.Tooltip(f"{value}:Q", title=value_title, format=value_format),
        ],
    )

    labels = base.mark_text(
        align="left", baseline="middle", dx=6, color=SLATE, fontSize=11
    ).encode(
        x=alt.X(f"{value}:Q"),
        y=alt.Y(f"{category}:N", sort="-x"),
        text=alt.Text(f"{value}:Q", format=value_format),
    )

    return style_chart(alt.layer(bars, labels), title, height)


def column_chart(data, category, value, order, title, value_title, value_format, color, height=300):
    """Vertical bars in a fixed category order, with values above each bar."""
    base = alt.Chart(data)

    bars = base.mark_bar(color=color, cornerRadiusEnd=4, size=44).encode(
        x=alt.X(f"{category}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{value}:Q", title=value_title, axis=alt.Axis(format=value_format)),
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Category"),
            alt.Tooltip(f"{value}:Q", title=value_title, format=value_format),
        ],
    )

    labels = base.mark_text(
        align="center", baseline="bottom", dy=-6, color=SLATE, fontSize=11
    ).encode(
        x=alt.X(f"{category}:N", sort=order),
        y=alt.Y(f"{value}:Q"),
        text=alt.Text(f"{value}:Q", format=value_format),
    )

    return style_chart(alt.layer(bars, labels), title, height)


def line_chart(data, x, x_type, y, title, x_title, y_title, value_format, color, height=300, x_axis=None):
    """Line with a soft area fill and point markers."""
    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:{x_type}", title=x_title, axis=x_axis or alt.Axis()),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=value_format)),
        tooltip=[
            alt.Tooltip(f"{x}:{x_type}", title=x_title),
            alt.Tooltip(f"{y}:Q", title=y_title, format=value_format),
        ],
    )

    area = base.mark_area(color=color, opacity=0.12, line=False)
    line = base.mark_line(color=color, strokeWidth=2.5, interpolate="monotone")
    points = base.mark_circle(color=color, size=40)

    return style_chart(alt.layer(area, line, points), title, height)

# ============================================================
# 6. API AND PREDICTION
# ============================================================

def check_api(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def predict_using_api(payload, api_url):
    response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
    response.raise_for_status()
    return response.json()


def predict_using_local_model(payload):
    artifact = load_model_artifact()
    if artifact is None:
        raise FileNotFoundError(
            "Model artifact not found. Run python -m src.train first."
        )

    df_raw = pd.DataFrame([payload])
    X = engineer_features(df_raw, feature_columns=artifact["features"])
    probability = float(artifact["model"].predict_proba(X)[0][1])

    return {
        "transaction_id": payload["transaction_id"],
        "is_fraud": probability >= FRAUD_THRESHOLD,
        "fraud_probability": round(probability, 4),
        "risk_level": (
            "HIGH" if probability >= 0.75
            else "MEDIUM" if probability >= 0.40
            else "LOW"
        ),
    }

# ============================================================
# 7. SIDEBAR (system status only)
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛡️ Fraud Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Enterprise risk monitoring</div>', unsafe_allow_html=True)

    api_url = st.text_input(
        "FastAPI base URL", value="http://127.0.0.1:8000"
    ).rstrip("/")

    health = check_api(api_url)
    if health:
        st.success("API connected")
        if health.get("model_loaded") is False:
            st.warning("API running, model not loaded")
    else:
        st.error("API offline")

    st.divider()

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Load the full dataset once. Filters on the Transactions tab narrow it further.
try:
    df_all = load_dataset()
    load_error = None
except Exception as e:
    df_all = pd.DataFrame()
    load_error = str(e)

# ============================================================
# 8. HEADER AND TABS
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Fraud Detection Dashboard</div>
        <div class="hero-subtitle">Transaction risk analytics and real-time scoring</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"Dataset error: {load_error}")

tab_overview, tab_scorer, tab_explorer = st.tabs([
    "Overview",
    "Score a transaction",
    "Transactions",
])

# ============================================================
# 9. OVERVIEW (full dataset, no filters)
# ============================================================

with tab_overview:
    if df_all.empty:
        st.info("Dataset not available. Check the DATA_PATH in src/config.py.")
    else:
        total = len(df_all)
        fraud_df = df_all[df_all["is_fraud"] == 1]
        fraud_count = len(fraud_df)
        fraud_rate = fraud_count / total * 100 if total else 0
        exposure = fraud_df["amount"].sum()
        avg_fraud = fraud_df["amount"].mean() if fraud_count else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{total:,}")
        c2.metric("Fraud cases", f"{fraud_count:,}")
        c3.metric("Fraud rate", f"{fraud_rate:.2f}%")
        c4.metric("Fraud exposure", f"₹{exposure:,.0f}")

        has_device = "device_type" in df_all.columns
        has_hour = "hour" in df_all.columns
        date_col = next(
            (c for c in ["timestamp", "transaction_timestamp", "transaction_date", "date"]
             if c in df_all.columns),
            None,
        )

        # Row 1: risk profile
        st.markdown('<div class="section-title">Risk profile</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")

        with left:
            by_type = group_rate(df_all, "transaction_type")
            st.altair_chart(
                ranked_bar(
                    by_type,
                    category="transaction_type",
                    value="rate",
                    title="Fraud rate by transaction type",
                    value_title="Fraud rate (%)",
                    value_format=".2f",
                    color=ROSE,
                    height=max(260, 46 * len(by_type)),
                ),
                use_container_width=True,
            )

        with right:
            band_data, band_order = amount_band_frame(df_all)
            st.altair_chart(
                column_chart(
                    band_data,
                    category="band",
                    value="rate",
                    order=band_order,
                    title="Fraud rate by transaction amount",
                    value_title="Fraud rate (%)",
                    value_format=".2f",
                    color=ROSE,
                ),
                use_container_width=True,
            )

        # Row 2: exposure and channel
        st.markdown('<div class="section-title">Exposure and channel</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")

        with left:
            if fraud_df.empty:
                st.caption("No fraud cases in the dataset.")
            else:
                by_location = (
                    fraud_df.assign(location=fraud_df["location"].astype(str))
                    .groupby("location")["amount"]
                    .sum()
                    .nlargest(10)
                    .reset_index(name="exposure")
                )
                st.altair_chart(
                    ranked_bar(
                        by_location,
                        category="location",
                        value="exposure",
                        title="Top 10 locations by fraud exposure",
                        value_title="Fraud exposure (₹)",
                        value_format=",.0f",
                        color=INDIGO,
                        height=max(260, 38 * len(by_location)),
                    ),
                    use_container_width=True,
                )

        with right:
            if has_device:
                by_device = group_rate(df_all, "device_type")
                st.altair_chart(
                    column_chart(
                        by_device,
                        category="device_type",
                        value="rate",
                        order=sorted(by_device["device_type"].tolist()),
                        title="Fraud rate by device type",
                        value_title="Fraud rate (%)",
                        value_format=".2f",
                        color=ROSE,
                    ),
                    use_container_width=True,
                )
            else:
                st.caption("Device data is not available in this dataset.")

        # Row 3: time patterns
        if has_hour:
            st.markdown('<div class="section-title">Time patterns</div>', unsafe_allow_html=True)
            hourly = hourly_frame(df_all)
            st.altair_chart(
                line_chart(
                    hourly,
                    x="hour",
                    x_type="Q",
                    y="rate",
                    title="Fraud rate by hour of day",
                    x_title="Hour of day",
                    y_title="Fraud rate (%)",
                    value_format=".2f",
                    color=ROSE,
                    x_axis=alt.Axis(values=list(range(0, 24, 2)), grid=False),
                ),
                use_container_width=True,
            )

        # Row 4: trend
        if date_col:
            daily = daily_frame(df_all, date_col)
            if not daily.empty:
                st.markdown('<div class="section-title">Trend</div>', unsafe_allow_html=True)
                st.altair_chart(
                    line_chart(
                        daily,
                        x="date",
                        x_type="T",
                        y="rate",
                        title="Daily fraud rate",
                        x_title="Date",
                        y_title="Fraud rate (%)",
                        value_format=".2f",
                        color=INDIGO,
                    ),
                    use_container_width=True,
                )

# ============================================================
# 10. SCORING
# ============================================================

with tab_scorer:
    st.markdown("#### Score a transaction")
    st.caption("Submit transaction details to the fraud model for a probability-based risk assessment.")

    engine = st.radio(
        "Scoring engine",
        ["FastAPI", "Local model"],
        horizontal=True,
    )

    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            txn_id = st.text_input("Transaction ID", "TXN99821")
            user_id = st.number_input("User ID", min_value=1, value=1042)
            merchant_id = st.number_input("Merchant ID", min_value=1, value=2015)
            amount = st.number_input(
                "Amount (₹)", min_value=1.0, value=24500.0, step=100.0
            )

        with c2:
            location = st.selectbox(
                "Location",
                ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"],
            )
            device_type = st.selectbox("Device", ["Android", "iOS", "Web"])
            transaction_type = st.selectbox(
                "Transaction type", ["transfer", "purchase", "top-up", "bill"]
            )

        with c3:
            location_deviation = st.selectbox(
                "Location deviation",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            mins_since_last = st.number_input(
                "Minutes since previous transaction",
                min_value=0.0, value=2.5, step=0.5,
            )
            hour = st.slider("Hour of day", 0, 23, 2)
            dayofweek = st.slider("Day of week (0 = Monday)", 0, 6, 6)

        submitted = st.form_submit_button(
            "Evaluate risk", type="primary", use_container_width=True
        )

    if submitted:
        if not txn_id.strip():
            st.error("Transaction ID cannot be empty.")
        else:
            payload = {
                "transaction_id": txn_id.strip(),
                "user_id": int(user_id),
                "amount": float(amount),
                "merchant_id": int(merchant_id),
                "location": location,
                "device_type": device_type,
                "transaction_type": transaction_type,
                "location_deviation": int(location_deviation),
                "mins_since_last_txn": float(mins_since_last),
                "hour": int(hour),
                "dayofweek": int(dayofweek),
            }

            try:
                with st.spinner("Analyzing transaction..."):
                    if engine == "FastAPI":
                        result = predict_using_api(payload, api_url)
                    else:
                        result = predict_using_local_model(payload)

                probability = min(max(float(result["fraud_probability"]), 0.0), 1.0)
                is_fraud = bool(result["is_fraud"])
                risk_level = result["risk_level"]

                st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Status", "Fraud" if is_fraud else "Legitimate")
                m2.metric("Fraud probability", f"{probability * 100:.2f}%")
                m3.metric("Risk level", risk_level)

                st.progress(
                    probability,
                    text=f"Decision threshold: {FRAUD_THRESHOLD * 100:.0f}%",
                )

                if risk_level == "HIGH":
                    st.error("High risk: this transaction needs immediate investigation.")
                elif risk_level == "MEDIUM":
                    st.warning("Medium risk: additional verification is recommended.")
                else:
                    st.success("Low risk: the model classified this transaction as legitimate.")

                with st.expander("Transaction details"):
                    st.json({
                        "transaction": payload,
                        "prediction": result,
                        "engine": engine,
                    })

            except requests.HTTPError as e:
                st.error(f"API request failed with status {e.response.status_code}")
                st.code(e.response.text)
            except requests.RequestException as e:
                st.error(f"Cannot reach the API: {e}")
            except Exception as e:
                logging.exception("Prediction failed")
                st.error(f"Prediction failed: {e}")

# ============================================================
# 11. TRANSACTION EXPLORER
# Enterprise-grade filtering, monitoring and data export
# ============================================================

with tab_explorer:

    if df_all.empty:
        st.info("No transaction records are currently available.")

    else:
        # ----------------------------------------------------
        # Prepare filter metadata
        # ----------------------------------------------------
        locations = sorted(df_all["location"].dropna().astype(str).unique())
        types = sorted(df_all["transaction_type"].dropna().astype(str).unique())

        amount_series = pd.to_numeric(df_all["amount"], errors="coerce")
        lo = float(amount_series.min())
        hi = float(amount_series.max())

        # ----------------------------------------------------
        # Reset callback
        # ----------------------------------------------------
        def reset_transaction_filters():
            st.session_state["txn_locations"] = locations
            st.session_state["txn_types"] = types
            st.session_state["txn_status"] = "All transactions"
            st.session_state["txn_search"] = ""

            if lo < hi:
                st.session_state["txn_amount_range"] = (lo, hi)

        # ----------------------------------------------------
        # Page heading
        # ----------------------------------------------------
        st.markdown(
            """
            <div style="margin-bottom:18px;">
                <div style="
                    color:#25B6A4;
                    font-size:0.72rem;
                    font-weight:700;
                    letter-spacing:0.14em;
                    text-transform:uppercase;">
                    RISK OPERATIONS / DATA EXPLORER
                </div>
                <div style="
                    font-size:1.55rem;
                    font-weight:700;
                    color:#E8EEF6;
                    margin-top:5px;">
                    Transaction Explorer
                </div>
                <div style="
                    color:#94A3B8;
                    font-size:0.88rem;
                    margin-top:4px;">
                    Search, filter and investigate transaction-level activity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # FILTER PANEL
        # ----------------------------------------------------
        with st.container(border=True):

            filter_heading, reset_col = st.columns([4, 1])

            filter_heading.markdown(
                """
                <div style="font-size:0.85rem;font-weight:700;
                            color:#E8EEF6;padding-top:5px;">
                    <span style="color:#25B6A4;">◈</span>
                    FILTER TRANSACTIONS
                </div>
                """,
                unsafe_allow_html=True,
            )

            reset_col.button(
                "↺ Reset filters",
                key="reset_txn_filters",
                use_container_width=True,
                on_click=reset_transaction_filters,
            )

            st.markdown(
                '<div style="color:#94A3B8;font-size:0.77rem;'
                'margin:5px 0 12px 0;">Apply multiple conditions to refine the transaction dataset.</div>',
                unsafe_allow_html=True,
            )

            f1, f2, f3 = st.columns([1.4, 1.4, 1.2])

            with f1:
                sel_locations = st.multiselect(
                    "Transaction location",
                    options=locations,
                    default=locations,
                    key="txn_locations",
                    placeholder="Select locations",
                )

            with f2:
                sel_types = st.multiselect(
                    "Transaction category",
                    options=types,
                    default=types,
                    key="txn_types",
                    placeholder="Select categories",
                )

            with f3:
                status = st.selectbox(
                    "Transaction outcome",
                    [
                        "All transactions",
                        "Fraud only",
                        "Legitimate only",
                    ],
                    key="txn_status",
                )

            st.markdown(
                '<div style="font-size:0.78rem;font-weight:600;'
                'color:#94A3B8;margin:10px 0 5px 0;">'
                'TRANSACTION VALUE RANGE</div>',
                unsafe_allow_html=True,
            )

            if lo < hi:
                amount_range = st.slider(
                    "Amount range",
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                    format="₹%.0f",
                    key="txn_amount_range",
                    label_visibility="collapsed",
                )
            else:
                amount_range = (lo, hi)
                st.caption(f"All available transactions have an amount of ₹{lo:,.2f}")

        # ----------------------------------------------------
        # APPLY FILTERS
        # ----------------------------------------------------
        view = df_all.copy()

        numeric_amount = pd.to_numeric(
            view["amount"], errors="coerce"
        )

        view = view[
            view["location"].astype(str).isin(sel_locations)
            & view["transaction_type"].astype(str).isin(sel_types)
            & numeric_amount.between(*amount_range)
        ]

        if status == "Fraud only":
            view = view[view["is_fraud"] == 1]

        elif status == "Legitimate only":
            view = view[view["is_fraud"] == 0]

        # ----------------------------------------------------
        # KPI SUMMARY
        # ----------------------------------------------------
        v_total = len(view)

        v_fraud = int((view["is_fraud"] == 1).sum())
        v_legitimate = int((view["is_fraud"] == 0).sum())

        v_rate = (
            v_fraud / v_total * 100
            if v_total else 0
        )

        v_amount = (
            pd.to_numeric(view["amount"], errors="coerce").sum()
            if v_total else 0
        )

        st.markdown(
            '<div style="font-size:0.75rem;font-weight:700;'
            'letter-spacing:0.1em;color:#94A3B8;'
            'margin:20px 0 10px 0;">FILTERED TRANSACTION SUMMARY</div>',
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Matching transactions",
            f"{v_total:,}",
        )

        k2.metric(
            "Fraud cases",
            f"{v_fraud:,}",
        )

        k3.metric(
            "Fraud rate",
            f"{v_rate:.2f}%",
        )

        k4.metric(
            "Total transaction value",
            f"₹{v_amount:,.0f}",
        )

        # ----------------------------------------------------
        # TRANSACTION REGISTER
        # ----------------------------------------------------
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;'
            'color:#E8EEF6;margin:25px 0 5px 0;">'
            'Transaction Register</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Review individual transactions, inspect their attributes "
            "and export the filtered records."
        )

        if view.empty:
            st.info(
                "No transactions match the selected filters. "
                "Try adjusting your filter criteria."
            )

        else:

            # Search controls
            search_col, info_col = st.columns([3, 1])

            search = search_col.text_input(
                "Search transactions",
                placeholder="Search transaction ID, location or user ID...",
                key="txn_search",
                label_visibility="collapsed",
            )

            table = view.copy()

            # Add readable status
            table["status"] = table["is_fraud"].map({
                1: "🔴 Fraud",
                0: "🟢 Legitimate",
            }).fillna("Unknown")

            # Search across relevant fields
            if search.strip():

                search_columns = [
                    col for col in [
                        "transaction_id",
                        "location",
                        "user_id",
                        "merchant_id",
                    ]
                    if col in table.columns
                ]

                mask = pd.Series(False, index=table.index)

                for col in search_columns:
                    mask |= (
                        table[col]
                        .astype(str)
                        .str.contains(
                            search.strip(),
                            case=False,
                            na=False,
                            regex=False,
                        )
                    )

                table = table[mask]

            # Sort newest transactions first
            if "timestamp" in table.columns:
                table = table.sort_values(
                    "timestamp",
                    ascending=False,
                )

            # Display count
            info_col.markdown(
                f"""
                <div style="text-align:right;padding-top:9px;
                            color:#94A3B8;font-size:0.8rem;">
                    <b style="color:#E8EEF6;">{len(table):,}</b>
                    matching records
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # TABLE CONFIGURATION
            # ------------------------------------------------
            display_cols = [
                col for col in [
                    "transaction_id",
                    "timestamp",
                    "user_id",
                    "merchant_id",
                    "amount",
                    "location",
                    "device_type",
                    "transaction_type",
                    "status",
                    "fraud_probability",
                ]
                if col in table.columns
            ]

            column_config = {
                "transaction_id": st.column_config.TextColumn(
                    "Transaction ID",
                    help="Unique transaction reference",
                ),

                "timestamp": st.column_config.DatetimeColumn(
                    "Transaction time",
                    format="DD MMM YYYY, HH:mm",
                ),

                "user_id": st.column_config.NumberColumn(
                    "User ID",
                    format="%d",
                ),

                "merchant_id": st.column_config.NumberColumn(
                    "Merchant ID",
                    format="%d",
                ),

                "amount": st.column_config.NumberColumn(
                    "Amount (INR)",
                    format="₹%.2f",
                ),

                "location": st.column_config.TextColumn(
                    "Location",
                ),

                "device_type": st.column_config.TextColumn(
                    "Device",
                ),

                "transaction_type": st.column_config.TextColumn(
                    "Transaction type",
                ),

                "status": st.column_config.TextColumn(
                    "Fraud status",
                    help="Observed fraud label from the dataset",
                ),

                "fraud_probability": st.column_config.NumberColumn(
                    "Fraud probability",
                    format="percent",
                    help="Model-estimated fraud probability, if available",
                ),
            }

            # ------------------------------------------------
            # DATA TABLE
            # ------------------------------------------------
            st.dataframe(
                table[display_cols].head(1000),
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                height=430,
            )

            st.caption(
                f"Showing {min(len(table), 1000):,} of {len(table):,} "
                "matching records. Export includes all matching records."
            )

            # ------------------------------------------------
            # EXPORT ACTIONS
            # ------------------------------------------------
            export_col, format_col = st.columns([1, 3])

            export_data = table.drop(
                columns=["is_fraud"],
                errors="ignore",
            ).to_csv(index=False).encode("utf-8")

            export_col.download_button(
                label="↓ Export transaction report",
                data=export_data,
                file_name="fraud_transactions_report.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

            format_col.caption(
                "CSV export includes all records matching the filters "
                "and search criteria, not just the first 1,000 displayed rows."
            )


# ============================================================
# 12. APPLICATION FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        border-top:1px solid #34465D;
        margin-top:30px;
        padding:16px 0 8px 0;
        display:flex;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:8px;
        color:#94A3B8;
        font-size:0.76rem;
    ">
        <span>
            <b style="color:#25B6A4;">FRAUD INTELLIGENCE</b>
            &nbsp; | &nbsp; Transaction Risk Monitoring
        </span>
        <span>
            Model-generated assessments require appropriate human review.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
