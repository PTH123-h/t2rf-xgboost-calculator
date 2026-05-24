import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("xgb_model.pkl")
FEATURES = ["HBDH", "AST", "BUN", "GLB", "MCHC", "HCT", "NLR", "MLR", "PNI"]
THRESHOLD = 0.167850

DEFAULTS = {
    "HBDH": 210.0,
    "AST": 24.0,
    "BUN": 6.2,
    "GLB": 30.0,
    "MCHC": 330.0,
    "HCT": 0.42,
    "NLR": 4.0,
    "MLR": 0.35,
    "PNI": 45.0,
}


st.set_page_config(
    page_title="T2RF Risk Calculator",
    page_icon="T2RF",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 980px;
        padding-top: 1.1rem;
        padding-bottom: 1.2rem;
    }
    html, body, [class*="css"] {
        font-size: 20px;
    }
    h1 {
        font-size: 2.0rem !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
        margin-bottom: 0.2rem !important;
    }
    h2, h3 {
        font-weight: 800 !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.5rem !important;
    }
    label, .stNumberInput label {
        font-size: 1.0rem !important;
        font-weight: 700 !important;
    }
    .stNumberInput input {
        min-height: 2.35rem !important;
        font-size: 1.0rem !important;
        font-weight: 650 !important;
    }
    .stButton > button {
        width: 100%;
        min-height: 3.0rem;
        font-size: 1.18rem;
        font-weight: 800;
        border-radius: 8px;
    }
    .risk-box {
        border: 2px solid #d0d7de;
        border-radius: 8px;
        padding: 1.0rem 1.1rem;
        margin-top: 0.9rem;
        background: #f6f8fa;
    }
    .risk-value {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1.05;
        color: #0f5132;
    }
    .risk-label {
        font-size: 1.12rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }
    .note {
        color: #57606a;
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .research-note {
        margin-top: 0.8rem;
        padding-top: 0.65rem;
        border-top: 1px solid #d8dee4;
        color: #57606a;
        font-size: 0.82rem;
        line-height: 1.42;
    }
    .main-note {
        margin-bottom: 0.55rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }
    hr {
        margin: 0.65rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


model = load_model()

st.title("Risk Prediction of Type II Respiratory Failure")
st.markdown(
    "<div class='note main-note'>XGBoost model based on admission laboratory indicators in acute COPD/AECOPD inpatients.</div>",
    unsafe_allow_html=True,
)

left_panel, right_panel = st.columns([1.25, 1.0], gap="large")

values = {}
with left_panel:
    st.subheader("Input Variables")
    input_cols = st.columns(2, gap="medium")
    for index, feature in enumerate(FEATURES):
        with input_cols[index % 2]:
            values[feature] = st.number_input(
                feature,
                value=float(DEFAULTS[feature]),
                step=0.01,
                format="%.3f",
            )

input_df = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

with right_panel:
    st.subheader("Prediction")
    calculated = st.button("Calculate Risk")

    if calculated:
        probability = float(model.predict_proba(input_df)[0, 1])
        risk_display = f"{probability * 100:.1f}%"
        prediction = "High risk" if probability >= THRESHOLD else "Low risk"
        label_display = f"Predicted probability of T2RF: {prediction}"
    else:
        risk_display = "--.-%"
        label_display = "Enter variables and calculate risk"

    st.markdown(
        f"""
        <div class="risk-box">
            <div class="risk-value">{risk_display}</div>
            <div class="risk-label">{label_display}</div>
            <div class="note">Classification threshold: {THRESHOLD:.6f}</div>
            <div class="research-note">This web-based calculator is intended for research use.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

 
