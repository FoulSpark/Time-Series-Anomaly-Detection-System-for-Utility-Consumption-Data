import io
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.validation.input_validator import validate_input_dataset
from src.models.hybrid import build_hybrid_flags
from src.models.incidents import cluster_incidents


APP_TITLE = "Utility Time-Series Anomaly Detection & Inspection Prioritization"


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _load_input(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    return pd.read_csv(uploaded_file)


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.read()


def _save_artifact(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _section_divider():
    st.markdown("---")


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader(
    "Upload consumption dataset (CSV or Parquet)",
    type=["csv", "parquet"],
)

use_repo_sample = st.sidebar.checkbox("Use repo sample data (data/Raw/consumption_daily.csv)")

artifact_dir = st.sidebar.text_input(
    "Artifacts folder (Parquet outputs)",
    value="data/Processed",
)

st.sidebar.header("Parameters")

min_days_per_meter = st.sidebar.number_input("Min days per meter", min_value=1, value=30, step=1)
max_missing_ratio = st.sidebar.slider("Max missing ratio per meter", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
max_flatline_run = st.sidebar.number_input("Max flatline run (days)", min_value=1, value=14, step=1)

stl_period = st.sidebar.number_input("STL period (days)", min_value=2, value=365, step=1)
stl_meter_quantile = st.sidebar.slider(
    "STL per-meter anomaly quantile (top fraction flagged)",
    min_value=0.90,
    max_value=0.999,
    value=0.98,
    step=0.001,
)

hybrid_stl_global_quantile = st.sidebar.slider(
    "Hybrid STL global quantile (strongest STL days kept)",
    min_value=0.90,
    max_value=0.999,
    value=0.995,
    step=0.001,
)

gap_days = st.sidebar.number_input("Incident clustering gap_days", min_value=0, value=2, step=1)


df = None
input_name = None

if use_repo_sample:
    sample_path = os.path.join("data", "Raw", "consumption_daily.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        input_name = sample_path
    else:
        st.warning(f"Repo sample not found at: {sample_path}")

if df is None and uploaded is not None:
    df = _load_input(uploaded)
    input_name = uploaded.name

if df is None:
    st.info("Upload a dataset from the sidebar or enable the repo sample data.")
    st.stop()


df = _ensure_datetime(df)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Input preview")
    st.dataframe(df.head(50), use_container_width=True)

with col2:
    st.subheader("Dataset summary")
    st.write({
        "rows": int(len(df)),
        "meters": int(df["meter_id"].nunique()) if "meter_id" in df.columns else None,
        "date_min": str(df["date"].min()) if "date" in df.columns else None,
        "date_max": str(df["date"].max()) if "date" in df.columns else None,
        "has_generated_anomaly": bool("generated_anomaly" in df.columns),
    })

_section_divider()


st.header("1) Validate Input")

if st.button("Run validation"):
    report = validate_input_dataset(
        df,
        min_days_per_meter=int(min_days_per_meter),
        max_missing_ratio_per_meter=float(max_missing_ratio),
        max_flatline_run=int(max_flatline_run),
    )

    st.session_state["validation_report"] = report

report = st.session_state.get("validation_report")
if report is not None:
    st.subheader("Validation result")
    st.write({
        "dataset_valid": bool(report.dataset_valid),
        "blocking_issues": report.blocking_issues,
        "warnings": report.warnings,
        "summary": report.summary,
    })

    report_json = json.dumps({
        "dataset_valid": report.dataset_valid,
        "blocking_issues": report.blocking_issues,
        "warnings": report.warnings,
        "summary": report.summary,
    }, indent=2, default=str)

    st.download_button(
        "Download validation_report.json",
        data=report_json.encode("utf-8"),
        file_name="input_validation_report.json",
        mime="application/json",
    )

_section_divider()


st.header("2) STL Residual Scoring")

run_stl_clicked = st.button("Run STL residual detector")
if run_stl_clicked:
    if report is not None and not report.dataset_valid:
        st.error("Validation failed. Fix blocking issues before running STL.")
    else:
        try:
            from src.models.stl_residual import stl_residual_detector
        except ModuleNotFoundError as e:
            st.error(
                "STL scoring requires the optional dependency 'statsmodels'. "
                "Install it with: pip install statsmodels"
            )
            st.stop()

        df_stl = df.copy()
        df_stl["date"] = pd.to_datetime(df_stl["date"], errors="coerce")

        with st.spinner("Running STL per meter... (can take time)"):
            df_scored = stl_residual_detector(df_stl, period=int(stl_period))

        st.session_state["df_stl"] = df_scored

stl_df = st.session_state.get("df_stl")
if stl_df is not None:
    st.subheader("STL output preview")
    st.dataframe(stl_df.head(50), use_container_width=True)

    out_path = os.path.join(artifact_dir, "consumption_stl.parquet")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Save consumption_stl.parquet"):
            _save_artifact(stl_df, out_path)
            st.success(f"Saved: {out_path}")
    with c2:
        st.download_button(
            "Download consumption_stl.parquet",
            data=_to_parquet_bytes(stl_df),
            file_name="consumption_stl.parquet",
            mime="application/octet-stream",
        )

_section_divider()


st.header("3) Hybrid Candidate Generation")

st.caption("Hybrid requires rolling outputs (predicted_anomaly + anomaly_score). If missing, you can load an existing parquet artifact.")

hyb_source = st.radio(
    "Rolling source",
    options=["Use current input (must already have predicted_anomaly + anomaly_score)", "Load consumption_rolling.parquet from artifacts folder"],
)

if st.button("Build hybrid candidates"):
    if stl_df is None:
        st.error("Run STL stage first (or load STL output into the app).")
    else:
        if hyb_source.startswith("Load"):
            roll_path = os.path.join(artifact_dir, "consumption_rolling.parquet")
            if not os.path.exists(roll_path):
                st.error(f"Missing rolling artifact: {roll_path}")
                st.stop()
            df_roll = pd.read_parquet(roll_path)
        else:
            df_roll = df.copy()

        missing = [c for c in ["predicted_anomaly", "anomaly_score", "meter_id", "date"] if c not in df_roll.columns]
        if missing:
            st.error(f"Rolling source is missing required columns: {missing}")
            st.stop()

        m = df_roll.merge(
            stl_df[["meter_id", "date", "stl_anomaly_score"]],
            on=["meter_id", "date"],
            how="left",
        )

        m = build_hybrid_flags(m, stl_global_quantile=float(hybrid_stl_global_quantile))
        st.session_state["df_hybrid"] = m

hyb_df = st.session_state.get("df_hybrid")
if hyb_df is not None:
    st.subheader("Hybrid output preview")
    st.write({
        "hybrid_days": int(hyb_df["hybrid_candidate"].sum()),
        "hybrid_meters": int(hyb_df.loc[hyb_df["hybrid_candidate"], "meter_id"].nunique()),
    })
    st.dataframe(hyb_df.head(50), use_container_width=True)

    out_path = os.path.join(artifact_dir, "consumption_hybrid.parquet")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Save consumption_hybrid.parquet"):
            _save_artifact(hyb_df, out_path)
            st.success(f"Saved: {out_path}")
    with c2:
        st.download_button(
            "Download consumption_hybrid.parquet",
            data=_to_parquet_bytes(hyb_df),
            file_name="consumption_hybrid.parquet",
            mime="application/octet-stream",
        )

_section_divider()


st.header("4) Incident Clustering")

if st.button("Cluster incidents"):
    if hyb_df is None:
        hyb_path = os.path.join(artifact_dir, "consumption_hybrid.parquet")
        if os.path.exists(hyb_path):
            hyb_df = pd.read_parquet(hyb_path)
        else:
            st.error("No hybrid dataframe in session and no consumption_hybrid.parquet found.")
            st.stop()

    required = ["meter_id", "date", "hybrid_candidate", "anomaly_score", "stl_anomaly_score"]
    missing = [c for c in required if c not in hyb_df.columns]
    if missing:
        st.error(f"Hybrid data missing required columns for incidents: {missing}")
        st.stop()

    with st.spinner("Clustering incidents..."):
        df_inc = cluster_incidents(hyb_df, gap_days=int(gap_days))

    st.session_state["df_incidents"] = df_inc

inc_df = st.session_state.get("df_incidents")
if inc_df is not None:
    st.subheader("Incidents")
    st.write({
        "incidents": int(len(inc_df)),
        "unique_meters": int(inc_df["meter_id"].nunique()) if len(inc_df) else 0,
        "mean_duration": float(inc_df["duration_days"].mean()) if len(inc_df) else None,
        "median_duration": float(inc_df["duration_days"].median()) if len(inc_df) else None,
    })
    st.dataframe(inc_df.head(200), use_container_width=True)

    out_path = os.path.join(artifact_dir, "incidents.parquet")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Save incidents.parquet"):
            _save_artifact(inc_df, out_path)
            st.success(f"Saved: {out_path}")
    with c2:
        st.download_button(
            "Download incidents.parquet",
            data=_to_parquet_bytes(inc_df),
            file_name="incidents.parquet",
            mime="application/octet-stream",
        )

_section_divider()


st.header("5) ML Re-ranking (Optional)")

st.caption("This UI shows and loads artifacts used by Random_Forest.py. Training is left to the script for now.")

risk_path = os.path.join(artifact_dir, "meter_risk.parquet")
labels_path = os.path.join(artifact_dir, "meter_labels.parquet")

c1, c2 = st.columns([1, 1])
with c1:
    st.write({"meter_risk.parquet exists": os.path.exists(risk_path), "path": risk_path})
with c2:
    st.write({"meter_labels.parquet exists": os.path.exists(labels_path), "path": labels_path})

if os.path.exists(risk_path):
    st.subheader("meter_risk preview")
    st.dataframe(pd.read_parquet(risk_path).head(50), use_container_width=True)

if os.path.exists(labels_path):
    st.subheader("meter_labels preview")
    st.dataframe(pd.read_parquet(labels_path).head(50), use_container_width=True)

st.info("Run ML training from terminal: python Random_Forest.py")
