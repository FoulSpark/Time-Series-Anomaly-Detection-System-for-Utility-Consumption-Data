# Time-Series Anomaly Detection & Inspection Prioritization  
### Utility Consumption Data (India-Focused)

---

## Executive Summary

Electricity utilities incur significant losses due to theft, faulty meters, and billing errors. These issues manifest as abnormal consumption patterns in large-scale time-series data. However, inspecting every meter is operationally infeasible.

This project builds an **end-to-end anomaly detection and inspection prioritization system** for utility consumption data. It combines **statistical baselines, seasonal decomposition, and machine learning re-ranking** to identify and prioritize high-risk meters for inspection.

The system is designed to be:
- Scalable
- Explainable
- Operationally realistic
- Robust under weak supervision

The final output is a **ranked list of meters** that maximizes inspection efficiency under limited resources.

---

## System Overview 
```
Raw Consumption Data
↓
Rolling Statistical Baseline (Spike Detection)
↓
STL Residual Analysis (Drift Detection)
↓
Hybrid Anomaly Candidate Generation
↓
Incident Clustering (Temporal Consolidation)
↓
Meter-Level Risk Aggregation
↓
ML Re-Ranking (Phase 5)
↓
Final Inspection Priority List
```

## Project Structure
```
.
├── data/
│ ├── raw/ # Synthetic CSV data
│ ├── schemas/ # JSON schema validation
│ └── processed/
│ ├── consumption_rolling.parquet
│ ├── consumption_stl.parquet
│ ├── consumption_hybrid.parquet
│ ├── incidents.parquet
│ ├── meter_risk.parquet
│ └── meter_labels.parquet
│
├── src/
│ ├── ingestion/ # Synthetic data generation
│ └── models/
│ ├── rolling.py # Rolling z-score detector
│ ├── stl_residual.py # STL residual detector
│ ├── hybrid.py # Hybrid candidate logic
│ └── incidents.py # Incident clustering
│
├── run_stl.py
├── run_hybrid.py
├── evaluation.py
└── README.md
```


---

## Phase 1 — Problem Framing

**Objective**  
Detect abnormal electricity consumption patterns and prioritize meters for inspection.

**Challenges**
- Strong seasonal effects (AC usage, heating, agriculture)
- Weak ground truth labels
- Large data volume
- Limited inspection capacity

**Design Decision**  
Focus on **inspection prioritization**, not perfect anomaly classification.

---

## Phase 2 — Data Generation & Validation

- Generated 2 years of daily consumption for 300 meters
- Customer types:
  - Residential
  - Commercial
  - Industrial
  - Agricultural
- Injected realistic anomalies:
  - Sudden spikes (theft / tampering)
  - Sudden drops (faulty meters)
  - Gradual drift (bypass / degradation)
- Enforced strict JSON schema validation
- Persisted data using Parquet

**Output**
- `consumption_daily.csv`
- Schema-validated, reproducible synthetic data

---

## Phase 3 — Statistical Baselines

### Rolling Z-Score Detector
- 14-day rolling mean and standard deviation
- Per-meter normalization
- Detects abrupt consumption spikes
- High precision, low recall

### STL Residual Detector
- Seasonal-Trend decomposition (STL)
- Robust MAD-based z-score on residuals
- Detects slow distributional drift
- High recall, noisy

**Key Insight**
Neither method alone is sufficient.

---

## Phase 4 — Hybrid Detection & Incident Prioritization

### Hybrid Anomaly Logic

A day is flagged as anomalous if:
- Rolling detector fires **OR**
- STL residual is in the **top 0.5% globally**

This balances precision and coverage.

---

### Incident Clustering

- Operates on hybrid day-level anomalies
- Groups temporally adjacent anomalous days into incidents
- `gap_days = 2` (validated via sensitivity analysis)

**Results**
- ~2,300 incidents over 300 meters
- Median incident duration: 1 day
- Stable across gap parameter variations

---

### Meter-Level Risk Aggregation

Incidents are aggregated per meter to compute:
- Number of incidents
- Total anomalous days
- Maximum rolling severity
- Maximum STL severity

A final interpretable risk score is computed:
```
final_risk_score =
0.6 × rolling_score_percentile
0.3 × stl_score_percentile
0.1 × log(1 + total_incident_days)
```


**Output**
- `meter_risk.parquet`
- Final rule-based inspection ranking

---

## Phase 5 — Machine Learning Re-Ranking

### Motivation
Improve inspection efficiency by learning how to optimally combine Phase-4 signals.

ML is used as a **re-ranking layer**, not a replacement for detectors.

---

### Label Definition (Weak Supervision)

Meters are labeled as **high-risk** if they fall in the **top 20% by true anomaly days**.

- Total meters: 300
- High-risk meters: 67
- Low-risk meters: 233

This reflects chronic risk, not isolated noise.

---

### Model

- RandomForestClassifier
- Shallow trees, balanced class weights
- Inputs: Phase-4 meter-level features only

---

### Evaluation Metrics

Primary:
- **Precision@K (meters)**

Secondary:
- ROC AUC

---

### Results

| Metric | ML Model | Rule-Based Baseline |
|------|---------|--------------------|
| ROC AUC | 0.79 | — |
| Precision@10 | **0.60** | 0.20 |
| Precision@20 | **0.50** | 0.30 |
| Precision@30 | **0.47** | 0.27 |

**Interpretation**
- ML improves inspection efficiency by ~3× at top-10
- Gains are consistent across K
- No data leakage or overfitting

---

## Final Outcome

The system delivers:
- Explainable anomaly detection
- Incident-level consolidation
- Meter-level prioritization
- Demonstrated ML uplift

This mirrors real-world utility analytics pipelines and is suitable for deployment, extension, or research.

---

## Key Design Principles

- Explainability over black-box models
- Weak supervision handled explicitly
- ML augments rules, does not replace them
- Reproducible, artifact-driven pipeline

---

## Future Extensions

- Weather-aware normalization
- Geographic clustering
- Temporal ML models for drift continuity
- Integration with inspection feedback loops

---

### Accepted Input Format

The pipeline accepts CSV or Parquet files with daily meter-level data.

Required columns:
- meter_id (string)
- date (ISO date)
- consumption_kwh (float, ≥ 0)

Optional columns:
- region_id
- customer_type
- temperature_c
- is_holiday

## Author

**Mayank Rautiya**  
End-to-end system design, modeling, and evaluation

---

