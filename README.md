# Utility Time-Series Anomaly Detection & Inspection Prioritization

An end-to-end, explainable system for detecting abnormal electricity consumption patterns and prioritizing meters for inspection under limited operational capacity.

This project focuses on **inspection prioritization**, not just anomaly detection, and is designed to resemble a real internal analytics pipeline used by utility companies.

---

## Problem Statement

Utility providers experience losses due to electricity theft, faulty meters, and billing errors. These issues appear as abnormal consumption patterns in meter-level time-series data.

However:
- Consumption data is highly seasonal
- Ground-truth labels are scarce
- Physical inspections are expensive and limited

The challenge is **not** to perfectly classify anomalies, but to **rank meters by risk** so that inspection resources are used effectively.

---

## High-Level Approach

The system combines **statistical methods + time-series decomposition + ML re-ranking**:

```
Daily Consumption Data
↓
Rolling Statistics (Spike Detection)
↓
STL Decomposition (Drift Detection)
↓
Hybrid Anomaly Candidates
↓
Incident Clustering
↓
Meter-Level Risk Aggregation
↓
ML Re-Ranking (Optional)
↓
Prioritized Meter List
```

Each stage is explainable and produces persisted artifacts.

---

## Project Structure

```
.
├── data/
│ ├── raw/ # Input datasets (CSV / Parquet)
│ ├── schemas/ # JSON schema for validation
│ └── processed/ # Pipeline outputs (Parquet)
│
├── src/
│ ├── ingestion/ # Synthetic data generation
│ ├── models/
│ │ ├── rolling.py # Rolling z-score detector
│ │ ├── stl_residual.py # STL residual detector
│ │ ├── hybrid.py # Hybrid anomaly logic
│ │ ├── incidents.py # Incident clustering
│ │ ├── meter_risk.py # Meter-level risk scoring
│ │ └── ml_rerank.py # ML re-ranking
│ └── validation/
│ └── input_validator.py # Input acceptance gate
│
├── run_validate_input.py # Validate custom input only
├── run_pipeline.py # Run full pipeline
├── evaluation.py
├── Design.md # Detailed system design
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Validate your data (recommended)
```
python run_validate_input.py --input my_data.csv
```
### 3. Run the full pipeline
```
python run_pipeline.py --input my_data.csv
```
## Outputs

All pipeline outputs are written to:


as **Parquet files**, one per pipeline stage.  
Raw user data is never modified.

---

## Using Your Own Data

### Accepted Input Format

The pipeline accepts **CSV or Parquet** files with **daily meter-level data**.

### Required Columns

| Column | Description |
|------|------------|
| `meter_id` | Unique meter identifier |
| `date` | Date (YYYY-MM-DD) |
| `consumption_kwh` | Daily energy consumption (≥ 0) |

### Optional Columns

- `region_id`
- `customer_type`
- `temperature_c`
- `is_holiday`

Missing optional columns are allowed.

---

## Input Validation (Important)

Before running detection, all inputs are validated for:

- Schema correctness
- Duplicate `(meter_id, date)` rows
- Sufficient historical coverage per meter
- Negative or invalid consumption values
- Suspicious flatlines or sparse readings

### Validation Results

- Printed to the console
- Saved as a JSON report

If blocking issues are found, the pipeline does **not** run.  
This prevents meaningless anomaly scores.

---

## Pipeline Phases (What Happens Internally)

### Phase 1 – Rolling Statistical Baseline
Detects abrupt spikes and drops using per-meter rolling z-scores.

### Phase 2 – STL Residual Detection
Removes seasonal structure and detects slow distributional drift.

### Phase 3 – Hybrid Detection
Combines rolling and STL signals to improve anomaly coverage.

### Phase 4 – Incident Clustering
Groups adjacent anomalous days into incidents to reduce alert noise.

### Phase 5 – Meter-Level Risk Scoring
Aggregates incidents into interpretable meter-level risk features.

### Phase 6 – ML Re-Ranking (Optional)
Uses a Random Forest model to re-rank meters and improve inspection efficiency.

---

## Evaluation Summary

- **Precision@10 improved from 0.20 → 0.60** with ML re-ranking
- ~3× improvement in inspection efficiency
- No data leakage (meter-level train/test split)
- ML augments, not replaces, rule-based logic

---

## Design Principles

- Explainability first
- Weak-supervision aware
- Artifact-driven pipeline
- ML used only where it adds measurable value
- Clear separation of validation, detection, and ranking

---

## What This Project Is Not

- Not a real-time streaming system
- Not an automated theft accusation engine
- Not a deep learning black box
- Not a dashboard or UI-heavy project

---

## Documentation

- `Design.md` — full system design and assumptions
- Inline docstrings and comments in code
- Persisted artifacts at every pipeline stage

---

## Author

**Mayank Rautiya**

This project demonstrates end-to-end system design, anomaly detection under weak supervision, and production-aware ML application.
