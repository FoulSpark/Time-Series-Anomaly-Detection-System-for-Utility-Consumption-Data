Utility Time-Series Anomaly Detection & Inspection Prioritization System
===============================================================

PROJECT NAME
------------
Time-Series Anomaly Detection System for Utility Consumption Data

PRIMARY GOAL (WHAT THIS PROJECT DOES)
------------------------------------
This project implements an end-to-end, explainable pipeline that:
- detects abnormal electricity consumption behavior in daily smart-meter time series, and
- converts day-level anomaly signals into an operationally useful output: a ranked list of meters to inspect,
  assuming inspection teams have limited capacity.

The system is designed around a realistic utility analytics framing:
- labels are scarce or weak ("weak supervision"),
- seasonality is strong (weather, agriculture, heating/cooling), and
- a production-style workflow is preferred (validate inputs, generate artifacts at each stage, and keep logic explainable).

The repository includes:
- synthetic data generation (India-focused customer types)
- input validation gate for user-provided datasets
- anomaly detection via STL residuals (implemented)
- hybrid candidate generation combining rolling + STL signals (hybrid logic implemented; rolling stage is referenced)
- incident clustering to reduce alert noise
- meter-level risk aggregation (features referenced in ML script; risk artifact present as parquet)
- optional ML re-ranking with a Random Forest classifier (script provided)

NOTE ABOUT CURRENT REPO STATE
-----------------------------
The README.md files describe a full multi-stage pipeline including a rolling z-score detector and meter-level risk scoring.
In this code snapshot, only the following core model modules are present under src/models:
- src/models/stl_residual.py
- src/models/hybrid.py
- src/models/incidents.py

There is no src/models/rolling.py or src/models/meter_risk.py in this workspace, but the repository already contains the
corresponding *outputs* as Parquet files under data/Processed (consumption_rolling.parquet, meter_risk.parquet, etc.).
So, the repository behaves like a “pipeline + artifacts” project where some stages may have been run previously or exist
outside this snapshot.

If you want this README.txt to exactly match the code, treat:
- stl_residual.py, hybrid.py, incidents.py, validation/input_validator.py, ingestion/generate_synthetic_data.py
as authoritative, and treat the rolling detector + meter risk scoring as referenced-by-artifacts.

HIGH-LEVEL PIPELINE (CONCEPTUAL)
--------------------------------
Daily Consumption Data (meter_id, date, consumption_kwh)
  -> Input Validation Gate (schema/quality checks)
  -> Rolling Statistical Baseline (spike/drop detection)  [referenced in docs; artifact exists]
  -> STL Decomposition on each meter (drift/seasonality-adjusted detection)  [implemented]
  -> Hybrid day-level candidate logic (rolling OR strong STL)  [implemented]
  -> Incident Clustering (group adjacent anomalous days)  [implemented]
  -> Meter-Level Risk Aggregation (incidents -> meter features + risk score)  [artifact exists]
  -> ML Re-Ranking (Random Forest)  [script provided]
  -> Final prioritized meter list


REPOSITORY STRUCTURE
--------------------
Top-level (project root):
- README.md
  High-level project overview and quick start.
- Random_Forest.py
  Trains/evaluates a RandomForestClassifier to re-rank meters using meter-level risk features.
- run_validate_input.py
  CLI entry point to validate an input dataset and write a JSON validation report.
- run_stl.py
  CLI entry point intended to run an STL stage (currently imports run_stl which is not defined in stl_residual.py).
- data/
  - Raw/
    - consumption_daily.csv
      Synthetic dataset (daily rows). Contains column generated_anomaly for evaluation.
  - Processed/
    - consumption_rolling.parquet
      Day-level output of rolling detector stage (artifact). Expected to include anomaly_score + predicted_anomaly.
    - consumption_stl.parquet
      Day-level output after STL residual detection stage (artifact). Expected to include residual + stl_anomaly_score + stl_predicted_anomaly.
    - consumption_hybrid.parquet
      Day-level output after merging rolling + STL and applying hybrid candidate logic (artifact). Expected to include hybrid_candidate.
    - incidents.parquet
      Incident-level table (artifact). Each row summarizes a cluster of anomalous days for a meter.
    - meter_risk.parquet
      Meter-level aggregated features and final risk score (artifact). Used by Random_Forest.py.
    - meter_labels.parquet
      Meter-level weak labels used for ML evaluation (artifact).
  - schemas/
    - consumption_schema.json
      JSON schema used to validate generated synthetic data.
- src/
  - README.md
    A long-form narrative of the system phases and intended outputs.
  - ingestion/
    - generate_synthetic_data.py
      Generates synthetic utility consumption data and injects anomalies.
  - models/
    - stl_residual.py
      STL residual anomaly scoring per meter.
    - hybrid.py
      Hybrid candidate generation combining rolling and STL signals.
    - incidents.py
      Clusters anomalous days into incidents.
  - validation/
    - input_validator.py
      Input dataset validation logic and report formatting.
- tools/
  Standalone scripts to run parts of the pipeline on existing artifacts.
  - evaluation.py
    Example evaluation of STL predictions vs generated_anomaly.
  - hybrid.py
    Example script merging rolling + STL parquet outputs and applying hybrid logic.
  - incident_cluster.py
    Example script to cluster hybrid day-level candidates into incident summaries.
- tests/
  - test.py
    Simple label exploration: per-meter label if any generated anomaly exists.
  - test2.py
    Creates weak labels by top-20% meters ranked by true anomaly days.


DATA MODEL AND SCHEMAS
----------------------
There are effectively two “schemas” in the project:

(A) Minimal schema for *user-provided* datasets (validation/input_validator.py)
Required columns:
- meter_id: identifier (converted to string)
- date: parseable to datetime
- consumption_kwh: numeric, non-negative

Optional columns (allowed but not required):
- region_id
- customer_type
- temperature_c
- is_holiday
- data_source
- generated_anomaly

Important assumptions in validator:
- daily granularity per meter
- one row per (meter_id, date)

(B) Synthetic-data JSON schema (data/schemas/consumption_schema.json)
Required:
- meter_id (string)
- date (string, format=date)
- consumption_kwh (number >= 0)
- customer_type (enum: residential/commercial/industrial/agri)
- region_id (string)
- data_source (enum: synthetic)

Optional:
- temperature_c (number|null)
- is_holiday (boolean|null)
- generated_anomaly (boolean|null)


INPUT VALIDATION (src/validation/input_validator.py)
---------------------------------------------------
Purpose:
- prevent scoring on unusable data (garbage in -> garbage out)
- produce an explicit ValidationReport with blocking issues + warnings + summary

Key checks (blocking):
1) Required columns present
2) meter_id not null
3) date parseable to datetime
4) consumption_kwh numeric (no NaN after coercion)
5) consumption_kwh non-negative
6) no duplicate keys on (meter_id, date)
7) sufficient history per meter (min_days_per_meter, default 30)

Key checks (warnings):
1) extra columns (ignored)
2) very sparse meter histories (> max_missing_ratio_per_meter, default 50% missing days in range)
3) flatline runs >= max_flatline_run (default 14) indicating stuck meters/ETL issues
4) suspicious scale hints:
   - p99 > 100,000 kWh suggests possible unit mismatch or aggregation
   - median < 0.01 kWh suggests Wh vs kWh mismatch

Outputs:
- console human-readable report via print_report
- JSON report file via report.save(path)

CLI usage:
- python run_validate_input.py --input <CSV|PARQUET> --report-out reports/input_validation_report.json
Exit codes:
- exits with code 1 if blocking issues exist
- exits with code 0 otherwise


SYNTHETIC DATA GENERATION (src/ingestion/generate_synthetic_data.py)
-------------------------------------------------------------------
Goal:
- provide a reproducible dataset with realistic seasonality and injected anomalies
- used to test/evaluate the pipeline when real labels/data aren’t available

Constants:
- NUM_METERS = 300
- NUM_DAYS = 730 (2 years)
- START_DATE = 2023-01-01
- ANOMALY_RATE = 0.02 (about 2% of total rows)
- OUTPUT_PATH = data/raw/consumption_daily.csv

Customer types and weights:
- residential (70%)
- commercial (20%)
- industrial (5%)
- agri (5%)

Profile fields per meter:
- meter_id: M_0000 ...
- customer_type
- base_consumption range depends on type
- seasonality_strength depends on type
- region_id: R_1..R_5

Normal series generation:
- daily seasonal factor: 1 + seasonality_strength * sin(2*pi*day_of_year/365)
- noise: Normal(0, 0.1 * base_consumption)
- consumption: max(0, base * seasonal_factor + noise)
- generated columns include temperature_c and is_holiday as None placeholders

Anomaly injection (inject_anomalies):
- chooses random rows until about ANOMALY_RATE of rows are marked
- anomaly types sampled with probabilities:
  - spike (0.6): multiply consumption by U(2.5, 4.0)
  - drop  (0.3): multiply consumption by U(0.1, 0.3)
  - drift (0.1): pick a meter and a 14-day window, multiply consumption by U(1.5, 2.0)
- safeguards:
  - prevents re-marking already anomalous rows
  - drift window requires meter has at least 14 days
  - max iterations to avoid infinite loop: target_anom_rows * 10

Schema validation:
- checks required columns and value constraints
- validates a sample of up to 500 rows against consumption_schema.json using jsonschema.validate

Run:
- python src/ingestion/generate_synthetic_data.py

Output:
- data/raw/consumption_daily.csv
- printed summary: total rows + anomaly count


STL RESIDUAL DETECTOR (src/models/stl_residual.py)
-------------------------------------------------
Goal:
Detect anomalies that remain after removing seasonal + trend components.
This is especially useful for gradual drift, slow changes, or deviations masked by seasonality.

Core functions:

1) robust_zscore_mad(x: pd.Series) -> pd.Series
- computes robust z-score using MAD (Median Absolute Deviation)
- formula: z = 0.6745 * (x - median) / MAD
- returns NaN series if MAD is 0 or NaN (cannot scale)

2) stl_residual_detector(df: pd.DataFrame, period=365, threshold=3.5) -> pd.DataFrame
IMPORTANT: In current code, the threshold argument is not used.
Instead:
- computes absolute robust z-score on residuals
- then uses a per-meter threshold at z.quantile(0.98)
- flags anomalies where z > that quantile

Step-by-step:
- sort by meter_id, date
- add output columns initialized:
  - residual (float)
  - stl_anomaly_score (float)
  - stl_predicted_anomaly (bool)
- for each meter_id group:
  - y = consumption_kwh values
  - run STL(y, period=365, robust=True)
  - residual = res.resid
  - z = abs(robust_zscore_mad(residual))
  - t = z.quantile(0.98)  (meter-specific)
  - pred = z > t
  - store residual/z/pred back into main df
- returns df with STL outputs

Failure handling:
- if STL fails for a meter, prints: "STL failed for <meter_id>: <error>" and skips the meter

Expected input columns:
- meter_id
- date (should be datetime-like or parseable)
- consumption_kwh

Expected output columns:
- residual
- stl_anomaly_score
- stl_predicted_anomaly


HYBRID CANDIDATE GENERATION (src/models/hybrid.py)
-------------------------------------------------
Goal:
Combine a high-precision rolling detector with a high-recall STL detector.
The intent is:
- keep clear spikes/drops detected by rolling baseline
- also capture strongest STL residual deviations that indicate drift

Function:
build_hybrid_flags(df: pd.DataFrame, stl_global_quantile: float = 0.995) -> pd.DataFrame

Requirements (columns must exist):
- predicted_anomaly (rolling stage boolean)
- stl_anomaly_score (STL stage score)

Logic:
- compute a global STL threshold: quantile(stl_global_quantile) across entire df
  default 0.995 => top 0.5% strongest STL scores
- hybrid_candidate is True if:
  - predicted_anomaly is True OR
  - stl_anomaly_score >= global STL threshold
- store the numeric threshold in column stl_global_threshold for traceability

Output columns:
- hybrid_candidate (bool)
- stl_global_threshold (float)


INCIDENT CLUSTERING (src/models/incidents.py)
---------------------------------------------
Goal:
Operational teams don’t want to inspect “anomaly days”; they inspect “issues/incidents”.
This stage clusters adjacent anomalous days into a single incident.

Function:
cluster_incidents(df: pd.DataFrame, gap_days: int = 2) -> pd.DataFrame

Inputs:
- df must have:
  - meter_id
  - date
  - hybrid_candidate (boolean)
  - anomaly_score (rolling severity)  (referenced)
  - stl_anomaly_score (STL severity)

Clustering logic:
- filter to df[df["hybrid_candidate"]]
- for each meter:
  - sort by date
  - walk days sequentially
  - if the gap between current day and previous flagged day is <= gap_days, keep in same incident
  - otherwise, close the current incident and start a new one
- summarize each incident using _summarize(rows)

Incident summary fields:
- meter_id
- start_date
- end_date
- duration_days  (number of flagged rows in the incident)
- max_rolling_score  (max anomaly_score in incident)
- max_stl_score      (max stl_anomaly_score in incident)
- dominant_signal
  - "spike" if max_rolling_score >= max_stl_score
  - otherwise "drift"

Output:
- DataFrame with one row per incident


METER-LEVEL RISK FEATURES + ML RE-RANKING
-----------------------------------------
This repo uses a two-level approach:

1) Rule-based risk score (artifact: data/Processed/meter_risk.parquet)
- The long-form src/README.md describes an interpretable final_risk_score composition:
  final_risk_score =
    0.6 * rolling_score_percentile +
    0.3 * stl_score_percentile +
    0.1 * log(1 + total_incident_days)

- The Random_Forest.py script expects these columns in meter_risk.parquet:
  FEATURES used for ML training:
  - num_incidents
  - total_incident_days
  - max_rolling_score
  - max_stl_score
  - roll_pct
  - stl_pct
  - final_risk_score

2) Weak labels for ML (artifact: data/Processed/meter_labels.parquet)
- tests/test2.py creates labels from the synthetic truth column generated_anomaly:
  Steps:
  - load data/processed/consumption_hybrid.parquet
  - compute per-meter true_anomaly_days by counting rows where generated_anomaly == True
  - compute threshold = 80th percentile (quantile 0.80)
  - label meter as high-risk if true_anomaly_days >= threshold
  - save meter_id, true_anomaly_days, label to data/processed/meter_labels.parquet


RANDOM FOREST RE-RANKING (Random_Forest.py)
-------------------------------------------
Purpose:
Train a simple ML model to re-rank meters more efficiently than the rule-based final_risk_score.

Inputs:
- data/processed/meter_risk.parquet
- data/processed/meter_labels.parquet

Process:
- inner join on meter_id
- x_data = meter-level FEATURES
- y_data = label (bool high-risk)
- split: train_test_split(test_size=0.3, random_state=42, stratify=y)
- model: RandomForestClassifier
  - n_estimators=300
  - max_depth=6
  - class_weight="balanced"
  - min_samples_leaf=5

Metrics:
- ROC AUC on predicted probabilities
- Precision@K for K in {10,20,30}
- Baseline Precision@K using final_risk_score

Interpretation:
- model acts as a re-ranking layer (does not replace explainable detectors)


TOOLS / HELPER SCRIPTS
----------------------
- tools/evaluation.py
  Demonstrates evaluating STL predictions against synthetic ground truth.
  It loads data/raw/consumption_daily.csv, runs stl_residual_detector, then prints:
  - confusion matrix counts
  - precision/recall
  - precision@100 and precision@500 using stl_anomaly_score

- tools/hybrid.py
  Demonstrates building hybrid candidates from existing parquet artifacts:
  - reads data/processed/consumption_rolling.parquet
  - reads data/processed/consumption_stl.parquet
  - merges stl_anomaly_score into rolling df by (meter_id, date)
  - calls build_hybrid_flags

- tools/incident_cluster.py
  Demonstrates incident clustering from consumption_hybrid.parquet.


ARTIFACTS (FILES WRITTEN BY THE PIPELINE)
-----------------------------------------
Artifacts are stored under data/Processed/ (note: folder casing in this repo is "Processed").
These are treated as persisted stage outputs.

1) data/Raw/consumption_daily.csv
- raw synthetic dataset
- includes generated_anomaly for evaluation

2) data/Processed/consumption_rolling.parquet
- day-level rolling detector output
- expected columns include predicted_anomaly and anomaly_score (based on downstream usage)

3) data/Processed/consumption_stl.parquet
- day-level STL output
- includes residual, stl_anomaly_score, stl_predicted_anomaly

4) data/Processed/consumption_hybrid.parquet
- day-level merged output after hybrid candidate generation
- includes hybrid_candidate plus upstream columns

5) data/Processed/incidents.parquet
- incident-level clustered summaries

6) data/Processed/meter_risk.parquet
- meter-level features and final_risk_score

7) data/Processed/meter_labels.parquet
- weak labels derived from synthetic truth


HOW TO RUN (PRACTICAL)
----------------------
Because this repo snapshot does not contain a single run_pipeline.py entry point, the most reliable way to reproduce
results is to run stages via the scripts that exist.

A) Generate synthetic data
1) python src/ingestion/generate_synthetic_data.py
   Output: data/raw/consumption_daily.csv

B) Validate your dataset (synthetic or real)
2) python run_validate_input.py --input data/raw/consumption_daily.csv
   Output: reports/input_validation_report.json

C) STL residual scoring (implemented as a function)
There is no fully-wired CLI wrapper in this snapshot, but you can use tools/evaluation.py as an example.
3) python tools/evaluation.py
   Note: tools/evaluation.py uses an absolute Windows path string; you may need to edit it to your local path.

D) Hybrid candidates
4) python tools/hybrid.py
   Requires that consumption_rolling.parquet and consumption_stl.parquet already exist.

E) Incident clustering
5) python tools/incident_cluster.py
   Requires consumption_hybrid.parquet.

F) Weak labels (for synthetic evaluation)
6) python tests/test2.py
   Output: data/processed/meter_labels.parquet (in this repo, artifacts also exist already)

G) Random Forest ML re-ranking
7) python Random_Forest.py
   Prints ROC AUC and Precision@K for model and baseline.


DEPENDENCIES (INFERRED)
-----------------------
No requirements.txt is present in this workspace.
From imports, you likely need:
- python 3.x
- numpy
- pandas
- scikit-learn
- statsmodels
- jsonschema

If you want, I can generate a requirements.txt consistent with these imports.


KNOWN INCONSISTENCIES / GOTCHAS
-------------------------------
- run_stl.py imports `run_stl` from src/models/stl_residual.py, but no function named run_stl exists there.
  The actual callable is stl_residual_detector(df, period=365, ...).
- Folder casing in docs vs filesystem:
  docs mention data/raw and data/processed; repo folders are data/Raw and data/Processed.
  Windows is case-insensitive, but this matters on Linux/macOS.
- tools/evaluation.py and tools/incident_cluster.py use absolute Windows paths. This reduces portability.
- stl_residual_detector has a `threshold` argument that is not used; it instead uses per-meter 0.98 quantile.


WHAT TO LOOK AT FIRST (IF YOU’RE READING THE CODE)
--------------------------------------------------
- src/validation/input_validator.py
  Defines accepted input data and quality gates.
- src/ingestion/generate_synthetic_data.py
  Explains how synthetic anomalies are created (ground truth generated_anomaly).
- src/models/stl_residual.py
  Core anomaly scoring logic (robust STL residual z-scores).
- src/models/hybrid.py
  Explains how different detectors are combined into candidate anomalies.
- src/models/incidents.py
  Converts noisy day-level flags into incident-level summaries.
- Random_Forest.py
  Demonstrates the inspection-efficiency uplift using ML re-ranking.


PROJECT OUTCOME
---------------
The end product is not just “an anomaly flag”, but a prioritization workflow:
- detect
- consolidate
- aggregate
- rank

This makes the system actionable for utility operations teams who must decide which meters to inspect next.
