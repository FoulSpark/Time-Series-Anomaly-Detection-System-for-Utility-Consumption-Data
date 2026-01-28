import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from jsonschema import validate
import json
import os

np.random.seed(42)

NUM_METERS = 300
NUM_DAYS = 730
START_DATE = datetime(2023, 1, 1)
ANOMALY_RATE = 0.02

SCHEMA_PATH = "data/schemas/consumption_schema.json"
OUTPUT_PATH = "data/raw/consumption_daily.csv"

def generate_meter_profiles():
    customer_types = ["residential" , "commercial" , "industrial" , "agri"]
    weights = [0.7 , 0.2 , .05 , .05]

    profiles = []

    for i in range(NUM_METERS):
        ctype = np.random.choice(customer_types, p = weights)

        if ctype == "residential":
            base = np.random.uniform(4,8)
            seasonality = .35
        elif ctype == "commercial":
            base = np.random.uniform(10,25)
            seasonality = .25
        elif ctype == "industrial":
            base = np.random.uniform(30,80)
            seasonality = .10
        elif ctype == "agri":
            base = np.random.uniform(15,40)
            seasonality = .30

        profiles.append({
            "meter_id" : f"M_{i:04d}",
            "customer_type": ctype,
            "base_consumption" : base,
            "seasonality_strength":seasonality,
            "region_id": f"R_{np.random.randint(1,6)}"
        })

    return profiles

def generate_normal_series(profile):
    rows = []

    for i in range(NUM_DAYS):
        date = START_DATE + timedelta(days=i)
        day_of_year = date.timetuple().tm_yday

        seasonal_factor = 1 + profile["seasonality_strength"] * np.sin(2 * np.pi * day_of_year / 365)

        noise = np.random.normal(0, 0.1 * profile["base_consumption"])

        consumption = max(0 ,profile["base_consumption"] * seasonal_factor + noise )

        rows.append({
            "meter_id" : profile["meter_id"],
            "region_id" : profile["region_id"],
            "customer_type":profile["customer_type"],
            "date": date.strftime("%Y-%m-%d"),
            "consumption_kwh":round(consumption,2),
            "temperature_c": None,
            "is_holiday":None,
            "data_source":"synthetic",
            "generated_anomaly":False
        })

    return rows
    
def inject_anomalies(df):
    target_anom_rows = int(len(df) * ANOMALY_RATE)

    marked = set(df.index[df["generated_anomaly"] == True].tolist())

    max_iters = target_anom_rows * 10
    iters = 0

    while len(marked) < target_anom_rows and iters < max_iters:
        iters += 1

        idx = int(np.random.choice(df.index, size=1)[0])
        if idx in marked:
            continue

        anomaly_type = np.random.choice(["spike", "drop", "drift"], p=[0.6, 0.3, 0.1])

        if anomaly_type == "spike":
            df.loc[idx, "consumption_kwh"] *= np.random.uniform(2.5, 4.0)
            df.loc[idx, "generated_anomaly"] = True
            marked.add(idx)

        elif anomaly_type == "drop":
            df.loc[idx, "consumption_kwh"] *= np.random.uniform(0.1, 0.3)
            df.loc[idx, "generated_anomaly"] = True
            marked.add(idx)

        else:
            meter_id = df.loc[idx, "meter_id"]
            meter_indices = df[df["meter_id"] == meter_id].index.tolist()

            if len(meter_indices) < 14:
                continue

            start_pos = np.random.randint(0, len(meter_indices) - 14)
            window = meter_indices[start_pos:start_pos + 14]

            new_indices = [w for w in window if w not in marked]
            if len(marked) + len(new_indices) > target_anom_rows:
                continue

            df.loc[window, "consumption_kwh"] *= np.random.uniform(1.5, 2.0)
            df.loc[window, "generated_anomaly"] = True
            marked.update(window)

    return df



def validate_schema(df, sample_size=500):

    required_cols = ["meter_id", "date", "consumption_kwh", "customer_type", "region_id", "data_source"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if (df["consumption_kwh"] < 0).any():
        raise ValueError("Found negative consumption_kwh")

    allowed_types = {"residential", "commercial", "industrial", "agri"}
    if not set(df["customer_type"].unique()).issubset(allowed_types):
        raise ValueError("Invalid customer_type found")

    if not set(df["data_source"].unique()).issubset({"synthetic"}):
        raise ValueError("Invalid data_source found")

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    for _, row in sample.iterrows():
        validate(instance=row.to_dict(), schema=schema)


def save_csv(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH),exist_ok=True)
    df.to_csv(OUTPUT_PATH , index = False)



def main():
    profiles = generate_meter_profiles()

    all_rows = []
    for profile in profiles:
        all_rows.extend(generate_normal_series(profile))

    df = pd.DataFrame(all_rows)
    df = inject_anomalies(df)

    validate_schema(df)
    save_csv(df)
    print(f"Generated {len(df)} rows")
    print(f"Anomalies injected: {df['generated_anomaly'].sum()}")


if __name__ == "__main__":
    main()
