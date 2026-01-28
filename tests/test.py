import pandas as pd

df = pd.read_parquet("data/processed/consumption_hybrid.parquet")

labels = (
    df.groupby("meter_id")["generated_anomaly"]
      .max()
      .reset_index()
      .rename(columns={"generated_anomaly": "label"})
)

print(labels["label"].value_counts())
