import pandas as pd

df = pd.read_parquet("data/processed/consumption_hybrid.parquet")

meter_truth = (
    df[df["generated_anomaly"]]
    .groupby("meter_id")
    .size()
    .reset_index(name="true_anomaly_days")
)

threshold = meter_truth["true_anomaly_days"].quantile(0.80)

meter_truth["label"] = (
    meter_truth["true_anomaly_days"] >= threshold
)

meter_truth.to_parquet(
    "data/processed/meter_labels.parquet",
    index=False
)

print("Threshold used:", threshold)
print(meter_truth["label"].value_counts())

print(meter_truth.describe())
