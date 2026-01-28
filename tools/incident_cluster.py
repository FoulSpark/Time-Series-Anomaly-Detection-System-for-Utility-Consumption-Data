from src.models.incidents import cluster_incidents
import pandas as pd
path = r"D:\Coading\Time-Series Anomaly Detection System for Utility Consumption Data\data\Processed\consumption_hybrid.parquet"
df = pd.read_parquet(path)
df_incidents = cluster_incidents(df)

print("Total incidents:", len(df_incidents))
print("Unique meters:", df_incidents["meter_id"].nunique())
print("Mean duration:", df_incidents["duration_days"].mean())
print("Median duration:", df_incidents["duration_days"].median())

df_incidents.head()
