import pandas as pd
from src.models.hybrid import build_hybrid_flags


df_roll = pd.read_parquet("data/processed/consumption_rolling.parquet")
df_stl  = pd.read_parquet("data/processed/consumption_stl.parquet")


df = df_roll.merge(
    df_stl[["meter_id", "date", "stl_anomaly_score"]],
    on=["meter_id", "date"],
    how="left"
)

df = build_hybrid_flags(df)


print(df.columns)

print("Hybrid days:", int(df["hybrid_candidate"].sum()))
print("Hybrid meters:", df.loc[df["hybrid_candidate"], "meter_id"].nunique())
