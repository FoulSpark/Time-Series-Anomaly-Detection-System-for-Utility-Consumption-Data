import pandas as pd


def build_hybrid_flags(
    df: pd.DataFrame,
    stl_global_quantile: float = 0.995
) -> pd.DataFrame:
    """
    Hybrid candidate generation (day-level).

    Rules:
      A) Keep high-confidence rolling anomalies: predicted_anomaly == True
      B) Add only the strongest STL signals: stl_anomaly_score >= global q (default top 0.5%)

    Returns df with:
      - hybrid_candidate (bool)
      - stl_global_threshold (float, stored as metadata column)
    """
    df = df.copy()

    if "predicted_anomaly" not in df.columns:
        raise ValueError("Missing column: predicted_anomaly (rolling flags)")
    if "stl_anomaly_score" not in df.columns:
        raise ValueError("Missing column: stl_anomaly_score (STL scores)")

    stl_threshold = df["stl_anomaly_score"].quantile(stl_global_quantile)

    df["hybrid_candidate"] = (df["predicted_anomaly"] == True) | (
        df["stl_anomaly_score"] >= stl_threshold
    )

  
    df["stl_global_threshold"] = stl_threshold

    return df
