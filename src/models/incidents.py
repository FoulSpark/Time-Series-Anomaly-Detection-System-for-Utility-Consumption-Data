import pandas as pd

def cluster_incidents(df :pd.DataFrame , gap_days: int = 2) -> pd.DataFrame:
    incidents = []

    for meter_id,mdf in df[df["hybrid_candidate"]].groupby("meter_id"):
        mdf = mdf.sort_values("date") 

        current = []

        for _,row in mdf.iterrows():
            if not current:
                current = [row]
                continue

            last_date = current[-1]["date"]
            if (row["date"]-last_date).days <= gap_days:
                current.append(row)
            else:
                incidents.append(_summarize(current))
                current = [row]

        if current:
            incidents.append(_summarize(current))

    return pd.DataFrame(incidents)

def _summarize(rows):   
    meter_id = rows[0]["meter_id"]
    start = rows[0]["date"]
    end = rows[-1]["date"]
    duration = len(rows)


    max_roll = max( 
        [r["anomaly_score"] for r in rows if pd.notna(r["anomaly_score"])],
        default = 0
    )

    max_stl = max( 
        [r["stl_anomaly_score"] for r in rows if pd.notna(r["stl_anomaly_score"])],
        default = 0
    )

    dominant = "spike" if max_roll >= max_stl else "drift"
    
    return {
        "meter_id": meter_id,
        "start_date": start,
        "end_date": end,
        "duration_days": duration,
        "max_rolling_score": max_roll,
        "max_stl_score": max_stl,
        "dominant_signal": dominant,
    }