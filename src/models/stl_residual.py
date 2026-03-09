import numpy as np 
import pandas as pd
try:
    from statsmodels.tsa.seasonal import STL
except ModuleNotFoundError:  
    STL = None


def robust_zscore_mad(x:pd.Series)  -> pd.Series:
    """
    Robust z-score using Median Absolute Deviation (MAD).
    z = 0.6745 * (x - median) / MAD
    """
    median = x.median()
    mad = (x - median).abs().median()

    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan , index=x.index)
    
    return 0.6745 * (x - median) / mad

def stl_residual_detector(
    df: pd.DataFrame,
    period: int = 365,
    threshold: float = 3.5
                          ) -> pd.DataFrame:
    """
    Per-meter STL decomposition; anomaly score on residuals via MAD z-score.

    threshold=3.5 is a common robust threshold (analogous to ~3 std dev).
    """

    if STL is None:
        raise ModuleNotFoundError(
            "statsmodels is required for STL residual detection. "
            "Install it with: pip install statsmodels"
        )

    df = df.sort_values(["meter_id","date"]).copy()
    
    df["residual"] = np.nan
    df["stl_anomaly_score"] = np.nan
    df["stl_predicted_anomaly"] = False

    for meter_id,mdf in df.groupby("meter_id"):
        mdf = mdf.sort_values("date").copy()

        y = mdf["consumption_kwh"].astype(float).values

        try:
            stl = STL(y,period=period , robust = True)
            res = stl.fit()
        except Exception as e:
            print(f"STL failed for {meter_id}: {e}")
            continue


        residual = pd.Series(res.resid,index= mdf.index)

        z = robust_zscore_mad(residual).abs()
        t = z.quantile(0.98)
        pred = (z > t).fillna(False)

        df.loc[mdf.index , "residual"] = residual.values
        df.loc[mdf.index , "stl_anomaly_score"] = z.values
        df.loc[mdf.index , "stl_predicted_anomaly"] = pred.values
    
    return df


