from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.models.stl_residual import stl_residual_detector
import pandas as pd

def precision_at_k(df, score_col, k):
    topk = df.sort_values(score_col, ascending=False).head(k)
    return topk["generated_anomaly"].mean()

path = r"D:\Coading\Time-Series Anomaly Detection System for Utility Consumption Data\data\Raw\consumption_daily.csv"
df = pd.read_csv(path, parse_dates=["date"])
df2 = stl_residual_detector(df)

print(df2)
y_true = df2["generated_anomaly"]            
y_pred = df2["stl_predicted_anomaly"]         

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print([tn, fp, fn, tp])
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("Precision:", precision)
print("Recall:", recall)

print("Precision@100 (STL):", precision_at_k(df2, "stl_anomaly_score", 100))
print("Precision@500 (STL):", precision_at_k(df2, "stl_anomaly_score", 500))
