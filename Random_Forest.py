import pandas as pd

x = pd.read_parquet("data/processed/meter_risk.parquet")

y = pd.read_parquet("data/processed/meter_labels.parquet")

df_ml = x.merge(
    y[["meter_id", "label"]],
    on="meter_id",
    how="inner",
    validate="one_to_one"
)

print(df_ml.shape)
print(df_ml["label"].value_counts())


FEATURES = [
    "num_incidents",
    "total_incident_days",
    "max_rolling_score",
    "max_stl_score",
    "roll_pct",
    "stl_pct",
    "final_risk_score"
]


x_data = df_ml[FEATURES]
y_data = df_ml["label"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x_train , x_test , y_train , y_test = train_test_split(x_data,y_data,test_size=.3 ,random_state=42 , stratify=y_data)


print("Train size:", x_train.shape)
print("Test size:", x_test.shape)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=5
)
clf.fit(x_train,y_train)


import numpy as np
from sklearn.metrics import roc_auc_score

proba = clf.predict_proba(x_test)[:, 1]

auc = roc_auc_score(y_test, proba)
print("ROC AUC:", auc)

def precision_at_k(y_true, scores, k):
    topk = np.argsort(scores)[-k:]
    return y_true.iloc[topk].mean()

for k in [10, 20, 30]:
    print(f"Precision@{k}:", precision_at_k(y_test, proba, k))

baseline_scores = x_test["final_risk_score"].values

for k in [10, 20, 30]:
    print(f"Baseline Precision@{k}:", precision_at_k(y_test, baseline_scores, k))




# ML is used as a re-ranking layer on top of explainable detectors, improving inspection efficiency without replacing domain logic.
