# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
import mlflow, pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F
from src.battery.config import TABLE, MODEL_NAME
from src.battery.features import FEATURES
from src.battery.monitoring import psi
from src.battery.train import setup_mlflow

dbutils.widgets.text("batch", "b2")
BATCH = dbutils.widgets.get("batch")
setup_mlflow(spark)

ref = spark.table(TABLE["gold_features"]).filter("batch = 'b1'").toPandas()   # 학습 데이터
cur = spark.table(TABLE["gold_features"]).filter(F.col("batch") == BATCH).toPandas()

# 1) 입력 드리프트
drift = (pd.DataFrame({"feature": FEATURES,
                       "psi": [psi(ref[f], cur[f]) for f in FEATURES]})
           .sort_values("psi", ascending=False))

# 2) 실제 성능 (라벨이 도착했다고 가정)
pred = (spark.table(TABLE["predictions"]).filter(F.col("batch") == BATCH).toPandas()
          .merge(cur[["cell_id", "cycle_life"]], on="cell_id"))
err = (pred.pred_cycle_life - pred.cycle_life).abs()
live = {"live_mae_cycles": err.mean(),
        "live_mape_pct": (err / pred.cycle_life).mean() * 100,
        "max_psi": drift.psi.max()}

with mlflow.start_run(run_name=f"monitoring_{BATCH}"):
    mlflow.log_metrics(live)
    mlflow.log_table(drift, "drift.json")
display(drift)

# 3) 재학습 규칙
v = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
cv_mape = mlflow.get_run(v.run_id).data.metrics["cv_mape_pct"]
retrain = live["max_psi"] > 0.25 or live["live_mape_pct"] > 1.2 * cv_mape
print(live, "→ 재학습 필요" if retrain else "→ 유지")
