# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
import mlflow, pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F
from src.battery.config import TABLE, MODEL_NAME
from src.battery.features import FEATURES
from src.battery.io import write_by_batch

dbutils.widgets.text("batch", "b2")
BATCH = dbutils.widgets.get("batch")

mlflow.set_registry_uri("databricks-uc")
version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

df = spark.table(TABLE["gold_features"]).filter(F.col("batch") == BATCH).toPandas()
out = df[["cell_id", "batch"]].copy()
out["pred_cycle_life"] = 10 ** model.predict(df[FEATURES])
out["model_version"] = str(version)                   # 어떤 모델이 만든 예측인지 기록
out["predicted_at"] = pd.Timestamp.now(tz="UTC")

write_by_batch(spark.createDataFrame(out), TABLE["predictions"], BATCH)
print(f"{BATCH}: {len(out)}개 셀 예측, model v{version}")
