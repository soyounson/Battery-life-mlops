# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
import time, numpy as np
from databricks.sdk import WorkspaceClient
from src.battery.config import TABLE
from src.battery.features import FEATURES
w = WorkspaceClient()

# COMMAND ----------

X = spark.table(TABLE["gold_features"]).limit(3).toPandas()[FEATURES]
t0 = time.time()
resp = w.serving_endpoints.query(name="battery-life", dataframe_records=X.to_dict("records"))
print(f"{(time.time() - t0) * 1000:.0f} ms")
print("예측 수명:", (10 ** np.array(resp.predictions)).round())
