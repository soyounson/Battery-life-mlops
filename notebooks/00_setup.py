# Databricks notebook source
# MAGIC %pip install xgboost lightgbm shap optuna

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys, os; sys.path.append(os.path.abspath(".."))
from src.battery.config import CATALOG, SCHEMA, VOLUME
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

import mlflow, xgboost, lightgbm
print(mlflow.__version__, xgboost.__version__, lightgbm.__version__)
