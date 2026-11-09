# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
import mlflow
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from src.battery.config import TABLE
from src.battery.features import FEATURES
from src.battery.train import setup_mlflow
setup_mlflow(spark)

# COMMAND ----------

df = spark.table(TABLE["gold_features"]).toPandas()
iso = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                ("iso", IsolationForest(contamination=0.05, random_state=42))])
with mlflow.start_run(run_name="isolation_forest"):
    iso.fit(df[FEATURES])
    df["anomaly_score"] = -iso.score_samples(df[FEATURES])   # 클수록 이상
    rho, p = spearmanr(df.anomaly_score, df.cycle_life)
    mlflow.log_params({"contamination": 0.05})
    mlflow.log_metrics({"spearman_score_vs_life": rho, "p_value": p})
display(df.sort_values("anomaly_score", ascending=False)[["cell_id", "policy", "cycle_life", "anomaly_score"]].head(5))
