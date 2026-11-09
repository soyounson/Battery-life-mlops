# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
import mlflow
from mlflow import MlflowClient
from src.battery.config import MODEL_NAME
from src.battery.train import setup_mlflow
exp_id = setup_mlflow(spark)
client = MlflowClient()

runs = mlflow.search_runs([exp_id],
                          filter_string="tags.stage = 'candidate'",
                          order_by=["metrics.cv_mae_cycles ASC"])   # CV 기준으로 선택
best = runs.iloc[0]
print("best:", best["tags.mlflow.runName"], round(best["metrics.cv_mae_cycles"], 1))
mv = mlflow.register_model(best["tags.model_uri"], MODEL_NAME)

def champion_metric(metric="cv_mae_cycles"):
    try:
        v = client.get_model_version_by_alias(MODEL_NAME, "champion")
        return mlflow.get_run(v.run_id).data.metrics[metric]
    except Exception:
        return None                                   # 아직 champion 없음

current, new = champion_metric(), best["metrics.cv_mae_cycles"]
# 주의: 같은 데이터, 같은 평가 방식으로 만든 점수끼리만 비교한다
if current is None or new < current * 0.95:           # 5% 이상 좋아야 교체
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print("새 champion: version", mv.version)
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    print("challenger로 보관: version", mv.version)
