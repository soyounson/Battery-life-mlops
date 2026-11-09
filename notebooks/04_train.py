# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
import numpy as np, mlflow
from sklearn.model_selection import train_test_split
from src.battery.config import TABLE
from src.battery.features import FEATURES
from src.battery.train import setup_mlflow, train_and_log, build_pipeline

setup_mlflow(spark)
data_version = spark.sql(f"DESCRIBE HISTORY {TABLE['gold_features']} LIMIT 1").first()["version"]
df = spark.table(TABLE["gold_features"]).filter("batch = 'b1'").toPandas()

# 행 하나 = 셀 하나이므로 행 분할 = 셀 단위 분할. test는 고정하고 끝까지 건드리지 않기
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_tr, y_tr = train_df[FEATURES], np.log10(train_df.cycle_life)
X_te = test_df[FEATURES]

RUNS = [
    ("dummy", {}),                                              # 바닥선: 평균만 예측
    ("elasticnet", {"alpha": 0.01, "l1_ratio": 0.5}),
    ("xgboost", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}),
    ("lightgbm", {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
                  "min_child_samples": 5}),
    ("random_forest", {"n_estimators": 300, "min_samples_leaf": 2}),
]
for name, params in RUNS[:2]:          # Chapter 7: 처음 2개. Chapter 8에서 [:5]로
    train_and_log(name, params, train_df, test_df, data_version)

# COMMAND ----------

# Chapter 8 (선택): Optuna 튜닝
import optuna, mlflow
from src.battery.train import cv_metrics

def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 100, 800),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
              "max_depth": trial.suggest_int("max_depth", 2, 6)}
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        m = cv_metrics("xgboost", params, X_tr, y_tr)        # test는 보지 않는다
        mlflow.log_params(params); mlflow.log_metrics(m)
    return m["cv_mae_cycles"]

with mlflow.start_run(run_name="xgboost_optuna"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_cv_mae_cycles", study.best_value)

# 최적 파라미터로 '후보' run을 하나 더 만든다 (등록 대상)
train_and_log("xgboost", study.best_params, train_df, test_df, data_version)

# COMMAND ----------

# Chapter 9: 해석과 오차 분석
import shap, matplotlib.pyplot as plt
pipe = build_pipeline("xgboost", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}).fit(X_tr, y_tr)
X_t = pipe[:-1].transform(X_te)
sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(X_t)
shap.summary_plot(sv, X_t, feature_names=FEATURES, show=False)
with mlflow.start_run(run_name="xgboost_shap"):
    mlflow.log_figure(plt.gcf(), "shap_summary.png")

err = test_df.assign(pred=10 ** pipe.predict(X_te))
err["abs_err"] = (err.pred - err.cycle_life).abs()
display(err.sort_values("abs_err", ascending=False)[["cell_id", "policy", "cycle_life", "pred", "abs_err"]].head(3))
