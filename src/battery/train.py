import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from mlflow.models import infer_signature
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .features import FEATURES
from .config import EXPERIMENT_NAME

# 모델 zoo: 이름만 바꾸면 같은 파이프라인에서 새 실험이 된다
MODELS = {
    "dummy": lambda p: DummyRegressor(**p),
    "elasticnet": lambda p: ElasticNet(max_iter=10000, **p),
    "random_forest": lambda p: RandomForestRegressor(random_state=42, **p),
    "xgboost": lambda p: XGBRegressor(random_state=42, **p),
    "lightgbm": lambda p: LGBMRegressor(random_state=42, verbose=-1, **p),
}
NEEDS_SCALING = {"dummy", "elasticnet"}

def setup_mlflow(spark):
    user = spark.sql("SELECT current_user()").first()[0]
    mlflow.set_registry_uri("databricks-uc")
    exp = mlflow.set_experiment(f"/Users/{user}/{EXPERIMENT_NAME}")
    return exp.experiment_id

def build_pipeline(name, params):
    """전처리 + 모델을 한 덩어리로: 학습과 추론에서 똑같이 적용된다"""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if name in NEEDS_SCALING:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODELS[name](params)))
    return Pipeline(steps)

def cycle_metrics(y_log_true, y_log_pred, prefix):
    """log10 수명 → 사이클 단위로 되돌려 평가"""
    t, p = 10 ** np.asarray(y_log_true), 10 ** np.asarray(y_log_pred)
    return {f"{prefix}_mae_cycles": mean_absolute_error(t, p),
            f"{prefix}_rmse_cycles": root_mean_squared_error(t, p),
            f"{prefix}_mape_pct": float(np.mean(np.abs(p - t) / t) * 100)}

def cv_metrics(name, params, X, y, n_splits=5):
    cv = KFold(n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(build_pipeline(name, params), X, y, cv=cv)
    return cycle_metrics(y, pred, "cv")

def train_and_log(name, params, train_df, test_df, data_version=None):
    X_tr, y_tr = train_df[FEATURES], np.log10(train_df.cycle_life)
    X_te, y_te = test_df[FEATURES], np.log10(test_df.cycle_life)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": name, "n_train_cells": len(X_tr), **params})
        if data_version is not None:
            mlflow.log_param("gold_table_version", data_version)
        metrics = cv_metrics(name, params, X_tr, y_tr)        # 모델 선택은 CV로
        pipe = build_pipeline(name, params).fit(X_tr, y_tr)
        pred_te = pipe.predict(X_te)
        metrics.update(cycle_metrics(y_te, pred_te, "test"))  # test는 마지막 확인용
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(10 ** y_te, 10 ** pred_te)
        lim = [0, max(10 ** y_te.max(), 10 ** pred_te.max()) * 1.05]
        ax.plot(lim, lim, "r--"); ax.set(xlabel="actual", ylabel="predicted", title=name)
        mlflow.log_figure(fig, "pred_vs_actual.png"); plt.close(fig)

        info = mlflow.sklearn.log_model(
            pipe, name="model",                    # MLflow 2.x: artifact_path="model"
            signature=infer_signature(X_tr, pipe.predict(X_tr)),
            input_example=X_tr.head(3),
            serialization_format="cloudpickle")
        mlflow.set_tags({"model_uri": info.model_uri, "stage": "candidate"})
        print(name, {k: round(v, 1) for k, v in metrics.items()})
        return info, metrics
