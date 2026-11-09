# Early Prediction of Battery Cycle Life: End-to-End ML Pipeline

> 작업 중. 튜토리얼 Chapter 16의 템플릿으로 채워나가기.

## Problem
## Data
Severson et al. (2019), Toyota Research Institute (data.matr.io). 원본 데이터는 저장소에 포함하지 않음.
## Architecture
## Results
| Model | CV MAE | CV MAPE | Test MAE |
|---|---|---|---|
## Deployment
## Monitoring
## Key decisions
## Limitations & next steps
## How to run
1. `python local/convert_mat_to_parquet.py <batch.mat> b1`
2. parquet를 `/Volumes/workspace/battery/raw`에 업로드
3. `notebooks/00_setup` → `01_bronze` → ... → `09_anomaly` 순서로 실행
4. 테스트: `pytest -q`
