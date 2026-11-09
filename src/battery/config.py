CATALOG, SCHEMA, VOLUME = "workspace", "battery", "raw"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
EARLY_CYCLE = 100                      # 이 사이클까지의 정보만 feature에 사용
TABLES = ["bronze_summary", "bronze_qdlin", "silver_summary", "silver_qdlin",
          "gold_features", "predictions"]
TABLE = {t: f"{CATALOG}.{SCHEMA}.{t}" for t in TABLES}
MODEL_NAME = f"{CATALOG}.{SCHEMA}.battery_life_model"
EXPERIMENT_NAME = "battery_life"       # /Users/<내 이메일>/battery_life 에 생성됨
