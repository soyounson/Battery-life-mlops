# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
from pyspark.sql import functions as F
from src.battery.config import TABLE, VOLUME_PATH
from src.battery.io import write_by_batch

dbutils.widgets.text("batch", "b1")
BATCH = dbutils.widgets.get("batch")

# COMMAND ----------

for name in ["summary", "qdlin"]:
    df = (spark.read.parquet(f"{VOLUME_PATH}/{BATCH}_{name}.parquet")
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path")))
    write_by_batch(df, TABLE[f"bronze_{name}"], BATCH)
    print(name, df.count())
