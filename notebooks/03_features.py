# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
from pyspark.sql import functions as F
from src.battery.config import TABLE, EARLY_CYCLE
from src.battery.io import write_by_batch
from src.battery.features import cell_features

dbutils.widgets.text("batch", "b1")
BATCH = dbutils.widgets.get("batch")

# COMMAND ----------

summary = spark.table(TABLE["silver_summary"]).filter(F.col("batch") == BATCH).toPandas()
qdlin = (spark.table(TABLE["silver_qdlin"]).filter(F.col("batch") == BATCH)
              .filter(F.col("cycle").isin(10, EARLY_CYCLE)).toPandas())

feat = cell_features(summary, qdlin, early=EARLY_CYCLE)
labels = summary.groupby("cell_id").agg(cycle_life=("cycle_life", "first"),
                                        policy=("policy", "first")).reset_index()
gold = feat.merge(labels, on="cell_id")
gold["batch"] = BATCH
write_by_batch(spark.createDataFrame(gold), TABLE["gold_features"], BATCH)
display(gold.describe())
