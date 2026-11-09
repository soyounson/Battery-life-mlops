# Databricks notebook source
import sys, os; sys.path.append(os.path.abspath(".."))
from pyspark.sql import functions as F
from src.battery.config import TABLE, EARLY_CYCLE
from src.battery.io import write_by_batch

dbutils.widgets.text("batch", "b1")
BATCH = dbutils.widgets.get("batch")

# COMMAND ----------

s = spark.table(TABLE["bronze_summary"]).filter(F.col("batch") == BATCH)

s_clean = (s.filter(F.col("cycle_life").isNotNull() & ~F.isnan("cycle_life"))
            .filter((F.col("qd_max") > 0.5) & (F.col("qd_max") < 1.2))   # 측정 오류 제거 (EDA로 근거 확인)
            .dropDuplicates(["cell_id", "cycle"]))

ok_cells = (s_clean.groupBy("cell_id").agg(F.max("cycle").alias("max_cycle"))
                   .filter(F.col("max_cycle") >= EARLY_CYCLE).select("cell_id"))
s_clean = s_clean.join(ok_cells, "cell_id")

n_before = s.select("cell_id").distinct().count()
n_after = s_clean.select("cell_id").distinct().count()
print(f"셀 {n_before} -> {n_after}")
assert n_after >= 0.7 * n_before, "30% 넘는 셀이 제외됨: 원본 확인 필요"

# COMMAND ----------

q = (spark.table(TABLE["bronze_qdlin"]).filter(F.col("batch") == BATCH)
          .join(ok_cells, "cell_id").filter(F.col("cycle") <= EARLY_CYCLE))
write_by_batch(s_clean.drop("_source_file"), TABLE["silver_summary"], BATCH)
write_by_batch(q.drop("_source_file"), TABLE["silver_qdlin"], BATCH)

# COMMAND ----------

import matplotlib.pyplot as plt
pdf = spark.table(TABLE["silver_summary"]).filter(F.col("batch") == BATCH).toPandas()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for cid, g in pdf.groupby("cell_id"):
    ax[0].plot(g.cycle, g.qd_max, lw=0.7)
ax[0].set(xlabel="cycle", ylabel="discharge capacity (Ah)")
pdf.groupby("cell_id").cycle_life.first().hist(ax=ax[1], bins=20)
ax[1].set(xlabel="cycle life")
plt.show()
