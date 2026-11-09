from pyspark.sql import SparkSession

def write_by_batch(df, table, batch):
    """같은 batch를 다시 실행해도 중복되지 않게 그 batch만 교체 (멱등성)"""
    spark = SparkSession.getActiveSession()
    if spark.catalog.tableExists(table):
        (df.write.mode("overwrite")
           .option("replaceWhere", f"batch = '{batch}'").saveAsTable(table))
    else:
        df.write.saveAsTable(table)
