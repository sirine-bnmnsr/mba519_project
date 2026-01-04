# =============================================================================
# PHASE 2: REAL-TIME STREAMING INGESTION - FULL 1M ROWS TO BIGQUERY
# =============================================================================

import os
import time
import threading
import shutil
import builtins
from datetime import datetime

import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import col, when, length

print("\n" + "=" * 80)
print("=== PHASE 2: STREAMING 1M REVIEWS TO BIGQUERY ===")
print("=" * 80)


# =============================================================================
# STREAMING SCHEMA
# =============================================================================
streaming_schema = StructType([
    StructField("brand", StringType(), True),
    StructField("review_type", StringType(), True),
    StructField("review_id", StringType(), True),
    StructField("review_ts", StringType(), True),
    StructField("stars", IntegerType(), True),
    StructField("review_text_eng", StringType(), True),
    StructField("review_title_eng", StringType(), True),
    StructField("processing_time", TimestampType(), True)
])


# =============================================================================
# DIRECTORIES
# =============================================================================
streaming_dir = "/content/streaming_reviews"
checkpoint_dir = "/content/streaming_checkpoint"

for d in [streaming_dir, checkpoint_dir]:
    os.makedirs(d, exist_ok=True)

# Always reset checkpoint for notebook runs
shutil.rmtree(checkpoint_dir, ignore_errors=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print("✓ Created streaming directories")


# =============================================================================
# PREPARE 1M ROWS
# =============================================================================
print("\n=== Preparing 1M rows for streaming ===")

expanded_pd = expanded_df.select(
    "brand",
    "review_type",
    "review_id",
    "review_ts",
    "stars",
    "review_text_eng",
    "review_title_eng"
).limit(1_000_000).toPandas()

print(f"✓ Converted {len(expanded_pd):,} rows to pandas")


# =============================================================================
# STREAMING DATA GENERATOR
# =============================================================================
def generate_streaming_reviews_full(batch_size=1000, interval=2):
    print("\n=== Starting Full Dataset Streaming ===")

    total_batches = (len(expanded_pd) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = builtins.min(start_idx + batch_size, len(expanded_pd))

        batch = expanded_pd.iloc[start_idx:end_idx].copy()
        now = datetime.now()

        batch["review_ts"] = now.strftime("%Y-%m-%d %H:%M:%S")
        batch["processing_time"] = now

        batch.to_json(
            f"{streaming_dir}/batch_{batch_num:05d}.json",
            orient="records",
            lines=True
        )

        if (batch_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"  {end_idx:,}/{len(expanded_pd):,} rows | "
                f"{end_idx / elapsed:.0f} rows/sec"
            )

        time.sleep(interval)

    print("✓ Streaming generator finished")


# =============================================================================
# START STREAMING GENERATOR THREAD
# =============================================================================
streaming_thread = threading.Thread(
    target=generate_streaming_reviews_full,
    daemon=True
)
streaming_thread.start()

time.sleep(5)


# =============================================================================
# SPARK STREAMING SETUP
# =============================================================================
print("\n=== Setting up Spark Structured Streaming ===")

streaming_df = spark.readStream \
    .schema(streaming_schema) \
    .option("maxFilesPerTrigger", 2) \
    .json(streaming_dir)

print("✓ Streaming DataFrame created")


# =============================================================================
# REAL-TIME TRANSFORMATIONS
# =============================================================================
streaming_processed = (
    streaming_df
    .withColumn("text_length", length(col("review_text_eng")))
    .withColumn(
        "sentiment_label",
        when(col("stars") >= 4, "Positive")
        .when(col("stars") <= 2, "Negative")
        .otherwise("Neutral")
    )
)


# =============================================================================
# FOREACH-BATCH BIGQUERY WRITER (CORRECT WAY)
# =============================================================================
def write_batch_to_bigquery(batch_df, batch_id):
    """
    Executed exactly once per micro-batch by Spark
    """
    if batch_df.count() == 0:
        return

    pdf = batch_df.toPandas()

    pandas_to_bq(
        pdf,
        table_name="phase2_streaming_reviews_full",
        if_exists="append"
    )

    print(f"✓ BQ batch {batch_id}: wrote {len(pdf):,} rows")


# =============================================================================
# START STREAMING QUERY
# =============================================================================
query = (
    streaming_processed.writeStream
    .foreachBatch(write_batch_to_bigquery)
    .outputMode("append")
    .option("checkpointLocation", checkpoint_dir)
    .start()
)

print("✓ Streaming query started")


# =============================================================================
# MONITOR PROGRESS
# =============================================================================
print("\n=== Monitoring Stream Progress ===")

while streaming_thread.is_alive():
    time.sleep(15)
    print("  Streaming still running...")

streaming_thread.join()
query.awaitTermination(timeout=60)


# =============================================================================
# COMBINE BATCH + STREAMING DATA
# =============================================================================
print("\n=== Combining Batch and Streaming Data ===")

combined_df = spark_df.unionByName(
    final_streaming_df.select(spark_df.columns)
)

combined_df.cache()



print(f"✓ Combined dataset: {combined_df.count():,} total reviews")
print(f"  Batch data: {spark_df.count():,}")
print(f"  Streaming data: {streaming_count:,}")

print("\n✓ Phase 2 Complete: Streaming Integration")
print("=" * 80)


print("\n Full Dataset Streaming to BigQuery")
print("=" * 80)
