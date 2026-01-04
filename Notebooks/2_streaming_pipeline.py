# =============================================================================
# PHASE 2: REAL-TIME STREAMING INGESTION
# Simulated Kafka-like streaming using file-based Spark Structured Streaming
# =============================================================================

import os
import time
import threading
import random
from datetime import datetime

import numpy as np
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import *

print("\n" + "=" * 80)
print("=== PHASE 2: REAL-TIME STREAMING INGESTION ===")
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
output_dir = "/content/streaming_output"

for directory in [streaming_dir, checkpoint_dir, output_dir]:
    os.makedirs(directory, exist_ok=True)

print("✓ Created streaming directories")


# =============================================================================
# CONVERT SPARK → PANDAS ONCE (CRITICAL FIX)
# =============================================================================
# Sample a smaller subset of the expanded_df to avoid out-of-memory errors
# when converting to Pandas, as the original expanded_df has 10 million rows.
expanded_pd = expanded_df.sample(False, 0.01, seed=42).select(
    "brand",
    "review_type",
    "review_id",
    "review_ts",
    "stars",
    "review_text_eng",
    "review_title_eng"
).toPandas()

print(f"✓ Converted expanded_df to pandas: {len(expanded_pd):,} rows")


# =============================================================================
# STREAMING DATA GENERATOR (PANDAS-BASED)
# =============================================================================
def generate_streaming_reviews(batch_size=100, interval=3, duration=60):
    """
    Simulate real-time review stream by writing JSON micro-batches
    """
    print(f"\n=== Starting Streaming Data Generator ===")
    print(f"  Batch size: {batch_size} reviews")
    print(f"  Interval: {interval} seconds")
    print(f"  Duration: {duration} seconds")

    start_time = time.time()
    batch_num = 0

    while time.time() - start_time < duration:
        batch = expanded_pd.sample(
            n=batch_size,
            replace=True
        ).copy()

        now = datetime.now()
        batch["review_ts"] = now.strftime("%Y-%m-%d %H:%M:%S")
        batch["processing_time"] = now

        # Simulate noisy star ratings (10%)
        mask = np.random.rand(len(batch)) < 0.1
        batch.loc[mask, "stars"] = np.random.randint(1, 6, size=mask.sum())

        batch_file = f"{streaming_dir}/batch_{batch_num:05d}.json"
        batch.to_json(batch_file, orient="records", lines=True)

        batch_num += 1
        print(f"  Batch {batch_num}: {batch_size} reviews written at {now.strftime('%H:%M:%S')}")

        time.sleep(interval)

    print(f"\n✓ Streaming simulation complete: {batch_num} batches generated")


# =============================================================================
# START STREAMING GENERATOR (BACKGROUND THREAD)
# =============================================================================
streaming_thread = threading.Thread(
    target=generate_streaming_reviews,
    args=(100, 3, 60),
    daemon=True
)
streaming_thread.start()

time.sleep(5)


# =============================================================================
# SPARK STRUCTURED STREAMING SETUP
# =============================================================================
print("\n=== Setting up Spark Structured Streaming ===")

streaming_df = spark.readStream \
    .schema(streaming_schema) \
    .option("maxFilesPerTrigger", 1) \
    .json(streaming_dir)

print("✓ Streaming DataFrame created")


# =============================================================================
# REAL-TIME TRANSFORMATIONS
# =============================================================================
streaming_processed = streaming_df \
    .withColumn("text_length", length(col("review_text_eng"))) \
    .withColumn(
        "sentiment_label",
        when(col("stars") >= 4, "Positive")\
        .when(col("stars") <= 2, "Negative")\
        .otherwise("Neutral")
    ) \
    .withWatermark("processing_time", "1 minute")


# =============================================================================
# REAL-TIME AGGREGATIONS
# =============================================================================
streaming_metrics = streaming_processed \
    .groupBy(
        window(col("processing_time"), "10 seconds"),
        col("sentiment_label")
    ) \
    .agg(
        count("*").alias("review_count"),
        avg("stars").alias("avg_stars"),
        avg("text_length").alias("avg_text_length")
    )


# =============================================================================
# STREAM OUTPUTS (MEMORY SINK)
# =============================================================================
query_reviews = streaming_processed.writeStream \
    .outputMode("append") \
    .format("memory") \
    .queryName("streaming_reviews") \
    .start()

query_metrics = streaming_metrics.writeStream \
    .outputMode("complete") \
    .format("memory") \
    .queryName("streaming_metrics") \
    .start()

print("✓ Streaming queries started")


# =============================================================================
# MONITOR STREAMING
# =============================================================================
print("\n=== Monitoring Real-Time Stream ===")

for i in range(6):
    time.sleep(5)

    current_count = spark.sql(
        "SELECT COUNT(*) AS count FROM streaming_reviews"
    ).collect()[0]["count"]

    print(f"\n[{i * 5}s] Processed reviews: {current_count:,}")

    if current_count > 0:
        spark.sql("""
            SELECT sentiment_label,
                   COUNT(*) AS count,
                   ROUND(AVG(stars), 2) AS avg_stars
            FROM streaming_reviews
            GROUP BY sentiment_label
            ORDER BY sentiment_label
        """).show()


# =============================================================================
# STOP STREAMING
# =============================================================================
print("\n=== Stopping Streaming Queries ===")
query_reviews.stop()
query_metrics.stop()


# =============================================================================
# FINAL STREAMING RESULTS
# =============================================================================
final_streaming_df = spark.sql("SELECT * FROM streaming_reviews")
final_streaming_df.cache()

streaming_count = final_streaming_df.count()

print(f"\n✓ Streaming Phase Complete")
print(f"  Total records processed: {streaming_count:,}")
print(f"  Processing rate: {streaming_count / 60:.1f} reviews/second")

final_streaming_pd = final_streaming_df.toPandas()

pandas_to_bq(
    final_streaming_pd,
    table_name="phase2_streaming_reviews",
    if_exists="replace"
)

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
print("=" * 80)"""
