import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import expr

print("=== Sentiment Analysis at Scale: Customer Feedback Pipeline ===")
print("Initializing Spark Session with optimized configuration...")

# Initialize Spark Session with performance optimizations
spark = SparkSession.builder \
    .appName("SentimentAnalysisAtScale") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"✓ Spark {spark.version} initialized successfully")
print(f"✓ Available cores: {spark.sparkContext.defaultParallelism}")

# ============================================================================
# PHASE 1: DATA INGESTION & EXPANSION (SPARK-NATIVE)
# ============================================================================

print("\n=== Loading original reviews.csv into Spark ===")

spark_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("reviews_37k_eng.csv")

spark_df = spark_df.filter(
    expr("try_cast(review_ts as date) IS NOT NULL OR review_ts IS NULL")
)


base_count = spark_df.count()
target_rows = 10_000_000
multiplier = (target_rows // base_count) + 1

print(f"✓ Loaded {base_count:,} original reviews")
print(f"✓ Expanding to {target_rows:,} records using Spark")

# Generate multiplier DataFrame
replication_df = spark.range(multiplier)

expanded_df = (
    spark_df
    .crossJoin(replication_df)
    .withColumn(
        "review_id",
        concat_ws(
            "-",
            lit("rev"),
            col("id"),
            monotonically_increasing_id()
        )
    )
    .withColumn(
        "review_ts_clean",
        to_date(col("review_ts"))
    )
    .withColumn(
        "review_ts",
        date_add(
            coalesce(col("review_ts_clean"), current_date()),
            - (rand() * 730).cast("int")
        )
    )
    .withColumn(
        "stars",
        when(
            rand() < 0.2,
            greatest(
                lit(1),
                least(
                    lit(5),
                    col("stars") + when(rand() < 0.5, -1).otherwise(1)
                )
            )
        ).otherwise(col("stars"))
    )
    .drop("review_ts_clean", "id")
    .limit(target_rows)
)

print("✓ Dataset expanded")


# NULL SANITIZATION (FIX BLANKS BEFORE TEXT GENERATION)

expanded_df = expanded_df \
    .withColumn(
        "stars",
        when(col("stars").isNull(), lit(3)).otherwise(col("stars"))
    ) \
    .withColumn(
        "review_type",
        when(col("review_type").isNull(), lit("product")).otherwise(col("review_type"))
    )

# INTELLIGENT TEXT FILLING
expanded_df = expanded_df \
    .withColumn(
        "review_text_eng",
        when(col("review_text_eng").isNull() | (col("review_text_eng") == ""),
            when(col("stars") >= 4, lit("Excellent product, very satisfied."))
            .when(col("stars") == 3, lit("Average product, acceptable quality."))
            .otherwise(lit("Disappointed with product quality."))
        ).otherwise(col("review_text_eng"))
    ) \
    .withColumn(
        "review_title_eng",
        when(col("review_title_eng").isNull() | (col("review_title_eng") == ""),
            when(col("stars") >= 4, lit("Great purchase"))
            .when(col("stars") == 3, lit("Okay"))
            .otherwise(lit("Not recommended"))
        ).otherwise(col("review_title_eng"))
    )

# INTELLIGENT TEXT FILLING (SPARK)

expanded_df = expanded_df \
    .withColumn(
        "stars",
        when(col("stars").isNull(), lit(3)).otherwise(col("stars"))
    ) \
    .withColumn(
        "review_type",
        when(col("review_type").isNull(), lit("product"))
        .otherwise(col("review_type"))
    ) \
    .withColumn(
        "review_text_eng",
        when(col("review_text_eng").isNull() | (col("review_text_eng") == ""),
            when(col("stars") >= 4, lit("Excellent product, very satisfied."))
            .when(col("stars") == 3, lit("Average product, acceptable quality."))
            .otherwise(lit("Disappointed with product quality."))
        ).otherwise(col("review_text_eng"))
    ) \
    .withColumn(
        "review_title_eng",
        when(col("review_title_eng").isNull() | (col("review_title_eng") == ""),
            when(col("stars") >= 4, lit("Great purchase"))
            .when(col("stars") == 3, lit("Okay"))
            .otherwise(lit("Not recommended"))
        ).otherwise(col("review_title_eng"))
    )


# Repartition + cache
expanded_df = expanded_df.repartition(200).cache()
expanded_df.count()

print(f"✓ Final record count: {expanded_df.count():,}")
print(f"✓ Partitions: {expanded_df.rdd.getNumPartitions()}")
print("\n✓ Phase 1 Complete: Data Ingestion & Expansion")
print("=" * 80)


# Cache for performance
spark_df.cache()
print(f"✓ Data loaded and cached in Spark")
print(f"  Total records: {spark_df.count():,}")
print(f"  Partitions: {spark_df.rdd.getNumPartitions()}")

# Show sample
print("\n=== Sample Data ===")
spark_df.show(5, truncate=50)

# Data profiling
print("\n=== Data Profiling ===")
spark_df.printSchema()
print("\nSummary Statistics (raw):")
spark_df.select('stars').summary().show()

print("\nNull value counts (raw):")
null_counts = spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns])
null_counts.show()

# Data profiling
print("\n=== Data Profiling ===")
expanded_df.printSchema()
print("\nSummary Statistics (extanded data):")
expanded_df.select('stars').summary().show()

print("\nNull value counts (extanded data):")
null_counts = expanded_df.select([count(when(col(c).isNull(), c)).alias(c) for c in expanded_df.columns])
null_counts.show()


(
    expanded_df
    .repartition(20)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv("reviews_expanded_10M_20parts")
)



print("\n✓ Phase 1 Complete: Data Ingestion & Expansion")
print("=" * 80)
