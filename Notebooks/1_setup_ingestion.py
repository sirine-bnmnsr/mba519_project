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
# PHASE 1: DATA INGESTION & EXPANSION
# ============================================================================

def expand_dataset_to_millions(original_df, target_rows=10_000_000):
    """
    Expand the dataset to 10M+ records by intelligent replication
    with variations to simulate real-world data at scale
    """
    print(f"\n=== Expanding dataset from {len(original_df)} to {target_rows:,} records ===")

    current_count = len(original_df)
    replication_factor = (target_rows // current_count) + 1

    expanded_data = []
    brands = original_df['brand'].unique().tolist()

    for iteration in range(replication_factor):
        for idx, row in original_df.iterrows():
            new_row = row.copy()

            # Add temporal variation
            base_date = pd.to_datetime(row['review_ts'])
            if pd.isna(base_date):
                # Assign a default valid date if original review_ts is invalid
                base_date = pd.to_datetime(datetime.now().date()) - timedelta(days=random.randint(0, 365))
            days_offset = random.randint(-730, 0)  # Last 2 years
            new_row['review_ts'] = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')

            # Generate new unique review_id
            new_row['review_id'] = f"rev-{iteration:04d}-{idx:06d}-{random.randint(1000,9999)}"

            # Add some variation to stars (±1 with 20% probability)
            if random.random() < 0.2 and not pd.isna(row['stars']):
                variation = random.choice([-1, 1])
                # Use built-in max and min for scalar operations
                new_row['stars'] = __builtins__.max(1, __builtins__.min(5, int(row['stars']) + variation))

            expanded_data.append(new_row)

            if len(expanded_data) >= target_rows:
                break

        if len(expanded_data) >= target_rows:
            break

        if iteration % 50 == 0:
            print(f"  Progress: {len(expanded_data):,} / {target_rows:,} records")

    expanded_df = pd.DataFrame(expanded_data[:target_rows])
    print(f"✓ Dataset expanded to {len(expanded_df):,} records")

    return expanded_df

def fill_missing_text_intelligent(df):
    """
    Intelligently fill missing review text based on stars rating
    """
    print("\n=== Filling missing review text ===")

    # Templates based on star ratings
    templates = {
        5: [
            "Excellent product! Highly recommend.",
            "Outstanding quality and service.",
            "Perfect! Exceeded expectations.",
            "Amazing product, will buy again!",
            "Best purchase ever, very satisfied."
        ],
        4: [
            "Good quality, happy with purchase.",
            "Nice product, meets expectations.",
            "Satisfied with the product quality.",
            "Good value, would recommend.",
            "Pretty good overall experience."
        ],
        3: [
            "It's okay, nothing special.",
            "Average product, decent quality.",
            "Meets basic expectations.",
            "Not bad, but could be better.",
            "Acceptable quality for the price."
        ],
        2: [
            "Disappointed with quality.",
            "Not what I expected.",
            "Below average, had issues.",
            "Product quality could be better.",
            "Not satisfied with purchase."
        ],
        1: [
            "Very poor quality, not recommended.",
            "Terrible experience, waste of money.",
            "Completely disappointed.",
            "Do not buy this product.",
            "Worst purchase, very unhappy."
        ]
    }

    filled_count = 0
    for idx, row in df.iterrows():
        if pd.isna(row['review_text_eng']) or row['review_text_eng'] == '':
            stars = int(row['stars']) if not pd.isna(row['stars']) else 3
            df.at[idx, 'review_text_eng'] = random.choice(templates[stars])
            filled_count += 1

        if pd.isna(row['review_title_eng']) or row['review_title_eng'] == '':
            stars = int(row['stars']) if not pd.isna(row['stars']) else 3
            if stars >= 4:
                df.at[idx, 'review_title_eng'] = random.choice(["Great!", "Excellent", "Love it", "Recommended"])
            elif stars == 3:
                df.at[idx, 'review_title_eng'] = random.choice(["Okay", "Average", "Decent", "Fair"])
            else:
                df.at[idx, 'review_title_eng'] = random.choice(["Disappointed", "Not good", "Poor", "Bad"])
            filled_count += 1

    print(f"✓ Filled {filled_count:,} missing text fields")
    return df

# Load original data
print("\n=== Loading original reviews.csv ===")
original_df = pd.read_csv('reviews_37k_eng.csv')
print(f"✓ Loaded {len(original_df):,} original reviews")
print(f"  Columns: {list(original_df.columns)}")

# Expand to 10M+ records
expanded_df = expand_dataset_to_millions(original_df, target_rows=10_000_000)

# Fill missing values
expanded_df = fill_missing_text_intelligent(expanded_df)

# Save expanded dataset
expanded_df.to_csv('reviews_expanded_10M.csv', index=False)
print(f"\n✓ Saved expanded dataset: reviews_expanded_10M.csv ({len(expanded_df):,} records)")

# Load into Spark
print("\n=== Loading data into Spark DataFrame ===")
spark_df = spark.createDataFrame(expanded_df)

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
print("\nSummary Statistics:")
spark_df.select('stars').summary().show()

print("\nNull value counts:")
null_counts = spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns])
null_counts.show()

print("\n✓ Phase 1 Complete: Data Ingestion & Expansion")
print("=" * 80)
