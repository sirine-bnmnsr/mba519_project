# Data Processing, Cleaning & Feature Engineering Pipeline

print("\n" + "="*80)
print("=== PHASE 3: DATA PROCESSING & FEATURE ENGINEERING ===")
print("="*80)

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

print("\n=== Running Data Quality Checks ===")

# Check 1: Missing critical fields
print("\n1. Missing Critical Fields:")
critical_fields = ['brand', 'stars', 'review_text_eng']
for field in critical_fields:
    null_count = combined_df.filter(col(field).isNull()).count()
    null_pct = (null_count / combined_df.count()) * 100
    print(f"  {field}: {null_count:,} nulls ({null_pct:.2f}%)")

# Check 2: Invalid star ratings
print("\n2. Invalid Star Ratings:")
invalid_stars = combined_df.filter((col('stars') < 1) | (col('stars') > 5)).count()
print(f"  Invalid ratings: {invalid_stars:,}")

# Check 3: Empty text reviews
print("\n3. Empty Review Text:")
empty_text = combined_df.filter((col('review_text_eng').isNull()) | (col('review_text_eng') == '')).count()
print(f"  Empty reviews: {empty_text:,}")

# Check 4: Duplicate review IDs
print("\n4. Duplicate Review IDs:")
total_reviews = combined_df.count()
unique_ids = combined_df.select('review_id').distinct().count()
duplicates = total_reviews - unique_ids
print(f"  Duplicates: {duplicates:,}")

# ============================================================================
# DATA CLEANING
# ============================================================================

print("\n=== Data Cleaning Pipeline ===")

# Remove records with missing critical fields
cleaned_df = combined_df.filter(
    col('review_text_eng').isNotNull() &
    (col('review_text_eng') != '') &
    col('stars').isNotNull() &
    col('brand').isNotNull()
)

# Filter valid star ratings
cleaned_df = cleaned_df.filter((col('stars') >= 1) & (col('stars') <= 5))

# Remove duplicates based on review_id
cleaned_df = cleaned_df.dropDuplicates(['review_id'])

print(f"✓ Cleaned data: {cleaned_df.count():,} records")
print(f"  Removed: {combined_df.count() - cleaned_df.count():,} invalid records")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n=== Feature Engineering ===")

# 1. Text-based features
print("\n1. Creating text-based features...")
cleaned_df = cleaned_df.withColumn('text_length', length(col('review_text_eng')))
cleaned_df = cleaned_df.withColumn('word_count', size(split(col('review_text_eng'), ' ')))
cleaned_df = cleaned_df.withColumn('has_title', when(col('review_title_eng').isNotNull(), 1).otherwise(0))

# 2. Sentiment labels (target variable)
print("2. Creating sentiment labels...")
cleaned_df = cleaned_df.withColumn('sentiment_label',
    when(col('stars') >= 4, 'Positive')
    .when(col('stars') <= 2, 'Negative')
    .otherwise('Neutral')
)

# Binary sentiment for some models
cleaned_df = cleaned_df.withColumn('sentiment_binary',
    when(col('stars') >= 4, 1).otherwise(0)
)

# 3. Temporal features
print("3. Creating temporal features...")
cleaned_df = cleaned_df.withColumn('review_date', to_date(col('review_ts')))
cleaned_df = cleaned_df.withColumn('review_year', year(col('review_date')))
cleaned_df = cleaned_df.withColumn('review_month', month(col('review_date')))
cleaned_df = cleaned_df.withColumn('review_quarter', quarter(col('review_date')))
cleaned_df = cleaned_df.withColumn('review_day_of_week', dayofweek(col('review_date')))

# 4. Brand encoding
print("4. Encoding categorical features...")
brand_indexer = StringIndexer(inputCol='brand', outputCol='brand_index')
cleaned_df = brand_indexer.fit(cleaned_df).transform(cleaned_df)

# 5. Review type encoding
review_type_indexer = StringIndexer(inputCol='review_type', outputCol='review_type_index')
cleaned_df = review_type_indexer.fit(cleaned_df).transform(cleaned_df)

# Cache the processed data
cleaned_df.cache()
print(f"\n✓ Feature engineering complete")
print(f"  Total features: {len(cleaned_df.columns)}")

# Show sample with new features
print("\n=== Sample Processed Data ===")
cleaned_df.select(
    'brand', 'stars', 'sentiment_label', 'text_length', 
    'word_count', 'review_month', 'brand_index'
).show(10)

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n=== Exploratory Data Analysis ===")

# 1. Sentiment distribution
print("\n1. Sentiment Distribution:")
sentiment_dist = cleaned_df.groupBy('sentiment_label') \
    .agg(count('*').alias('count')) \
    .withColumn('percentage', (col('count') / cleaned_df.count() * 100)) \
    .orderBy('sentiment_label')
sentiment_dist.show()

# 2. Brand distribution
print("\n2. Top 10 Brands by Review Count:")
brand_dist = cleaned_df.groupBy('brand') \
    .agg(
        count('*').alias('review_count'),
        avg('stars').alias('avg_rating')
    ) \
    .orderBy(desc('review_count')) \
    .limit(10)
brand_dist.show()

# 3. Temporal trends
print("\n3. Reviews by Month:")
monthly_dist = cleaned_df.groupBy('review_year', 'review_month') \
    .agg(
        count('*').alias('count'),
        avg('stars').alias('avg_stars')
    ) \
    .orderBy('review_year', 'review_month')
monthly_dist.show(12)

# 4. Text statistics
print("\n4. Text Length Statistics:")
cleaned_df.select('text_length', 'word_count').summary().show()

# 5. Star rating distribution
print("\n5. Star Rating Distribution:")
stars_dist = cleaned_df.groupBy('stars') \
    .agg(count('*').alias('count')) \
    .withColumn('percentage', (col('count') / cleaned_df.count() * 100)) \
    .orderBy('stars')
stars_dist.show()

# ============================================================================
# PARTITIONING STRATEGY
# ============================================================================

print("\n=== Implementing Partitioning Strategy ===")

# Repartition by sentiment for balanced processing
print("\nRepartitioning by sentiment label...")
partitioned_df = cleaned_df.repartition(50, 'sentiment_label')
print(f"  New partition count: {partitioned_df.rdd.getNumPartitions()}")

# Test partition distribution
partition_dist = partitioned_df.groupBy('sentiment_label') \
    .agg(count('*').alias('count')) \
    .orderBy('sentiment_label')
print("\nRecords per sentiment:")
partition_dist.show()

# Cache partitioned data
partitioned_df.cache()
partitioned_df.count()  # Materialize cache

print("\n✓ Phase 3 Complete: Data Processing & Feature Engineering")
print("="*80)

# Store processed data reference
processed_df = partitioned_df
