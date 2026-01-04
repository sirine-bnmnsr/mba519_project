# Performance Analysis & Optimization

print("\n" + "="*80)
print("=== PHASE 6: PERFORMANCE OPTIMIZATION & BENCHMARKING ===")
print("="*80)

import time
import pandas as pd
from pyspark.sql.functions import *
import builtins # Import builtins to access original Python functions

# ============================================================================
# BENCHMARK 1: PARTITIONING IMPACT
# ============================================================================

print("\n=== Benchmark 1: Impact of Partitioning ===")

# Test different partition strategies
partition_configs = [
    ("No Repartition", None),
    ("10 Partitions", 10),
    ("50 Partitions", 50),
    ("100 Partitions", 100),
    ("200 Partitions", 200)
]

benchmark_results = []

# Sample data for testing
test_df = processed_df.sample(fraction=0.1, seed=42)
test_count = test_df.count()
print(f"Testing on {test_count:,} records")

for config_name, num_partitions in partition_configs:
    print(f"\nTesting: {config_name}")

    # Apply partitioning
    if num_partitions:
        test_data = test_df.repartition(num_partitions)
    else:
        test_data = test_df

    # Benchmark: Aggregation operation
    start_time = time.time()
    result = test_data.groupBy('sentiment_label', 'brand') \
        .agg(
            count('*').alias('count'),
            avg('stars').alias('avg_stars'),
            avg('text_length').alias('avg_length')
        ).collect()
    execution_time = time.time() - start_time

    actual_partitions = test_data.rdd.getNumPartitions()

    benchmark_results.append({
        'Configuration': config_name,
        'Partitions': actual_partitions,
        'Execution_Time': execution_time,
        'Records_Processed': test_count
    })

    print(f"  Partitions: {actual_partitions}")
    print(f"  Execution time: {execution_time:.3f}s")
    print(f"  Throughput: {test_count/execution_time:,.0f} records/sec")

# Display results
print("\n" + "="*70)
print("Partitioning Benchmark Results:")
print("="*70)
benchmark_df = pd.DataFrame(benchmark_results)
print(benchmark_df.to_string(index=False))

# ============================================================================
# BENCHMARK 2: CACHING STRATEGY
# ============================================================================

print("\n\n=== Benchmark 2: Caching Strategy Impact ===")

cache_results = []

# Test without cache
print("\nTest 1: Without caching")
test_data = processed_df.sample(fraction=0.05, seed=42)

# Multiple operations without cache
start_time = time.time()
count1 = test_data.count()
agg1 = test_data.groupBy('sentiment_label').count().collect()
filter1 = test_data.filter(col('stars') >= 4).count()
no_cache_time = time.time() - start_time

print(f"  Total time (3 operations): {no_cache_time:.3f}s")

# Test with cache
print("\nTest 2: With caching")
test_data_cached = test_data.cache()
test_data_cached.count()  # Materialize cache

start_time = time.time()
count2 = test_data_cached.count()
agg2 = test_data_cached.groupBy('sentiment_label').count().collect()
filter2 = test_data_cached.filter(col('stars') >= 4).count()
cache_time = time.time() - start_time

print(f"  Total time (3 operations): {cache_time:.3f}s")
print(f"  Speedup: {no_cache_time/cache_time:.2f}x")

cache_results.append({
    'Strategy': 'Without Cache',
    'Time': no_cache_time,
    'Speedup': 1.0
})
cache_results.append({
    'Strategy': 'With Cache',
    'Time': cache_time,
    'Speedup': no_cache_time/cache_time
})

test_data_cached.unpersist()

# ============================================================================
# BENCHMARK 3: DIFFERENT PROCESSING STRATEGIES
# ============================================================================

print("\n\n=== Benchmark 3: Processing Strategy Comparison ===")

strategy_results = []
sample_data = processed_df.sample(fraction=0.05, seed=42)

# Strategy 1: RDD-based processing
print("\nStrategy 1: RDD-based approach")
start_time = time.time()
rdd_result = sample_data.rdd \
    .map(lambda row: (row['sentiment_label'], (row['stars'], 1))) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
    .map(lambda x: (x[0], x[1][0] / x[1][1])) \
    .collect()
rdd_time = time.time() - start_time
print(f"  Execution time: {rdd_time:.3f}s")

# Strategy 2: DataFrame API
print("\nStrategy 2: DataFrame API approach")
start_time = time.time()
df_result = sample_data.groupBy('sentiment_label') \
    .agg(avg('stars').alias('avg_stars')) \
    .collect()
df_time = time.time() - start_time
print(f"  Execution time: {df_time:.3f}s")

# Strategy 3: Spark SQL
print("\nStrategy 3: Spark SQL approach")
sample_data.createOrReplaceTempView("reviews_temp")
start_time = time.time()
sql_result = spark.sql("""
    SELECT sentiment_label, AVG(stars) as avg_stars
    FROM reviews_temp
    GROUP BY sentiment_label
""").collect()
sql_time = time.time() - start_time
print(f"  Execution time: {sql_time:.3f}s")

strategy_results = pd.DataFrame({
    'Strategy': ['RDD API', 'DataFrame API', 'Spark SQL'],
    'Execution_Time': [rdd_time, df_time, sql_time],
    'Relative_Performance': [
        rdd_time / builtins.min(rdd_time, df_time, sql_time),
        df_time / builtins.min(rdd_time, df_time, sql_time),
        sql_time / builtins.min(rdd_time, df_time, sql_time)
    ]
})

print("\n" + "="*60)
print("Processing Strategy Comparison:")
print("="*60)
print(strategy_results.to_string(index=False))

# ============================================================================
# BENCHMARK 4: SCALABILITY TEST
# ============================================================================

print("\n\n=== Benchmark 4: Scalability Analysis ===")

scalability_results = []
data_sizes = [10000, 50000, 100000, 500000, 1000000]

for size in data_sizes:
    print(f"\nProcessing {size:,} records...")

    # Sample data
    scale_data = processed_df.sample(fraction=builtins.min(1.0, size/processed_df.count()), seed=42)
    actual_size = scale_data.count()

    # Benchmark operation
    start_time = time.time()
    result = scale_data.groupBy('sentiment_label', 'brand') \
        .agg(
            count('*').alias('count'),
            avg('stars').alias('avg_stars')
        ).count()
    execution_time = time.time() - start_time

    throughput = actual_size / execution_time

    scalability_results.append({
        'Records': actual_size,
        'Time': execution_time,
        'Throughput': throughput
    })

    print(f"  Actual records: {actual_size:,}")
    print(f"  Execution time: {execution_time:.3f}s")
    print(f"  Throughput: {throughput:,.0f} records/sec")

scalability_df = pd.DataFrame(scalability_results)

print("\n" + "="*60)
print("Scalability Test Results:")
print("="*60)
print(scalability_df.to_string(index=False))

# ============================================================================
# BENCHMARK 5: RESOURCE UTILIZATION
# ============================================================================

print("\n\n=== Benchmark 5: Resource Utilization Analysis ===")

# Get Spark configuration
spark_conf = spark.sparkContext.getConf().getAll()
print("\nCurrent Spark Configuration:")
important_configs = ['spark.driver.memory', 'spark.executor.memory',
                     'spark.sql.shuffle.partitions', 'spark.default.parallelism']
for key, value in spark_conf:
    if any(config in key for config in important_configs):
        print(f"  {key}: {value}")

# Analyze task distribution
print("\nTask Distribution Analysis:")
test_data = processed_df.sample(fraction=0.1, seed=42).repartition(20)

# Count records per partition
partition_counts = test_data.rdd.mapPartitions(lambda it: [builtins.sum(1 for _ in it)]).collect()
print(f"  Total partitions: {len(partition_counts)}")
print(f"  Min records per partition: {builtins.min(partition_counts):,}")
print(f"  Max records per partition: {builtins.max(partition_counts):,}")
print(f"  Avg records per partition: {builtins.sum(partition_counts)/len(partition_counts):,.0f}")
print(f"  Partition imbalance ratio: {builtins.max(partition_counts)/builtins.min(partition_counts):.2f}x")

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

print("\n\n" + "="*80)
print("=== PERFORMANCE OPTIMIZATION SUMMARY ===")
print("="*80)

print("\n1. PARTITIONING RECOMMENDATIONS:")
optimal_partitions = benchmark_df.loc[benchmark_df['Execution_Time'].idxmin()]
print(f"   ✓ Optimal partition count: {optimal_partitions['Partitions']}")
print(f"   ✓ Best execution time: {optimal_partitions['Execution_Time']:.3f}s")
print(f"   ✓ Throughput: {optimal_partitions['Records_Processed']/optimal_partitions['Execution_Time']:,.0f} records/sec")

print("\n2. CACHING IMPACT:")
speedup = cache_results[1]['Speedup']
print(f"   ✓ Performance improvement: {speedup:.2f}x faster with caching")
print(f"   ✓ Recommendation: Cache frequently accessed DataFrames")

print("\n3. PROCESSING STRATEGY:")
best_strategy = strategy_results.loc[strategy_results['Execution_Time'].idxmin(), 'Strategy']
print(f"   ✓ Fastest approach: {best_strategy}")
print(f"   ✓ Recommendation: Use DataFrame API for best Catalyst optimization")

print("\n4. SCALABILITY:")
avg_throughput = scalability_df['Throughput'].mean()
print(f"   ✓ Average throughput: {avg_throughput:,.0f} records/second")
print(f"   ✓ Linear scalability: {'Yes' if scalability_df['Throughput'].std() / avg_throughput < 0.3 else 'Needs optimization'}")

print("\n5. RESOURCE OPTIMIZATION:")
print(f"   ✓ Partition balance: {builtins.max(partition_counts)/builtins.min(partition_counts):.2f}x imbalance")
print(f"   ✓ Recommendation: {'Good balance' if builtins.max(partition_counts)/builtins.min(partition_counts) < 2 else 'Consider repartitioning'}")

# Save performance results
print("\n=== Saving Performance Results ===")

benchmark_df.to_csv('performance_partitioning.csv', index=False)
pd.DataFrame(cache_results).to_csv('performance_caching.csv', index=False)
strategy_results.to_csv('performance_strategies.csv', index=False)
scalability_df.to_csv('performance_scalability.csv', index=False)

print("✓ Saved performance_partitioning.csv")
print("✓ Saved performance_caching.csv")
print("✓ Saved performance_strategies.csv")
print("✓ Saved performance_scalability.csv")



pandas_to_bq(
    benchmark_df,
    "phase5_partition_benchmark",
    if_exists="replace"
)


pandas_to_bq(
    pd.DataFrame(cache_results),
    "phase5_caching_benchmark",
    if_exists="replace"
)


pandas_to_bq(
    strategy_results,
    "phase5_processing_strategies",
    if_exists="replace"
)


pandas_to_bq(
    scalability_df,
    "phase5_scalability",
    if_exists="replace"
)


print("\n✓ Phase 6 Complete: Performance Optimization & Benchmarking")
print("="*80)
