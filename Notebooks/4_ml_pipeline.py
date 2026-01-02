# Machine Learning Pipeline - Multi-Model Sentiment Classification

print("\n" + "="*80)
print("=== PHASE 4: MACHINE LEARNING PIPELINE ===")
print("="*80)

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline
import time
import builtins # Import builtins to access original Python functions

# ============================================================================
# DATA PREPARATION FOR ML
# ============================================================================

print("\n=== Preparing Data for ML ===")

# Sample for faster training (you can use full dataset if resources allow)
ml_sample_size = 1_000_000  # 1M records for training
current_df_count = processed_df.count()
# Calculate the fraction to sample, ensuring it doesn't exceed 1.0
fraction_to_sample = builtins.min(1.0, ml_sample_size / current_df_count)
ml_df = processed_df.sample(fraction=fraction_to_sample, seed=42)
ml_df = ml_df.cache()

print(f"✓ ML dataset size: {ml_df.count():,} records")

# Convert sentiment labels to numeric indices
label_indexer = StringIndexer(inputCol='sentiment_label', outputCol='label')
ml_df = label_indexer.fit(ml_df).transform(ml_df)

# Split data: 70% train, 15% validation, 15% test
train_df, val_df, test_df = ml_df.randomSplit([0.7, 0.15, 0.15], seed=42)

print(f"\nData splits:")
print(f"  Training:   {train_df.count():,} ({train_df.count()/ml_df.count()*100:.1f}%) {val_df.count()/ml_df.count()*100:.1f}%) ")
print(f"  Validation: {val_df.count():,} ({val_df.count()/ml_df.count()*100:.1f}%) ")
print(f"  Test:       {test_df.count():,} ({test_df.count()/ml_df.count()*100:.1f}%)")

# Cache splits
train_df.cache()
val_df.cache()
test_df.cache()

# ============================================================================
# TEXT PROCESSING PIPELINE
# ============================================================================

print("\n=== Building Text Processing Pipeline ===")

# Tokenization
tokenizer = Tokenizer(inputCol='review_text_eng', outputCol='words')

# Remove stop words
remover = StopWordsRemover(inputCol='words', outputCol='filtered_words')

# TF-IDF vectorization
cv = CountVectorizer(inputCol='filtered_words', outputCol='raw_features', vocabSize=10000)
idf = IDF(inputCol='raw_features', outputCol='tfidf_features')

# Assemble all features
assembler = VectorAssembler(
    inputCols=['tfidf_features', 'text_length', 'word_count', 'brand_index', 'review_type_index'],
    outputCol='features',
    handleInvalid='skip'
)

print("✓ Text processing pipeline created")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION (Baseline)
# ============================================================================

print("\n=== Model 1: Logistic Regression ===")
start_time = time.time()

# Build pipeline
lr = LogisticRegression(
    featuresCol='features',
    labelCol='label',
    maxIter=10,
    regParam=0.01
)

lr_pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, assembler, lr])

# Train model
print("Training Logistic Regression...")
lr_model = lr_pipeline.fit(train_df)
lr_train_time = time.time() - start_time

# Predictions
lr_train_pred = lr_model.transform(train_df)
lr_test_pred = lr_model.transform(test_df)

# Evaluation
evaluator_multi = MulticlassClassificationEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='accuracy'
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='f1'
)

lr_train_acc = evaluator_multi.evaluate(lr_train_pred)
lr_test_acc = evaluator_multi.evaluate(lr_test_pred)
lr_test_f1 = evaluator_f1.evaluate(lr_test_pred)

print(f"\n✓ Logistic Regression Results:")
print(f"  Training time: {lr_train_time:.2f}s")
print(f"  Training accuracy: {lr_train_acc:.4f}")
print(f"  Test accuracy: {lr_test_acc:.4f}")
print(f"  Test F1-score: {lr_test_f1:.4f}")

# ============================================================================
# MODEL 2: RANDOM FOREST (Ensemble Method)
# ============================================================================

print("\n=== Model 2: Random Forest Classifier ===")
start_time = time.time()

rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='label',
    numTrees=20,
    maxDepth=10,
    seed=42,
    maxBins=100 # Increased maxBins to handle higher cardinality categorical features
)

rf_pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, assembler, rf])

print("Training Random Forest...")
rf_model = rf_pipeline.fit(train_df)
rf_train_time = time.time() - start_time

# Predictions
rf_train_pred = rf_model.transform(train_df)
rf_test_pred = rf_model.transform(test_df)

# Evaluation
rf_train_acc = evaluator_multi.evaluate(rf_train_pred)
rf_test_acc = evaluator_multi.evaluate(rf_test_pred)
rf_test_f1 = evaluator_f1.evaluate(rf_test_pred)

print(f"\n✓ Random Forest Results:")
print(f"  Training time: {rf_train_time:.2f}s")
print(f"  Training accuracy: {rf_train_acc:.4f}")
print(f"  Test accuracy: {rf_test_acc:.4f}")
print(f"  Test F1-score: {rf_test_f1:.4f}")

# Feature importance
rf_classifier = rf_model.stages[-1]
feature_importance = rf_classifier.featureImportances
print(f"\n  Feature importances (top 5): {feature_importance.toArray()[:5]}")

# ============================================================================
# MODEL 3: NAIVE BAYES (Probabilistic)
# ============================================================================

print("\n=== Model 3: Naive Bayes Classifier ===")
start_time = time.time()

nb = NaiveBayes(
    featuresCol='features',
    labelCol='label',
    smoothing=1.0
)

nb_pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, assembler, nb])

print("Training Naive Bayes...")
nb_model = nb_pipeline.fit(train_df)
nb_train_time = time.time() - start_time

# Predictions
nb_test_pred = nb_model.transform(test_df)

# Evaluation
nb_test_acc = evaluator_multi.evaluate(nb_test_pred)
nb_test_f1 = evaluator_f1.evaluate(nb_test_pred)

print(f"\n✓ Naive Bayes Results:")
print(f"  Training time: {nb_train_time:.2f}s")
print(f"  Test accuracy: {nb_test_acc:.4f}")
print(f"  Test F1-score: {nb_test_f1:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING (Random Forest)
# ============================================================================

print("\n=== Hyperparameter Tuning (Random Forest) ==")
start_time = time.time()

# Create parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Cross-validator
cv_evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')

crossval = TrainValidationSplit(
    estimator=rf_pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=cv_evaluator,
    trainRatio=0.8,
    seed=42
)

# Use smaller sample for tuning
tuning_sample = train_df.sample(fraction=0.2, seed=42)
print(f"Tuning on {tuning_sample.count():,} samples...")

cv_model = crossval.fit(tuning_sample)
tuning_time = time.time() - start_time

# Best model predictions
best_model_pred = cv_model.transform(test_df)
best_model_acc = evaluator_multi.evaluate(best_model_pred)
best_model_f1 = evaluator_f1.evaluate(best_model_pred)

print(f"\n✓ Tuned Model Results:")
print(f"  Tuning time: {tuning_time:.2f}s")
print(f"  Best test accuracy: {best_model_acc:.4f}")
print(f"  Best test F1-score: {best_model_f1:.4f}")

# ============================================================================
# CLUSTERING ANALYSIS (Unsupervised)
# ============================================================================

print("\n=== Clustering Analysis (K-Means) ===")
start_time = time.time()

# Prepare features for clustering
clustering_pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])
clustering_features = clustering_pipeline.fit(train_df).transform(train_df)

# Assemble features
clustering_assembler = VectorAssembler(
    inputCols=['raw_features'],
    outputCol='features',
    handleInvalid='skip'
)
clustering_data = clustering_assembler.transform(clustering_features)

# Sample for clustering
clustering_sample = clustering_data.sample(fraction=0.1, seed=42).cache()
print(f"Clustering on {clustering_sample.count():,} samples...")

# K-Means clustering
kmeans = KMeans(k=3, seed=42, featuresCol='features')
kmeans_model = kmeans.fit(clustering_sample)
clustering_time = time.time() - start_time

# Predict clusters
clustered = kmeans_model.transform(clustering_sample)

# Analyze clusters
cluster_analysis = clustered.groupBy('prediction', 'sentiment_label') \
    .agg(count('*').alias('count')) \
    .orderBy('prediction', 'sentiment_label')

print(f"\n✓ Clustering Results:")
print(f"  Training time: {clustering_time:.2f}s")
print(f"  Silhouette score: {kmeans_model.summary.trainingCost:.4f}")
print("\nCluster distribution by sentiment:")
cluster_analysis.show()

# ============================================================================
# MODEL COMPARISON SUMMARY
# ============================================================================

print("\n=== MODEL COMPARISON SUMMARY ===")
print("="*80)

comparison_data = [
    ("Logistic Regression", lr_train_time, lr_test_acc, lr_test_f1),
    ("Random Forest", rf_train_time, rf_test_acc, rf_test_f1),
    ("Naive Bayes", nb_train_time, nb_test_acc, nb_test_f1),
    ("Tuned Random Forest", tuning_time, best_model_acc, best_model_f1)
]

print(f"\n{'Model':<25} {'Train Time':<15} {'Test Acc':<12} {'F1-Score':<12}")
print("-" * 64)
for model, train_time, acc, f1 in comparison_data:
    print(f"{model:<25} {train_time:>10.2f}s     {acc:>8.4f}     {f1:>8.4f}")


print("\n✓ Phase 4 Complete: Machine Learning Pipeline")
print("="*80)


import os
from pyspark.sql.functions import col

print("\n" + "="*80)
print("=== FULL DATASET INFERENCE ===")
print("="*80)

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PREDICTION_OUTPUT_PATH = "outputs/full_dataset_predictions"

# Create output directory if needed (local / compatible with Spark submit)
os.makedirs(PREDICTION_OUTPUT_PATH, exist_ok=True)

# ============================================================================
# RUN INFERENCE
# ============================================================================

print("\nRunning inference on full dataset...")

full_predictions = best_model.transform(processed_df)

prediction_df = full_predictions.select(
    "review_id",
    "brand",
    "stars",
    "sentiment_label",
    col("prediction").alias("predicted_label"),
    col("probability").cast(StringType()).alias("probability") # Convert vector to string
)

record_count = prediction_df.count()
print(f"✓ Inference complete on {record_count:,} records")

# ============================================================================
# SAVE AS CSV
# ============================================================================

print("\nSaving predictions as CSV...")

prediction_df \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(PREDICTION_OUTPUT_PATH)

print(f"✓ Predictions saved at: {PREDICTION_OUTPUT_PATH}")
print("="*80)
