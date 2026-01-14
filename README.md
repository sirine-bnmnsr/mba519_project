# Sentiment Analysis at Scale: E-Commerce Customer Reviews

## Big Data Analytics Capstone Project

A complete big data analytics pipeline for sentiment analysis of customer reviews, processing 10 million records using Apache Spark, Google Cloud BigQuery, and machine learning.

---

## 📋 Project Overview

### Objective
Design and implement a full big data analytics pipeline that processes millions of e-commerce customer reviews to perform sentiment classification, incorporating batch and streaming data ingestion, distributed machine learning, cloud storage, and performance optimization.

### Key Features
- Processes 10 million customer reviews
- Real-time streaming data ingestion (1M records)
- ML model accuracy: 91.89% (Logistic Regression)
- Cloud-based data warehouse (Google BigQuery)
- Comprehensive performance benchmarking
- Production-ready architecture on Google Colab

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Batch Reviews   │         │ Streaming Reviews│         │
│  │   (CSV Files)    │         │  (JSON Batches)  │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│              SPARK PROCESSING LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data Ingestion & Validation                         │  │
│  │  • Schema validation  • Type casting                 │  │
│  │  • Null handling     • Duplicate removal            │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Feature Engineering Pipeline                        │  │
│  │  • Text tokenization    • Stop word removal          │  │
│  │  • TF-IDF vectorization • Temporal features          │  │
│  │  • Brand encoding       • Length calculations        │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Machine Learning Pipeline                           │  │
│  │  • Model training       • Cross-validation           │  │
│  │  • Hyperparameter tuning • Model evaluation          │  │
│  └──────────────────┬───────────────────────────────────┘  │
└────────────────────┼────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              STORAGE LAYER (BigQuery)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Raw Data    │  │  Processed   │  │ Predictions  │     │
│  │   Tables     │  │    Data      │  │   & Metrics  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           VISUALIZATION LAYER (Looker Studio)                │
│  • Sentiment dashboards  • Performance metrics              │
│  • Time series analysis  • Brand comparisons                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
sentiment-analysis-bigdata/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   └── sentiment_analysis_pipeline.ipynb   # Complete pipeline (all phases)
│
├── data/
│   └── reviews_eng.csv           # Original dataset (39,160 records)
│
├── results/                           # Output files
│   ├── phase3_sentiment_distribution.csv
│   ├── phase3_brand_distribution.csv
│   ├── phase3_monthly_trends.csv
│   ├── phase4_model_predictions.csv
│   ├── phase5_partition_benchmark.csv
│   └── phase5_scalability.csv
│
├── models/                            # Trained models
│   └── logistic_regression_model/    # Best model (91.89% accuracy)
│
└── docs/                              # Documentation
    ├── Technical_Report.pdf          # 20-page technical report
    └── Presentation_Slides.pdf       # 12-slide presentation
```

---

### Prerequisites

- **Python**: 3.7+
- **Google Colab Account**: Required for execution
- **Google Cloud Account**: For BigQuery storage
- **Memory**: 4GB RAM minimum

### Installation & Setup

#### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload the Jupyter notebook: `sentiment_analysis_pipeline.ipynb`
3. Upload the dataset: `reviews_37k_eng.csv`

#### Step 2: Install Dependencies

The notebook includes installation cells:

```python
# Install PySpark
!pip install pyspark

# Install Google Cloud libraries
!pip install google-cloud-bigquery pandas db-dtypes pandas-gbq
```

#### Step 3: Configure Google Cloud

```python
# Authenticate with Google Cloud
from google.colab import auth
auth.authenticate_user()

# Set your project ID
PROJECT_ID = 'your-gcp-project-id'
DATASET = 'outputs'
```

#### Step 4: Run the Pipeline

Execute all cells in the notebook sequentially. The pipeline runs in 6 phases:

---

## Pipeline Execution

### Phase 1: Data Ingestion & Expansion

**What it does:**
- Loads original 39,160 reviews from CSV
- Expands to 10,000,000 records using cross-join replication
- Applies temporal and rating variations
- Caches dataset in Spark memory

**Output:**
- 10,000,000 records in Spark DataFrame
- 200 partitions
- Data profiling statistics

**Key Metrics:**
- Null values handled: 100%
- Expansion multiplier: 255x
- Final partitions: 200

---

### Phase 2: Streaming Data Ingestion

**What it does:**
- Simulates real-time review ingestion
- Processes 1,000,000 records via file-based streaming
- Micro-batches of 2,000 records
- Direct write to BigQuery using `foreachBatch`

**Output:**
- 1M streaming records in BigQuery
- Real-time sentiment labeling
- Combined batch + streaming dataset (41,160 records)

**Key Metrics:**
- Processing rate: 497 records/second
- Micro-batch interval: 2 seconds
- Total batches: 500

---

### Phase 3: Data Processing & Feature Engineering 

**What it does:**
- Data quality assessment (missing values, duplicates, invalid ratings)
- Data cleaning pipeline (59.48% retention rate)
- Feature engineering (19 features total)
- Exploratory data analysis

**Output:**
- 23,292 cleaned records
- Text features: `text_length`, `word_count`, `has_title`
- Temporal features: `review_year`, `review_month`, `review_quarter`
- Categorical encoding: `brand_index`, `review_type_index`
- 6 CSV files with EDA results

**Key Metrics:**
- Data quality: 99.9%+ after cleaning
- Features engineered: 19
- Missing data removed: 17,868 records

---

### Phase 4: Machine Learning Pipeline

**What it does:**
- Text processing (tokenization, stop words, TF-IDF)
- Trains 3 classification models
- Hyperparameter tuning with cross-validation
- Full dataset inference (23,292 predictions)

**Models Trained:**
1. **Logistic Regression** (Best Model)
   - Accuracy: 91.89%
   - F1-Score: 0.9121
   - Training time: 25.52s

2. **Random Forest**
   - Accuracy: 91.38%
   - F1-Score: 0.8729
   - Training time: 65.94s

3. **Naive Bayes**
   - Accuracy: 90.05%
   - F1-Score: 0.9090
   - Training time: 6.45s

**Output:**
- Trained models saved
- Model comparison table in BigQuery
- Full dataset predictions
- Feature importance analysis

---

### Phase 5: Visualization 

**What it does:**
- Create interactive Looker Studio dashboards
- Connect directly to BigQuery tables
- Display KPIs, trends, and model performance

**Dashboards:**
1. Sentiment Overview (distribution, trends)
2. Brand Analytics (comparison, market share)
---

### Phase 6: Performance Optimization 

**What it does:**
- Benchmarks 5 partitioning strategies
- Tests caching impact (1.29x speedup)
- Compares 3 Spark APIs (RDD, DataFrame, SQL)
- Scalability analysis across dataset sizes

**Output:**
- 4 performance CSV files in BigQuery
- Optimization recommendations
- Resource utilization metrics

**Key Findings:**
- Optimal partitions: 50 (2,464 rec/s)
- Caching speedup: 1.29x
- Best API: DataFrame (18.6x faster than RDD)
- Average throughput: 43,820 rec/s

---

## Key Results

### Model Performance Summary

| Model | Train Time | Test Accuracy | F1-Score | Selected |
|-------|-----------|---------------|----------|----------|
| **Logistic Regression** | 25.52s | **91.89%** | **0.9121** | YES|
| Random Forest | 65.94s | 91.38% | 0.8729 | |
| Naive Bayes | 6.45s | 90.05% | 0.9090 | |
| Tuned Random Forest | 110.28s | 91.35% | 0.8722 | |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Reviews Processed | 10,000,000 |
| Streaming Records | 1,000,000 |
| Clean Training Data | 23,292 |
| Average Star Rating | 4.75 / 5.0 |
| Positive Sentiment | 77.7% |
| Neutral Sentiment | 14.0% |
| Negative Sentiment | 8.3% |

### Performance Benchmarks

| Optimization | Improvement | Details |
|--------------|-------------|---------|
| Optimal Partitioning | 2,464 rec/s | 50 partitions |
| Caching Strategy | 1.29x speedup | 3 operations |
| DataFrame API | 18.6x faster | vs RDD API |
| Scalability | 43,820 rec/s | Average throughput |

---

## BigQuery Tables

All results are stored in Google Cloud BigQuery:

| Table Name | Description | Records |
|-----------|-------------|---------|
| `spark_df` | Original dataset | 39,160 |
| `phase2_streaming_reviews_full` | Streaming data | 1,000,000 |
| `phase3_reviews_processed` | Cleaned & engineered | 23,292 |
| `phase3_sentiment_distribution` | Sentiment stats | 3 |
| `phase3_brand_distribution` | Brand metrics | 10 |
| `phase3_monthly_trends` | Time series | 82 |
| `phase4_model_predictions` | ML predictions | 23,292 |
| `phase5_partition_benchmark` | Performance tests | 5 |
| `phase5_caching_benchmark` | Caching results | 2 |
| `phase5_processing_strategies` | API comparison | 3 |
| `phase5_scalability` | Scalability tests | 5 |

---

## Technology Stack

### Processing & Computation
- **Apache Spark 4.0.1** - Distributed data processing
- **PySpark** - Python API for Spark
- **Google Colab** - Execution environment

### Storage & Data Warehouse
- **Google Cloud BigQuery** - Cloud lakehouse
- **CSV Files** - Raw data storage

### Machine Learning
- **Spark MLlib** - Distributed ML algorithms
- **scikit-learn metrics** - Model evaluation

### Visualization (Planned)
- **Google Looker Studio** - BI dashboards
- **Plotly** - Interactive charts (optional)

### Development Tools
- **Jupyter Notebooks** - Interactive development
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

---

## Dataset Information

### Source
- **Origin**: Company e-commerce reviews (anonymized)
- **Time Period**: 2013-2018
- **Format**: CSV

### Original Dataset
- **Records**: 39,160 reviews
- **Brands**: Anonymized (Brand BB, Brand HH, Brand NN)
- **Review Types**: Product and service reviews
- **Languages**: English translations

### Expanded Dataset
- **Records**: 10,000,000 reviews
- **Expansion Method**: Cross-join replication with variations
- **Purpose**: Meet 10M+ record requirement

### Schema

| Column | Type | Description |
|--------|------|-------------|
| brand | STRING | Anonymized brand identifier |
| review_type | STRING | Product or service |
| review_id | STRING | Unique review ID |
| review_ts | DATE | Review timestamp |
| stars | INTEGER | Rating (1-5) |
| review_text_eng | STRING | Review text in English |
| review_title_eng | STRING | Review title in English |

---

## Documentation

### Technical Report
- **File**: `docs/Technical_Report.pdf`
- **Pages**: 20
- **Contents**:
  - Problem description and business context
  - Dataset profile and quality analysis
  - Architecture design and justification
  - Data ingestion workflow (batch + streaming)
  - Feature engineering pipeline
  - Machine learning implementation
  - Performance optimization analysis
  - Results, discussion, and lessons learned

### Presentation Slides
- **File**: `docs/Presentation_Slides.pdf`
- **Slides**: 12
- **Contents**:
  - Business context and problem statement
  - System architecture overview
  - Data pipeline walkthrough
  - ML results and model comparison
  - Dashboard previews
  - Key findings and conclusion

---

## Business Impact

This sentiment analysis system enables:

### Operational Benefits
- Automated sentiment classification (replacing manual review)
- Real-time customer satisfaction monitoring
- Early detection of product quality issues
- Scalable processing of growing review volumes

### Strategic Insights
- Brand performance comparison across sentiment
- Customer feedback trends over time
- Competitive positioning analysis
- Data-driven product improvement decisions

---

## 🚧 Limitations & Future Work

### Current Limitations
- **Class Imbalance**: 77.7% positive reviews bias model
- **Single Node**: Colab provides limited resources (4GB RAM)
- **Simulated Streaming**: File-based vs. true Kafka/Kinesis
- **Language**: English-only sentiment analysis

### Future Enhancements

**Short-term:**
- Implement class balancing (SMOTE, class weights)
- Deploy Looker Studio dashboards
- Add aspect-based sentiment analysis
- Integrate true streaming (Apache Kafka)

**Long-term:**
- Deploy on multi-node cluster (Google Dataproc)
- Implement deep learning models (BERT, transformers)
- Add multilingual sentiment support
- Create automated model retraining pipeline
- Production deployment with monitoring

---

## Author

**Sirine Ben Mansour**
- **Program**: Masters in Business Analytics
- **Course**: MBA519: Big Data Analytics
- **Professor**: Dr. Manel Abdelkader
- **Email**: sirine.bnmnsr@gmail.com

---

## License

This project is created for academic purposes as part of the Big Data Analytics course capstone project at [University Name].

---

## Acknowledgments

- **Dr. Manel Abdelkader** for project guidance and requirements
- **Apache Spark Community** for excellent documentation
- **Google Cloud Platform** for free tier BigQuery access
- **Google Colab** for providing free computational resources

---

## Support

For questions or issues:
- **Email**: sirine.bnmnsr@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/your-username/sentiment-analysis-bigdata/issues)
