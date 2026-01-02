# Sentiment Analysis at Scale: Customer Feedback Streams

## Big Data Analytics Capstone Project

A Big Data analytics pipeline for real-time sentiment analysis of customer reviews at scale, processing 10M+ records using Apache Spark.

---

## 📋 Project Overview

### Objective
Design and implement a full Big Data Analytics pipeline that processes millions of customer reviews to perform sentiment analysis, incorporating batch and streaming data ingestion, machine learning models, and interactive visualizations.

### Key Features
-  Processes 10M+ customer reviews
-  Real-time streaming sentiment analysis
-  ML model accuracy 
-  Interactive dashboards and visualizations
-  Comprehensive performance benchmarking
-  Production-ready architecture

---

##  Project Requirements Met

| Requirement | Implementation | Status |
|------------|----------------|---------|
| 10M+ records | 10,000,000 reviews processed | ✅ |
| Streaming component | Spark Structured Streaming with 10s windows | ✅ |
| Data platform | Spark DataFrame API with caching | ✅ |
| ML model | Random Forest with 87.3% accuracy | ✅ |
| Hyperparameter tuning | TrainValidationSplit with param grid | ✅ |
| Visualization | Plotly interactive dashboards | ✅ |
| Performance analysis | Partitioning, caching, scalability tests | ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              DATA INGESTION LAYER                   │
│  • Batch: CSV → 10M records                        │
│  • Streaming: Simulated real-time feed             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│           APACHE SPARK PROCESSING                   │
│  • Data cleaning & validation                       │
│  • Feature engineering (25+ features)               │
│  • Partitioning (50 partitions)                     │
│  • Caching strategy (3.2x speedup)                  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│          MACHINE LEARNING PIPELINE                  │
│  • Text processing (Tokenization, TF-IDF)           │
│  • Models: Logistic Reg, Random Forest, Naive Bayes│
│  • Best Model: Random Forest (87.3% accuracy)       │
│  • Hyperparameter tuning with cross-validation      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│        VISUALIZATION & ANALYTICS                    │
│  • Interactive Plotly dashboards                    │
│  • KPIs: sentiment distribution, trends             │
│  • Model performance metrics                        │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
sentiment-analysis-bigdata/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # Jupyter/Colab notebooks
│   ├── 1_setup_ingestion.py          # Data loading & expansion
│   ├── 2_streaming_pipeline.py       # Real-time streaming
│   ├── 3_data_processing.py          # Cleaning & features
│   ├── 4_ml_pipeline.py              # ML models training
│   ├── 5_visualization.py            # Dashboards
│   └── 6_performance_analysis.py     # Benchmarking
│
├── data/                              # Data files
│   ├── reviews.csv                   # Original dataset (37K)
│   ├── reviews_expanded_10M.csv      # Expanded dataset (10M)
│   └── streaming_reviews/            # Streaming data directory
│
├── results/                           # Output files
│   ├── dashboard_metrics.csv         # KPI metrics
│   ├── model_performance.csv         # ML results
│   ├── performance_*.csv             # Benchmark results
│   └── visualizations/               # Saved charts
│
├── models/                            # Trained models
│   └── random_forest_model/          # Best model
│
├── docs/                              # Documentation
│   ├── Technical_Report.pdf          # 15-20 page report
│   └── Presentation_Slides.pdf       # 10-12 slides
│
└── demo/                              # Demo materials
    └── demo_script.md                # Demo walkthrough
```

---

##  Quick Start

### Prerequisites

- **Python**: 3.7 or higher
- **Java**: OpenJDK 8 (for Spark)
- **Google Colab Account**: Recommended for easy setup
- **Memory**: 4GB RAM minimum (8GB recommended)

### Installation

#### Option 1: Google Colab (Recommended)

1. Open Google Colab: https://colab.research.google.com
2. Upload `reviews.csv` to Colab
3. Run the following installation commands:

```python
# Install PySpark
!pip install pyspark

# Install visualization libraries
!pip install pandas numpy matplotlib seaborn plotly

# Install NLP libraries
!pip install textblob vaderSentiment

# Install Java for Spark
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
```

4. Run notebooks in sequence (1 → 6)

#### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/your-username/sentiment-analysis-bigdata.git
cd sentiment-analysis-bigdata

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

---

##  Running the Pipeline

### Step-by-Step Execution

#### Phase 1: Setup & Data Ingestion (10 minutes)
```python
# Run: 1_setup_ingestion.py
# - Loads original 37K reviews
# - Expands to 10M records
# - Fills missing values
# - Loads into Spark DataFrame
```

**Expected Output:**
- `reviews_expanded_10M.csv` (10M records)
- Spark DataFrame with 10M rows cached

#### Phase 2: Streaming Pipeline (5 minutes)
```python
# Run: 2_streaming_pipeline.py
# - Simulates real-time review stream
# - Processes 100 reviews every 3 seconds
# - Windowed aggregations
# - Real-time metrics
```

**Expected Output:**
- 2,000+ streaming reviews processed
- Real-time sentiment metrics
- Combined batch + streaming dataset

#### Phase 3: Data Processing (15 minutes)
```python
# Run: 3_data_processing.py
# - Data quality checks
# - Data cleaning
# - Feature engineering (25+ features)
# - Exploratory data analysis
```

**Expected Output:**
- Cleaned dataset with 99.9% quality
- 25+ engineered features
- EDA statistics and distributions

#### Phase 4: Machine Learning (20 minutes)
```python
# Run: 4_ml_pipeline.py
# - Text processing pipeline
# - Train 4 ML models
# - Hyperparameter tuning
# - Model evaluation
```

**Expected Output:**
- Random Forest model (87.3% accuracy)
- Model comparison results
- Feature importance analysis
- Saved model artifacts

#### Phase 5: Visualization (5 minutes)
```python
# Run: 5_visualization.py
# - KPI dashboard
# - Sentiment distribution charts
# - Temporal trends
# - Model performance visualizations
```

**Expected Output:**
- 6+ interactive Plotly charts
- Dashboard metrics CSV
- Model performance CSV

#### Phase 6: Performance Analysis (5 minutes)
```python
# Run: 6_performance_analysis.py
# - Partitioning benchmarks
# - Caching impact analysis
# - Scalability tests
# - Resource utilization
```

**Expected Output:**
- Performance benchmark CSVs
- Optimization recommendations
- Scalability metrics

### Total Runtime: ~60 minutes

---

##  Key Results

### Model Performance

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | 83.2% | 0.826 | 245s |
| **Random Forest** ⭐ | **87.3%** | **0.868** | 412s |
| Naive Bayes | 81.7% | 0.809 | 156s |
| Tuned Random Forest | 87.3% | 0.868 | 480s |

### Performance Optimizations

| Optimization | Improvement | Details |
|--------------|-------------|---------|
| Optimal Partitioning | +51% throughput | 50 partitions optimal |
| Caching Strategy | 3.2x speedup | For iterative operations |
| DataFrame API | 45% faster | vs RDD API |
| Scalability | Linear | 10K → 10M records |

### Business Metrics

- **Total Reviews**: 10,000,000
- **Average Rating**: 4.2 / 5.0
- **Positive Rate**: 70%
- **Processing Speed**: 7,000+ records/second
- **Real-time Latency**: <3 seconds

---

##  Documentation

### Technical Report
- **File**: `docs/Technical_Report.pdf`
- **Pages**: 20
- **Contents**:
  - Problem description
  - Dataset profile
  - Architecture design
  - Technology justification
  - ML implementation
  - Performance analysis
  - Lessons learned

### Presentation Slides
- **File**: `docs/Presentation_Slides.pdf`
- **Slides**: 12
- **Contents**:
  - Business context
  - Architecture
  - Data pipeline
  - ML results
  - Dashboards
  - Conclusion
  

## Contributions

Project performed and submitted by [Sirine Ben Mansour]

---

## 📧 Contact

For questions or issues:
- Email: [sirine.bnmnsr@gmail.com]
  
---

## 📜 License

This project is created for academic purposes as part of the Big Data Analytics course capstone project.
