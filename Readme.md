# Review Mirror: Sentiment Drift Detection System

A deep learning-based system for detecting and analyzing sentiment drift in user reviews over time. This project implements a self-supervised contrastive learning approach to identify users and products whose opinion patterns have shifted significantly.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Methodology](#methodology)
- [Results & Visualization](#results--visualization)
- [Output Files](#output-files)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

## 🎯 Overview

Review Mirror analyzes temporal patterns in user reviews to detect **sentiment drift** - significant changes in user opinions over time. The system processes large-scale review data (e.g., Yelp reviews), extracts semantic embeddings using DistilBERT, and applies machine learning techniques to identify users whose review patterns have evolved.

### Key Applications:
- **User Behavior Analysis**: Identify users with changing preferences
- **Product Quality Monitoring**: Detect products receiving increasingly negative/positive reviews
- **Fraud Detection**: Flag suspicious review pattern changes
- **Customer Insight**: Understand long-term customer satisfaction trends

## ✨ Features

- **GPU-Accelerated Embedding Generation**: Efficient batch processing using DistilBERT
- **Multi-Method Drift Detection**:
  - Baseline methods (rating slope, semantic similarity)
  - Self-supervised contrastive learning (SSCD)
  - Combined logistic regression detector
- **Comprehensive Visualizations**: Timeline plots, PCA trajectories, ROC/PR curves
- **Dual-Level Analysis**: Both user-level and product-level drift detection
- **Changepoint Detection**: Identifies sudden shifts in sentiment using ruptures
- **Scalable Architecture**: Memory-efficient chunked processing for large datasets

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for embeddings generation)
- ~10GB RAM minimum
- ~5GB disk space for models and outputs

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Review-Mirror
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn matplotlib tqdm ruptures
```

### Step 3: Download Dataset
Place your `yelp_academic_dataset_review.json` file in the project root directory.

**Dataset Source**: [Yelp Open Dataset](https://www.yelp.com/dataset)

## 📊 Dataset

The system is designed for the **Yelp Academic Dataset** but can be adapted for any review dataset with:
- User ID
- Timestamps
- Text reviews
- Ratings (1-5 stars)

### Data Requirements:
- Minimum 30 reviews per user (configurable)
- JSON Lines format
- Fields: `user_id`, `stars`, `text`, `date`

## 📁 Project Structure

```
Review-Mirror/
│
├── main.ipynb              # Main Jupyter notebook with full pipeline
├── Readme.md               # This file
├── main.tex                # LaTeX documentation
│
├── outputs/                # Generated output files (created automatically)
│   ├── sscd_proj_head.pth              # Trained projection head model
│   ├── drift_detector_lr.pkl            # Logistic regression drift detector
│   ├── eval_user_results.pkl            # User-level evaluation results
│   ├── eval_product_results.pkl         # Product-level evaluation results
│   └── user_drift_summary.csv           # Per-user drift statistics
│
└── yelp_academic_dataset_review.json   # Input dataset (not included)
```

## 🚀 Usage Guide

### Quick Start

1. **Open the Notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Configure Parameters** (Cell 3)
   ```python
   DATA_PATH = "yelp_academic_dataset_review.json"
   MIN_REVIEWS_PER_USER = 30
   MAX_REVIEWS_PER_USER = 200
   ```

3. **Run All Cells** in sequence (Runtime: ~30-60 minutes depending on hardware)

### Step-by-Step Walkthrough

#### Phase 1: Data Preprocessing (Cells 1-5)
- **Purpose**: Load and clean review data
- **What happens**:
  - Filters users with minimum review count
  - Cleans text (removes URLs, special characters)
  - Creates normalized ratings (-1 to +1 scale)
  - Computes temporal features

**Expected Output**: 
```
✅ Retained XXX,XXX reviews across X,XXX active users.
```

#### Phase 2: Embedding Generation (Cells 6-10)
- **Purpose**: Convert review text to semantic vectors
- **What happens**:
  - Loads DistilBERT model
  - Processes text in GPU-accelerated batches
  - Generates 768-dimensional embeddings
  - Normalizes vectors

**Expected Output**:
```
✅ Done! Computed XXXX embeddings on device cuda:0 for 1000 users.
```

**⚠️ Note**: This is the most time-consuming step (~20-40 minutes on GPU)

#### Phase 3: Drift Analysis (Cells 11-14)
- **Purpose**: Compute drift metrics per user
- **What happens**:
  - Calculates rating slope (linear regression)
  - Detects changepoints using Pelt algorithm
  - Measures semantic drift (first vs last embedding)
  - Combines metrics into drift score

**Expected Output**:
```
✅ Drift computed for XXX users
```

#### Phase 4: Visualization (Cells 15-18)
- **Purpose**: Visualize drift patterns
- **Features**:
  - Timeline plots with changepoints
  - 3-panel dashboard (sentiment, similarity, PCA trajectory)
  - Top-K drifting users display

**Example Visualization**:
- **Panel 1**: Rating timeline with red dashed lines showing changepoints
- **Panel 2**: Similarity to first review over time
- **Panel 3**: 2D PCA embedding trajectory with arrows

#### Phase 5: Self-Supervised Learning (Cells 19-27)
- **Purpose**: Train contrastive learning model
- **What happens**:
  - Creates anchor-positive pairs from temporal sequences
  - Trains projection head with NT-Xent loss
  - Learns to distinguish drifted vs stable users
  - Saves model to `outputs/sscd_proj_head.pth`

**Expected Output**:
```
[sscd][epoch 8/8] avg loss 0.XXXX
[sscd] training complete and saved
```

#### Phase 6: Detector Training (Cells 28-31)
- **Purpose**: Build comprehensive drift classifier
- **Features**:
  - Combines 14 hand-crafted + learned features
  - Trains logistic regression detector
  - Generates synthetic drifted users for training

**Feature Categories**:
1. Learned features (proj_dist, proj_cos)
2. Baseline features (rating change, embedding similarity)
3. Advanced features (changepoints, tfidf shift, burstiness)

#### Phase 7: Evaluation (Cells 32-33)
- **Purpose**: Test detection performance
- **Metrics**:
  - **AUC-ROC**: Area under ROC curve
  - **Average Precision**: Summary of precision-recall curve
  - **P@10/P@20**: Precision at top 10/20 ranked users

**Expected Performance** (typical results):
```
detector_lr: AUC=0.85+, AP=0.80+, P@10=0.90+
```

#### Phase 8: Results Visualization (Cells 34-37)
- **Final outputs**:
  - ROC curves comparing all methods
  - Precision-Recall curves
  - Bar charts of metrics
  - Scatter plots of detector vs baseline scores

## 🔬 Methodology

### 1. Embedding Extraction
- **Model**: DistilBERT-base-uncased (66M parameters)
- **Method**: Mean pooling of final hidden states
- **Optimization**: GPU batch processing, normalized L2 vectors

### 2. Drift Metrics

#### Sentiment Slope
```python
slope = LinearRegression().fit(time, normalized_ratings).coef_[0]
```

#### Semantic Drift
```python
drift = 1 - cosine_similarity(embedding_first, embedding_last)
```

#### Changepoint Detection
Uses **Pelt algorithm** with RBF kernel to detect abrupt rating changes

#### Combined Drift Score
```python
drift_score = 0.6 × |total_rating_change| + 0.4 × semantic_drift
```

### 3. Contrastive Learning
- **Architecture**: 2-layer MLP projection head (768→128→64)
- **Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Training**: Anchor-positive pairs from consecutive reviews
- **Supervision**: Pseudo-labels from high/low drift users

### 4. Detection Pipeline
1. Extract 14 features per user/product
2. Apply trained logistic regression classifier
3. Rank by drift probability score
4. Evaluate top-K precision

## 📈 Results & Visualization

### Key Outputs

1. **User Drift Summary CSV**
   - Columns: user_id, n_reviews, slope, semantic_drift, changepoints
   - Use for further analysis or filtering

2. **Interactive Dashboards**
   - Top-5 drifting users shown automatically
   - Customizable: change `TOP_K = 5` parameter

3. **Model Performance**
   - Comparison table across all methods
   - ROC/PR curves saved as images

### Interpretation Guide

**High Drift Score (>0.7)**: User has significantly changed opinion
- Check for product quality changes
- Potential reviewer fatigue
- May indicate fraud if sudden

**Low Drift Score (<0.3)**: Stable, consistent reviewer
- Reliable for product benchmarking
- Good for training sentiment models

**Changepoints**: Red lines in timeline plots
- Indicate when user opinion shifted
- Correlate with product changes or events

## 📤 Output Files

All outputs saved to `outputs/` directory:

| File | Description | Size |
|------|-------------|------|
| `sscd_proj_head.pth` | Trained contrastive model | ~500KB |
| `drift_detector_lr.pkl` | Logistic regression detector + features | ~50KB |
| `user_drift_summary.csv` | Per-user drift statistics | ~1-5MB |
| `eval_user_results.pkl` | User-level evaluation metrics | ~100KB |
| `eval_product_results.pkl` | Product-level evaluation metrics | ~100KB |

## ⚙️ Technical Details

### Configuration Parameters

```python
# Data filtering
MIN_REVIEWS_PER_USER = 30        # Minimum reviews to include user
MAX_REVIEWS_PER_USER = 200       # Cap per user for efficiency

# Embedding
BATCH_SIZE = 16                  # Increase for faster GPU
MAX_LENGTH = 128                 # Token limit per review

# Training
EPOCHS = 8                       # Contrastive learning epochs
LR = 1e-3                       # Learning rate
PROJ_DIM = 64                   # Projection head output dimension

# Evaluation
TEST_FRACTION = 0.2             # 20% held out for testing
SYN_STRENGTHS = ['weak','medium','strong']  # Synthetic drift levels
```

### Memory Optimization

**For large datasets**:
1. Reduce `CHUNK_SIZE` in preprocessing (default: 100K)
2. Sample fewer users: modify `sample_users` selection
3. Use smaller embeddings: change to `distilbert-base-uncased` (already minimal)

**GPU Memory Issues**:
```python
BATCH_SIZE = 8  # Reduce from 16
torch.cuda.empty_cache()  # Add after embedding generation
```

## 📦 Requirements

### Python Packages
```
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
ruptures>=1.1.0
tqdm>=4.65.0
```

### Hardware
- **Minimum**: 8GB RAM, CPU-only (slow)
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, RTX 3080+ GPU

### Platform
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (Apple Silicon supported with MPS backend)
