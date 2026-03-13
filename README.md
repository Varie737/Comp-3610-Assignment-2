# COMP 3610 — Assignment 2: ML Model Training & Evaluation (NYC Yellow Taxi Tips)

## Project Overview
This project builds, evaluates, and interprets machine learning models to predict **NYC Yellow Taxi** tipping behavior using the cleaned dataset from **Assignment 1**. The primary task is **regression** (predict `tip_amount`), with an additional **classification** task (predict `high_tip`, defined as `tip_amount > 0.20 × fare_amount`). Since `tip_amount` is only reliably recorded for credit card trips, the dataset is filtered to **`payment_type = 1`** prior to modeling. :contentReference[oaicite:0]{index=0}

## Repository Contents
Required repository contents for submission include:
- `assignment2.ipynb` — Jupyter notebook containing Part 1, Part 2, and Part 3
- `README.md` — setup + run instructions (this file)
- `requirements.txt` — pinned Python package versions
- `.gitignore` — excludes data files and model artifacts (no datasets or saved models committed) :contentReference[oaicite:1]{index=1}

### Expected local files (example)
- `yellow_taxi_cleaned.parquet` — cleaned dataset from Assignment 1  
- `taxi_zone_lookup.csv` — zone lookup table (borough mapping)

## Methods Summary (Notebook)
The notebook is structured as a complete analytical report with clear markdown headings and explanatory text. 

### Part 1 — Data Preprocessing & Feature Engineering
- Temporal features: `pickup_hour`, `pickup_day_of_week` (0=Monday), `is_weekend`
- Trip features: `trip_duration_minutes`, `trip_speed_mph`, `log_trip_distance`
- Fare features: `fare_per_mile`, `fare_per_minute` (with division-by-zero handling)
- Zone features: pickup/dropoff borough encoding using the zone lookup table 
- Targets: `tip_amount` (regression), `high_tip` (classification) 
- Split: 70/15/15 (train/val/test) with stratification for `high_tip` 
- Scaling: StandardScaler/MinMaxScaler fit on training only; applied to val/test

### Part 2 — Model Training & Tuning
Baseline Scikit-learn models:
- Regression: Linear Regression, Random Forest Regressor
- Classification: Logistic Regression, Random Forest Classifier  

Hyperparameter tuning:
- RandomizedSearchCV or GridSearchCV (≥3 hyperparameters)
- 5-fold cross-validation
- A stratified subsample (recommended 200,000–500,000 rows) may be used to reduce runtime; sample size is documented in the notebook

Neural network:
- PyTorch feedforward network with ≥2 hidden layers
- DataLoader batching, appropriate loss (MSELoss or BCEWithLogitsLoss), optimizer (Adam/SGD), ≥20 epochs
- Training/validation loss curves plotted
- Same evaluation metrics reported as Scikit-learn models  

### Part 3 — Model Evaluation & Interpretation
- Comprehensive test evaluation with summary table
- ROC curves for classification models + confusion matrix for best classifier
- Predicted vs actual plot + residual analysis for best regressor
- Feature importance (RF) + coefficients interpretation (Linear/Logistic)
- Optional SHAP explanations for 3 sample trips 
## Setup Instructions

### 1) Create and activate a virtual environment 
**Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2) Install dependencies 
pip install -r requirements.txt

### 3) Run the notebook 

