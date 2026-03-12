### Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance
Course: DSAI3202 – Winter 2026
Dataset: NASA Turbofan Engine Degradation (C-MAPSS) – FD001 subset
Goal: Build a pipeline that predicts the Remaining Useful Life (RUL) of aircraft engines

## Overview
Industrial sensors generate large amounts of time-series data. Raw sensor signals need to be transformed into meaningful features before they can be used in machine learning models. This lab builds a full pipeline that:

1. Loads and preprocesses raw sensor data
2. Extracts time-series features using tsfresh
3. Reduces features using filter-based methods
4. Further refines features using a Genetic Algorithm (DEAP)
5. Trains a regression model to predict RUL
6. Evaluates and times the full pipeline

## Environment

Platform: Azure Databricks
Storage: Azure Data Lake Storage Gen2 (ADLS)
Language: Python (PySpark + Pandas)
Containers used:

    -> raw/cmapss/ – original .txt files
    -> processed/cmapss/ – cleaned and normalized data
    -> curated/cmapss/ – final selected features

## Dataset
The NASA C-MAPSS FD001 dataset contains sensor readings from 100 aircraft engines, each run until failure. It has:

- 20,631 training rows
- 13,096 test rows
- 21 sensors + 3 operating conditions per row
- One row per engine per cycle

Downloaded from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ (item 6)


## How to Run
1. Setup
Upload the following files to your ADLS raw/cmapss/ container:
```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

In your Databricks notebook, configure ADLS access:
```
pythonstorage_account_name = "your_storage_account_name"
access_key = "your_access_key"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    access_key
)

raw_path       = f"abfss://raw@{storage_account_name}.dfs.core.windows.net/cmapss"
processed_path = f"abfss://processed@{storage_account_name}.dfs.core.windows.net/cmapss"
curated_path   = f"abfss://curated@{storage_account_name}.dfs.core.windows.net/cmapss"
```

2. Install dependencies
```
python%pip install tsfresh deap xgboost
```
Restart the kernel after installing.

3. Run the notebook
Run lab5_pipeline.ipynb cell by cell from top to bottom.

## Pipeline Steps

# Step 1 – Load Data (PySpark)

Data is loaded using PySpark since files are space-separated with no headers. Each row is parsed into 26 named columns: unit, cycle, 3 operating settings, and 21 sensors.
```
col_names = ['unit', 'cycle'] + [f'op_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
```

# Step 2 – Compute RUL

RUL is computed per engine as:
```
RUL = max_cycle_for_that_engine - current_cycle
```
This gives each row a label representing how many cycles remain until failure.

# Step 3 – Drop Constant Sensors

Sensors with near-zero standard deviation carry no useful information. We drop them:
```
Dropped: sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
Kept: 14 out of 21 sensors
```

# Step 4 – Normalize

MinMaxScaler is applied to scale all sensor values to [0, 1]. The scaler is fit on training data only and applied to test data to prevent data leakage.

# Step 5 – Save to Processed Container

Preprocessed data is saved as Parquet to the processed/cmapss/ container.

## Feature Extraction (tsfresh)

tsfresh requires Pandas, so the Spark DataFrame is converted. tsfresh treats each engine (unit) as one time series and extracts statistical features per sensor across all its cycles.

```
extracted = extract_features(
    ts_input,
    column_id='unit',
    column_sort='cycle',
    n_jobs=4
)
```

Result: 10,962 features extracted across 100 engines
Runtime: 111 seconds

-> Note on n_jobs: I originally tried n_jobs=-1 (use all cores) but Databricks does not support this — it throws ValueError: Number of processes must be at least 1. Setting n_jobs=4 fixed the error.

## Feature Selection

tsfresh generates thousands of features. Most are noise. We use 3 filter passes to reduce them efficiently before the Genetic Algorithm.

# Pass 1 – Variance Threshold
Removes features with near-zero variance; they don't change between engines so they can't help predict RUL.

```
10,962 → 6,701 features
```

# Pass 2 – Pearson Correlation with RUL
Computes how linearly correlated each feature is with RUL. We keep the top 150 most correlated features. This is extremely fast even on thousands of features.
```
6,701 → 150 features
```

# Pass 3 – Mutual Information
Mutual information captures non-linear relationships between features and RUL. We keep the top 50 features.
```
150 → 50 features
```

-> I originally ran Mutual Information before the correlation filter on the full 6,701 features. This caused the cell to run for over 27 minutes without finishing. I killed it and switched the order.
The new order runs Pearson first (near-instant on large sets), bringing features down to 150 before MI runs. The tradeoff is that MI scores are computed on a pre-filtered set, meaning some features that are non-linearly related to RUL but not linearly related might be discarded. I searched about it and in practice this seemed acceptable because the Genetic Algorithm in the next step compensates for any suboptimal choices here.

## Genetic Algorithm (DEAP)
A Genetic Algorithm evolves subsets of the 50 filtered features to find the combination that gives the best RUL prediction.

# Design

# Parameter:	Value
Chromosome:	Binary vector of length 50 
Population Size: 	20 individuals
Generations:	10
Crossover Probability:	0.7
Mutation Probability:	0.2
Selection:	Tournament (size 3)

# Fitness Function

```
fitness = RMSE + 0.1 * number_of_selected_features
```
This penalizes using too many features hence encouraging the GA to find a small but accurate subset.

# Result
```
GA done in 21.64s
Selected 10 features from 50
```

## Model Training and Evaluation
An XGBoost regressor is trained on the 10 GA-selected features.
```
model = XGBRegressor(n_estimators=100, random_state=42)
```

# Results:
RMSE: 4.802
MAE:  1.937
R2:   0.992
Features used: 10

An R² of 0.992 means the model explains 99.2% of variance in RUL which is a strong predictive performance with only 10 features.


========================================
## PIPELINE SUMMARY
========================================
Features after tsfresh:       10962
Features after variance:       6701
Features after Pearson:         150
Features after MI:               50
Features after GA:               10
========================================
tsfresh extraction time:  111.16s
GA time:                   21.64s
========================================
RMSE:  4.802
MAE:   1.937
R2:    0.992
========================================

## Optimizations
- Used n_jobs=4 in tsfresh for parallel extraction
- Ran Pearson correlation before Mutual Information (saved ~25 minutes)
- Kept only 50 features before GA instead of passing all filtered features
- Used a small GA population (20) and limited generations (10) to balance speed and quality

## Dependencies
```
pyspark
tsfresh
deap
xgboost
scikit-learn
pandas
numpy
```

Install with:
```
%pip install tsfresh deap xgboost
```

## References

- NASA C-MAPSS Dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- tsfresh documentation: https://tsfresh.readthedocs.io
- DEAP documentation: https://deap.readthedocs.io



