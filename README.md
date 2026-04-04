# Assignment 2 — Model Training

**Course:** Cloud Computing — DSAI3202  
**Student ID:** 60308963

---

## Project Overview

An end-to-end MLOps pipeline built on Azure ML that trains a binary sentiment classifier on Amazon Electronics reviews. The pipeline covers data splitting, feature engineering, model training, hyperparameter tuning, and CI/CD automation via Azure DevOps.

The task is binary classification: given an Amazon Electronics review, predict whether it is **positive** (rating ≥ 4) or **negative** (rating < 4).

---

## Repository Structure

```
amazon-reviews-pipeline/
├── src/
│   ├── train.py              # Training script
│   ├── score.py              # Scoring script for endpoint
│   └── invoke_endpoint.py    # Script to invoke deployed endpoint
├── env/
│   ├── conda.yml             # Training environment dependencies
│   ├── environment.yml       # Azure ML training environment definition
│   └── inference_conda.yml   # Inference environment dependencies
├── jobs/
│   ├── train_job.yml         # Azure ML training job definition
│   ├── sweep_job.yml         # Hyperparameter sweep job definition
│   ├── deployment.yml        # Managed online endpoint deployment
│   ├── data_asset_train.yml  # Data asset — train split
│   ├── data_asset_val.yml    # Data asset — val split
│   ├── data_asset_test.yml   # Data asset — test split
│   └── data_asset_deploy.yml # Data asset — deploy split
├── azure-pipelines.yml       # Azure DevOps CI/CD pipeline
└── README.md
```

---

## Lab 4 — Feature Engineering Pipeline

The feature engineering pipeline was built in Azure ML and produces merged feature datasets for all four splits.

| Component | Script | Description |
|---|---|---|
| `split_dataset` | `split.py` | Splits data: train (60%), val (15%), test (15%), deploy (10%) |
| `normalize_text` | `normalize.py` | Cleans and normalises review text |
| `review_length` | `length.py` | Computes word and character count features |
| `sentiment_features` | `sentiment.py` | Extracts sentiment scores using VADER |
| `semantic_embeddings` | `sbert.py` | Generates 384-dim BERT embeddings per review |
| `tfidf_features` | `tfidf.py` | Fits TF-IDF on train, transforms all splits |
| `merge_features` | `merge.py` | Merges all feature outputs into one parquet per split |

![4split_pipeline](image-9.png)

---

## Lab 5 — Model Training

### Data Assets

The following Azure ML Data Assets are registered and reference the merged feature outputs from the Lab 4 pipeline:

| Asset Name | Split |
|---|---|
| `amazon_reviews_merged_features_train` | Training (60%) |
| `amazon_reviews_merged_features_val` | Validation (15%) |
| `amazon_reviews_merged_features_test` | Test (15%) |
| `amazon_reviews_merged_features_deploy` | Deployment (10%) |

![datassets](image-8.png)

---

### Model Choice — Logistic Regression

A `LogisticRegression` classifier from scikit-learn was chosen for the following reasons:

- Fast to train on high-dimensional feature vectors
- Performs well on linearly separable text classification tasks
- Easily interpretable coefficients
- Supports probability outputs for AUC computation
- Well-suited for CI/CD pipelines where training speed matters

---

### Features Used

The merged dataset contains the following feature types, all pre-computed by the Lab 4 pipeline:

| Feature Type | Columns | Description |
|---|---|---|
| BERT embeddings | `bert_embedding_0` … `bert_embedding_383` | Dense 384-dim semantic vectors from SBERT |
| Sentiment | `sentiment_pos`, `sentiment_neg`, `sentiment_neu`, `sentiment_compound` | VADER sentiment scores |
| Review length | `review_length_words`, `review_length_chars` | Word and character count statistics |
| Helpfulness | `helpfulness_ratio` | Ratio of helpful votes to total votes |

The feature matrix is constructed by combining all numeric columns and BERT embedding columns using `np.hstack`. TF-IDF is fit only on training data and transformed on all splits to prevent data leakage.

---

### Feature Experiments

Three feature configurations were tested to understand the impact of different feature representations:

| Run | Feature Config | `--features` flag |
|---|---|---|
| Run 1 | SBERT embeddings only | `sbert_only` |
| Run 2 | SBERT + TF-IDF | `sbert_tfidf` |
| Run 3 | All features (SBERT + numeric + TF-IDF) | `all` |

//

---

### Hyperparameter Tuning

A sweep job was run using **random sampling** over the following search space:

| Hyperparameter | Type | Range |
|---|---|---|
| `C` | uniform | 0.01 – 10.0 |
| `max_iter` | choice | 500, 1000, 2000 |

**Sweep configuration:**
- Sampling algorithm: Random
- Max total trials: 6
- Max concurrent trials: 2
- Objective: Maximise `val_accuracy`

**Best hyperparameters found:**

| Hyperparameter | Value |
|---|---|
| `C` | `3.1353264492857433` |
| `max_iter` | `1000` |

![best_params](image-7.png)

---

### MLflow Metrics Logged

The following metrics are logged for each split (train, val, test):

| Metric | Description |
|---|---|
| `{split}_accuracy` | Classification accuracy |
| `{split}_f1` | Weighted F1 score |
| `{split}_precision` | Weighted precision |
| `{split}_recall` | Weighted recall |
| `{split}_auc` | ROC-AUC score |
| `training_runtime_seconds` | Total training time |
| `C` | Regularisation parameter used |
| `max_iter` | Max iterations used |
| `feature_config` | Feature configuration used |

![metrics](image-6.png)



---

### Final Model Performance

Trained using optimal hyperparameters (`C=3.135`, `max_iter=1000`) on all features.

![final_model_performance](image-5.png)

---

### Registered Model

The trained model artifact (`model.pkl`) is registered in the Azure ML Model Registry as:

- **Name:** `amazon-reviews-classifier`
- **Version:** 1
- **Type:** Custom model

![resgistered_model](image-4.png)

---

## CI/CD — Azure DevOps

Training is automated via `azure-pipelines.yml`. Every push to `assignment2_model_training` automatically submits the training job to Azure ML.

### Pipeline Steps
1. Installs Azure ML CLI extension
2. Sets workspace defaults
3. Submits `jobs/train_job.yml` as an Azure ML job
4. Streams job logs

### Trigger
```yaml
trigger:
  branches:
    include:
      - assignment2_model_training
```

**Service Connection:** `SC-UDST-CCIT-DSAI3202-1`

![devOps_pipeline_automation](image-2.png)

---

## Deployment

### Endpoint

The model is deployed to an Azure ML Managed Online Endpoint:

- **Endpoint name:** `amazon-review-endpoint-60308963`
- **Deployment name:** `amazon-review-deployment`
- **Instance type:** `Standard_F2s_v2`
- **Instance count:** 1
- **Auth mode:** Key

### Inference Environment

Defined in `env/inference_conda.yml`. Dependencies: `numpy`, `pandas`, `scikit-learn`, `joblib`, `azureml-defaults`.

![deployed_endpoint](image-3.png)

---

## Deployment Evaluation Results

The deployment dataset (`amazon_reviews_merged_features_deploy`) was used to invoke the endpoint and evaluate real predictions returned by the deployed model:

| Metric | Score |
|---|---|
| Total predictions | 37,577 |
| Accuracy | 0.8291 |
| F1 Score | 0.8110 |
| Precision | 0.8115 |
| Recall | 0.8291 |
| AUC | 0.8245 |

![deployment_results](image.png)

---

## Azure ML Resources

| Resource | Name |
|---|---|
| Workspace | `Amazon-Electronics-Lab-60308963` |
| Resource Group | `rg-60308963` |
| Subscription | `UDST-CCIT-DSAI3202-1` |
| Compute Cluster | `cluster-feature-engineering` |
| Training Environment | `amazon-review-training-env` |
| Inference Environment | `amazon-review-inference-env` |
| Registered Model | `amazon-reviews-classifier` |
| Endpoint | `amazon-review-endpoint-60308963` |