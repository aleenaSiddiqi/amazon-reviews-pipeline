### Amazon Electronics Review — Feature Engineering Pipeline

## Lab 4 | Azure ML + Databricks

## Project Overview

This lab builds a full feature engineering pipeline for the Amazon Electronics review dataset using Azure Databricks for data exploration and Azure ML Pipelines for modular, reproducible feature extraction. The engineered features are registered in the Azure ML Feature Store for reuse in downstream modeling labs.

The pipeline produces a feature-enriched dataset containing:

- Text length features
- Sentiment scores
- TF-IDF term frequency features
- Semantic embeddings 
- Helpfulness ratio (additional feature)

## Part A — Databricks: Data Exploration & Visualizations

Before building the pipeline, the Gold Dataset (features_v1) was loaded in Databricks and explored through visualizations to understand the data distribution and identify any quality issues relevant to feature engineering.

# Rating Distribution

```
import matplotlib.pyplot as plt

rating_dist = clean_gold_df.groupBy("overall").count().orderBy("overall").toPandas()

plt.figure(figsize=(8, 5))
plt.bar(rating_dist["overall"], rating_dist["count"], color="steelblue", edgecolor="black")
plt.title("Rating Distribution")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.show()
```

- It shows the distribution of star ratings (1–5) across all reviews.
- It matters because a heavily skewed distribution (e.g. mostly 5-star reviews) can bias downstream models. Understanding class imbalance informs decisions like whether to stratify splits or apply class weighting during training.

# Review Length Distribution

```
from pyspark.sql.functions import length

clean_reviews_df = clean_gold_df.withColumn("review_length", length(col("reviewText")))
review_lengths = clean_reviews_df.select("review_length").toPandas()

plt.figure(figsize=(8, 5))
plt.hist(review_lengths["review_length"], bins=50, color="coral", edgecolor="black")
plt.title("Review Length Distribution")
plt.xlabel("Review Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```


- It shows how long reviews tend to be in characters.
- It matters because very short reviews carry less signal for NLP tasks. This informs minimum length thresholds during cleaning. It also motivates the review_length_chars and review_length_words features, as length itself can be predictive of rating or helpfulness.

# Reviews over time

```
reviews_per_year = clean_gold_df.groupBy("review_year").count().orderBy("review_year").toPandas()

plt.figure(figsize=(8, 5))
plt.plot(reviews_per_year["review_year"], reviews_per_year["count"], marker="o", color="green")
plt.title("Reviews Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()
```

- It shows the volume of reviews per year.
- It matters because if review volume is concentrated in certain years, a random sample may over represent those years, and hence introduce temporal bias. This visualization motivated the stratified sampling approach used later.

# Helpfulness Ratio Distribution

```
from pyspark.sql.functions import when

clean_reviews_df = clean_reviews_df.withColumn(
    "helpfulness_ratio",
    when(col("helpful")[1] > 0, col("helpful")[0] / col("helpful")[1]).otherwise(None)
)

helpfulness = clean_reviews_df.select("helpfulness_ratio").dropna().toPandas()

plt.figure(figsize=(8, 5))
plt.hist(helpfulness["helpfulness_ratio"], bins=40, color="purple", edgecolor="black")
plt.title("Helpfulness Ratio Distribution")
plt.xlabel("Helpful Votes / Total Votes")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

- It shows the proportion of users who found each review helpful.
- It matters because Helpfulness ratio is a signal of review quality. Reviews with high helpfulness ratios may carry stronger training signal, and the ratio itself is a useful engineered feature for downstream models.

# Mean Review Length Over Years (Drift Check)

```
import seaborn as sns

drift_df = clean_reviews_df.groupBy("review_year").avg("review_length").orderBy("review_year").toPandas()

plt.figure(figsize=(10, 5))
sns.lineplot(data=drift_df, x="review_year", y="avg(review_length)", marker="o")
plt.title("Mean Review Length Over Years (Drift Check)")
plt.ylabel("Avg Characters")
plt.show()
```

- It shows whether the average review length changes significantly across years.
- It matters cuz if review writing behaviour changes over time (data drift), a model trained on older data may not generalise to newer reviews. This plot helps detect temporal drift in text characteristics.

## Part B — Drift Analysis & Stratified Sampling

The drift analysis revealed that review behaviour varies across years. To produce a representative 300,000-row sample that mirrors the full dataset's temporal distribution, stratified sampling by year was used.

```
total_rows = clean_gold_df.count()
target_rows = 300000
fraction = target_rows / total_rows

fractions = (
    clean_gold_df.select("review_year")
    .distinct()
    .withColumn("fraction", col("review_year") * 0 + fraction)
    .toPandas()
    .set_index("review_year")["fraction"]
    .to_dict()
)

df_sampled = clean_gold_df.stat.sampleBy("review_year", fractions, seed=42)

print(f"Sampled rows: {df_sampled.count()}")
display(df_sampled.groupBy("review_year").count().orderBy("review_year"))
```

A simple random sample risks over-representing years with high review volumes. Stratified sampling ensures every year contributes proportionally to the sample, making the dataset resistant to temporal drift.
The sampled dataset was saved to the curated container in the Data Lake as features_v1_sampled and registered as an Azure ML Data Asset for use in the pipeline.


## Part C — Azure ML Setup: Datastore & Data Asset

# Datastore (datastores/curated_adls.yml)

```
$schema: https://azuremlschemas.azureedge.net/latest/datastore.schema.json
name: blobkey
type: azure_blob
account_name: <STORAGE_ACCOUNT_NAME>
container_name: curated
credentials:
  account_key: <STORAGE_ACCOUNT_KEY>
```

This file registers the Azure Blob Storage container as a named datastore called blobkey in Azure ML. The name blobkey is referenced throughout the pipeline and data asset definitions. It was registered using:

```
az ml datastore create --file datastores/curated_adls.yml \
  --resource-group <RESOURCE_GROUP> \
  --workspace-name <WORKSPACE_NAME>
```

# Data Asset (data/features_v1_sampled.yml)

```
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: amazon_electronics_features_v1_sampled
version: 1
type: uri_folder
path: azureml://datastores/blobkey/paths/features_v1_sampled/
description: Sampled Gold dataset used as input for Lab 4 feature engineering
```

This registers the sampled Parquet dataset as a versioned Azure ML Data Asset. The path field uses the blobkey datastore name and points to the folder where the sampled data was saved from Databricks. It was registered using:

```
az ml data create --file data/features_v1_sampled.yml \
  --resource-group <RESOURCE_GROUP> \
  --workspace-name <WORKSPACE_NAME>
```

## Part D — Feature Engineering Components

Each feature engineering step is implemented as a reusable Azure ML component; as a self-contained unit with defined inputs, outputs, and a Python script. Components are defined using a component.yml file and registered in Azure ML.

Every component.yml follows this structure:

```
$schema: ...          # Tells Azure ML which schema to validate against
name: ...             # Unique identifier used when registering and referencing the component
display_name: ...     # Human-readable name shown in Azure ML Studio
type: command         # This is a command component (runs a Python script)

inputs:               # What data/parameters the component receives
  data:
    type: uri_folder  # A folder of files (e.g. Parquet files)

outputs:              # What data the component produces
  out:
    type: uri_folder

code: .               # The folder containing the script (. = same folder as component.yml)

command: >            # The shell command to run, with input/output placeholders
  python script.py
  --data ${{inputs.data}}
  --out ${{outputs.out}}

environment: ...      # The Docker/conda environment to run the script in
```

Each component is then registered similarly as done above

# Split Dataset

Splits the sampled dataset into train (70%), validation (15%), and test (15%) splits. 

```
- def parse_args():
    # Defines command-line arguments the script accepts.
    # --data: path to input dataset
    # --seed: random seed for reproducibility (default 42)
    # --train_ratio: proportion for training set (default 0.7)
    # --val_ratio: proportion for validation set (default 0.15)
    # --train_out, --val_out, --test_out: output folder paths
```

The split is done in two stages:

First split: Separates the training set from the remaining data using train_test_split
Second split: Splits the remaining data into validation and test using a recalculated ratio:

```
val_size = args.val_ratio / (1 - args.train_ratio)
# e.g. 0.15 / (1 - 0.70) = 0.15 / 0.30 = 0.5
# So the remainder is split 50/50 into val and test
```

This two-stage approach is necessary because train_test_split only splits into two sets at a time.
Uses the default Azure ML sklearn environment since only pandas and scikit-learn are needed.

# Normalize text 

Cleans review text before feature extraction to ensure consistency across all splits.

Steps:
- Lowercase all text
- Remove URLs using regex: re.sub(r'http\S+|www\.\S+', '', text)
- Remove numbers: re.sub(r'\d+', '', text)
- Remove punctuation: re.sub(r'[^\w\s]', '', text)
- Strip whitespace: text.strip()
- Filter out reviews shorter than 10 characters after cleaning

The reason why we normalize before feature extraction is because features like TF-IDF and embeddings are sensitive to text format. Without normalization, "GREAT!", "great" and "great." would be treated as different tokens. Normalization ensures consistent vocabulary and more meaningful features.

# Review Length features

Creates two simple but informative numeric features from review text.
- review_length_words: Number of words in the review
- review_length_charsNumber of characters in the review

Review length is correlated with rating and helpfulness. Very short reviews ("Great!") and very long detailed reviews carry different information. These features are cheap to compute and add signal without requiring NLP.

# Sentimental features

Extracts emotional tone from review text using VADER which stands for Valence Aware Dictionary and sEntiment Reasoner.

- sentiment_pos: Proportion of positive sentiment words
- sentiment_neg: Proportion of negative sentiment words
- sentiment_neu: Proportion of neutral sentiment words
- sentiment_compound: Overall polarity score from -1 (very negative) to +1 (very positive)

VADER is specifically designed for social media and informal text, making it well-suited for product reviews. It handles capitalisation, punctuation emphasis, and slang without requiring model training.

# Custom environment (conda.yml):

```
dependencies:
  - pip:
    - nltk
    - textblob
```

VADER is part of nltk which is not included in the default Azure ML environments, so a custom conda environment is required. The component.yml references this inline:
```
environment:
  conda_file: conda.yml
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
```
The image field specifies the base Docker image. Azure ML installs the conda packages on top of this base image at runtime.

# TF-IDF Features

Represents review text as a numeric vector based on word frequency and importance.

Settings used:

- max_features=100: limits vocabulary to top 100 terms (reduced from 500 to avoid memory issues)
- stop_words='english': removes common filler words like "the", "and", "is"
- ngram_range=(1,2): captures both single words and two-word phrases (e.g. "not good")

# Design decision to fit only on training data

```
vectorizer.fit_transform(train_df["reviewText"])  # fit + transform on train
vectorizer.transform(val_df["reviewText"])         # transform only on val
vectorizer.transform(test_df["reviewText"])        # transform only on test
```

Fitting the vectorizer on validation or test data would cause data leakage, that is, the model would have indirect knowledge of the test set vocabulary. The vectorizer must only learn the vocabulary from training data.
Note: Due to memory constraints on the Azure ML compute cluster and Parquet compatibility issues with sparse matrices, the TF-IDF matrices are saved separately as .npz files alongside the original Parquet data rather than merged into one file.
All TF-IDF column names are prefixed with tfidf_ to avoid clashes with existing column names (e.g. the word "brand" exists both as a column name and a TF-IDF feature).

# Semantic Embeddings

Encodes each review as a dense 384-dimensional vector using Sentence-BERT, capturing semantic meaning beyond word frequency.

Model used: all-MiniLM-L6-v2 — a lightweight Sentence-BERT model that produces high-quality embeddings efficiently. Each review becomes 384 float values (bert_embedding_0 through bert_embedding_383).

Embeddings over TF-IDF: TF-IDF treats the worlds "terrible" and "awful" as completely different words. Sentence-BERT understands they are semantically similar, producing similar vectors for reviews with the same meaning even if they use different words.

# Helpfulness feature

Derives a helpfulness ratio from the helpful array column [helpful_votes, total_votes]. It is the proportion of users.
Highly helpful reviews may be more informative for training. Including helpfulness ratio allows models to weight or filter reviews by quality.

```
lambda x: x[0] / x[1] if x is not None and len(x) == 2 and x[1] > 0 else 0.0
```

# Merge all features

Joins all feature outputs into a single feature-enriched Parquet dataset on the entity keys asin and reviewerID.

Inputs:

- Review length features
- Sentiment features
- TF-IDF features (train split)
- Semantic embeddings
- Helpfulness features

A left join is used to preserve all rows from the base dataset even if individual feature components dropped some rows during processing. Duplicate columns that appear across multiple component outputs are deduplicated before merging.

## Part E — Azure ML Pipeline

# pipeline.yml

```
$schema: ...
type: pipeline
compute: azureml:cpu-cluster    # The compute cluster to run all jobs on

inputs:
  sampled_data:                 # The pipeline's single input — the registered data asset
    type: uri_folder
    path: azureml:amazon_electronics_features_v1_sampled@latest

jobs:
  split:                        # Each job is a step in the pipeline
    type: command
    component: azureml:split_dataset@latest    # References the registered component
    inputs:
      data: ${{parent.inputs.sampled_data}}    # Wires the pipeline input to the component input
```

Jobs are connected by wiring one job's output to the next job's input:

```
normalize_train:
  inputs:
    data: ${{parent.jobs.split.outputs.train}}  # Takes the train split from the split step
```

Azure ML automatically infers the execution order from these dependencies and runs independent steps in parallel where possible.

Progress can be monitored in Azure ML Studio → Jobs, where each step appears as a node in a visual graph with its own logs and outputs.

## Part F — Feature Store Registration

# Entity Definition (feature_store/entity_amazon_review.yml)

```
$schema: https://azuremlschemas.azureedge.net/latest/featurestoreentity.schema.json
name: AmazonReview
version: "1"
description: Amazon Electronics review entity
index_columns:
  - name: asin
    type: string
  - name: reviewerID
    type: string
```

The entity defines the primary keys that uniquely identify each record. In this case the combination of product ID (asin) and reviewer ID (reviewerID). Every feature set is linked to this entity.

# Feature Set Spec (feature_store/FeatureSetSpec.yaml)

This file describes the schema of the feature set such as where the data lives, what the index columns are, and the name and type of every feature. The timestamp_column field is required by Azure ML for parquet feature sources to support point-in-time lookups:

```
source:
  type: parquet
  path: azureml://subscriptions/.../paths/...
  timestamp_column:
    name: reviewTime
```

# Feature Set Definition (feature_store/feature_set.yml)

``` 
$schema: https://azuremlschemas.azureedge.net/latest/featureSet.schema.json
name: amazon_review_features
version: "1"
description: Amazon Electronics review features including length, sentiment, TF-IDF, embeddings and helpfulness.
entities:
  - azureml:AmazonReview:1       # Links to the registered entity
specification:
  path: ./                       # Points to FeatureSetSpec.yaml in the same folder
materialization_settings:
  offline_enabled: true          # Allows offline feature retrieval for training
```

Registered the same way as in the steps above.










