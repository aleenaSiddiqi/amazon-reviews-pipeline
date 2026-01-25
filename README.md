# Overview

- Azure Blob storage (ADLS Gen2): data lake for storing raw,
- Azure Machine learning compute instance: VM for data download and preprocessing
- Azure data factory (ADF): Pipeline orchestration and data transformation
- AzCopy: Command-line tool for blob storage operations

## Implementation

### Azure Storage Account Setup
The first thing we did was set up the Azure Storage Account. I created a storage account named amazondatalake60308963 in the Qatar Central region. During configuration, I made sure to enable Hierarchical Namespaces, which is critical for ADLS Gen2. Why? Because hierarchical namespaces allow the storage to behave more like a traditional file system with directories, which makes it much more efficient for big data analytics and allows for better access control and organization.After creating the storage account, I created three containers to organize our data at different stages:
-raw: for storing source data exactly as we receive it
-processed: for transformed and cleaned data
-curated: for analytics-ready data that's been aggregated or joined
This three-zone architecture is a best practice in data engineering because it keeps your original data safe while also allowing you to build increasingly refined versions for different use cases.

### Data Acquisition
- The second step was about data acquisition. I started with the product metadata file. I downloaded meta_Electronics.json.gz from the Stanford SNAP dataset website and uploaded it via the Azure Portal UI directly to the raw container. This was straightforward since the file was small enough to handle through the web interface.
- For the larger reviews dataset, I took a different approach using command-line tools, which is more realistic for handling big data files. First, I created an Azure ML Compute Instance, which is essentially a cloud-based virtual machine. Once the VM was running, I opened a terminal and downloaded the reviews dataset (500 MB compressed) directly to the VM using wget:
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
- Then I uploaded the unzipped JSON file to blob storage using AzCopy, which is much faster than the portal for large files: 
azcopy copy \
  "./reviews_Electronics_5.json" \
  "https://amazondatalake60308963.blob.core.windows.net/raw/reviews_Electronics_5.json?<SAS_TOKEN>" \
  --overwrite=true

### Fixing the Metadata File
- The metadata file had a problem: even though it had a .json extension, it wasn't actually valid JSON. Each line was formatted as a Python dictionary with single quotes instead of the double quotes that JSON requires.
- To fix this, I downloaded the metadata file from blob storage, decompressed it, and ran a Python script to convert each line from Python dictionary format to proper JSON format. The script reads each line, parses it as a Python literal, and writes it back out as valid JSON. The fixd file was uploaded back to blob storage

python3 << 'EOF'
import ast
import json

input_file = "meta_Electronics.json"
output_file = "meta_Electronics_fixed.json"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        obj = ast.literal_eval(line)
        fout.write(json.dumps(obj) + "\n")
        
print("Metadata conversion complete.")
EOF

### Azure Data Factory pipeline
- The next step involved creating an Azure Data Factory pipeline to automate the data transformation process. I created an ADF instance named `amazon-adf-60308963` in the same region as my storage account to minimize latency and data transfer costs.
- Then we set up a linked service. A Linked Service in ADF is basically a saved connection to your data source. I created a linked service that connects ADF to my ADLS Gen2 storage account using account key authentication. This allows the pipeline to read from and write to my storage containers.
- I set up two datasets that define where data comes from and where it goes:
  Source Dataset: ds_reviews_raw_json
  Sink Dataset: ds_reviews_processed_parquet
[Schema drift when enabled, allows the pipeline to handle variations in the data structure]











