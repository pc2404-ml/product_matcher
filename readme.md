# Product Matcher
### This project is about matching product parts between a client’s internal catalog and a web-scraped supplier catalog.
# Project Structure

product_matcher_assignment/

     ├── data/                      # Input & output data files
     │   ├── data.json              # Raw JSON input file (training or testing data)     
     │   ├── feature_eng.csv        # Feature engineered dataset (intermediate output)
     │   ├── final_prediction.csv   # Final combined predictions (rules + model)
     │   ├── model_stats.txt        # Model evaluation metrics (Accuracy, Recall, F1, AUC)
     │   ├── test_sample.json       # Example testing data file
     │
     ├── eda_output/                # Output directory for EDA plots
     │   └── *.png                  # PNG visualizations (price distributions, label counts, etc.)
     │
     ├── models/                    # Saved ML models
     │   └── model.pkl              # Trained LightGBM pipeline (ready for inference)
     │
     ├── notebooks/                 # Optional Jupyter notebooks for exploration
     │
     ├── src/                       # Source code     
     │   ├── eda.py                 # EDA: Loads JSON, prints stats, saves plots
     │   ├── feature_engineering.py # Feature engineering pipeline
     │   ├── model.py               # LightGBM model & training helpers
     │   ├── model_predict.py       # Rule-first inference on engineered DataFrame
     │   ├── model_train_n_predict.py # Full workflow: FE → train/test split → model → combine → save
     │   ├── prediction_pipeline.py # Inference pipeline (load .pkl and run predict)
     │   ├── train_pred_pipeline.py # End-to-end script for training + inference
     │   └── utilities.py           # Helper functions (text normalization, part cleaning, etc.)
     │
     ├── requirements.txt           # Python dependencies
     └── readme.md                  # Project overview & usage guide

# Data schema 

## Raw JSON should contain at least:

#### page_part_title, page_part_number, page_part_description,page_part_product_group, page_part_manufacturer, page_part_price,client_part_manufacturer, client_part_number, client_part_type,client_part_internal_number, client_part_product_group, client_part_price
#### Optional (for training): label (0/1)
     
# Quick start
## 1) Create a virtual environment & install dependencies
#### python -m venv .venv
#### source .venv/bin/activate 
#### pip install -r requirements.txt

## 2) Put input data
#### Place your raw file at data/data.json
