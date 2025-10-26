"""
Main Orchestration Script
=========================

This script coordinates the **entire product-matching workflow**:

1. **Path Setup**:
   - Defines all input, output, and model file paths in a consistent structure.

2. **Feature Engineering**:
   - Reads raw JSON input data.
   - Applies preprocessing, feature engineering, and hard-rule logic.
   - Saves the enriched dataset to data/feature_eng.csv.

3. **Rule vs ML Split**:
   - Splits data into:
       a) df_rules: rows already decided by *hard rules* (clear matches/non-matches).
       b) df_ml: undecided rows requiring ML classification.
   - Fills missing rule decisions with "NA" placeholders.

4. **Model Training + Evaluation**:
   - Trains the ML pipeline (LightGBM classifier) on the undecided rows.
   - Evaluates model performance (classification report, AUC, confusion matrix).
   - Combines ML-decided rows with rule-decided rows.

5. **Final Predicted Output**:
   - Creates a consolidated DataFrame with stable column ordering.
   - Saves the combined output to final_prediction.csv .

Artifacts
---------
- data/feature_eng.csv  : Intermediate enriched dataset after FE.
- models/model.pkl      : Saved ML pipeline for re-use on future data.
- data/final_prediction.csv: Combined rule + ML predictions.

"""

from feature_engineering import *  # your FE on DataFrame
from model_train_n_predict import *
from eda import *

# ---------------------------------------------------------------------------
# 1. Build Paths
# ---------------------------------------------------------------------------

root_dir                = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_file_path         = os.path.join(root_dir, "data", "data.json")
output_file_path        = os.path.join(root_dir, "data", "feature_eng.csv")
output_stat_path        =os.path.join(root_dir, "eda_output")
final_pred_output_path  = os.path.join(root_dir, "data", "final_prediction.csv")
model_path              = os.path.join(root_dir, "models/model.pkl")

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------

print("Starting EDA...")
eda_plots(input_file_path,output_stat_path)
print(f"Completed EDA, Result Plots saved to:|{output_stat_path}")
# ---------------------------------------------------------------------------
# 3. Feature Engineering Phase
# ---------------------------------------------------------------------------
print("Starting Feature Engineering...")
df_fe =feature_engg(input_file_path,output_file_path)
print(f"Feature Engineering complete. Rows: {len(df_fe)} | Saved → {output_file_path}")

# ---------------------------------------------------------------------------
# 4. Split into Rule-Decided vs ML-Decided
# ---------------------------------------------------------------------------
print("Splitting dataset into rule-decided vs undecided rows...")

df_rules = df_fe[df_fe["hard_decision"].notna()]  # already decided
df_ml = df_fe[df_fe["hard_decision"].isna()]

# For undecided rows, mark rule fields as NA (kept for schema consistency)
df_ml['hard_decision'] = 'NA'
df_ml['hard_reason'] = 'NA'
print(f"Split complete → Rule-decided: {len(df_rules)}, ML-decided: {len(df_ml)}")

# ---------------------------------------------------------------------------
# 5. Train ML Pipeline + Combine with Rule-Decided
# ---------------------------------------------------------------------------
print("Training ML model on undecided rows...")
X = df_ml.drop(columns=["label"])
y = df_ml["label"]
final_df=train_with_rules_and_model(X,y,final_pred_output_path,df_rules,model_path)
print("Model training + combination complete.")


# ---------------------------------------------------------------------------
# 6. Wrap-Up
# ---------------------------------------------------------------------------
print(f"Final prediction file saved → {final_pred_output_path}")
print(f"Trained ML pipeline saved → {model_path}")
print(f"Combined DataFrame shape: {final_df.shape}")
