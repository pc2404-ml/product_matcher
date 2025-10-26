
"""
Main orchestrator script for the Part Matcher project.

Steps:
1. Load input JSON file
2. Run feature engineering (preprocessing, enrichment, hard rules)
3. Apply prediction:
   - First, apply hard rules
   - Remaining undecided rows go through the trained ML model
4. Save final predictions to CSV
5. filter and display only "matches" (final_pred == 1)
"""

from feature_engineering import feature_engg
from model_train_n_predict import *
from model_predict import *
from eda import *




def main():
    """Run the full FE + rules + ML prediction pipeline."""
    # ========================
    # Paths configuration
    # ========================

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_file_path = os.path.join(root_dir, "data", "test_unlabeled_100.json")
    output_file_path = os.path.join(root_dir, "data", "feature_eng.csv")
    output_stat_path = os.path.join(root_dir, "eda_output")
    final_pred_output_path = os.path.join(root_dir, "data", "final_prediction.csv")
    model_path = os.path.join(root_dir, "models/model.pkl")


    # --- Step 1: Feature Engineering ---
    print("Running feature engineering...")
    fe_df = feature_engg(input_file_path, output_file_path)
    print(f"Feature engineering done. Shape: {fe_df.shape}")

    # --- Step 2: Prediction (Rules + Model) ---
    print("Running prediction with rules + model...")
    df_pred = predict_with_rules_then_model(fe_df, model_path, final_pred_output_path)
    print(f"Prediction done. Saved to {final_pred_output_path}")

    # --- Step 3: Filter matches ---
    df_matches = df_pred[df_pred["final_pred"] == 1]
    print(f"Matches found: {df_matches.shape[0]} out of {df_pred.shape[0]} records")
    print(df_matches.head(10))  # preview first 10 matches

    return df_pred, df_matches
if __name__ == "__main__":
    df_pred, df_matches = main()
