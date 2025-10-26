import numpy as np
import pandas as pd
import joblib

def predict_with_rules_then_model(fe_df, model_path,output_file_path):
    """
        Apply the two-stage matcher:
          1) Use hard rules for rows already decided in `fe_df` (hard_decision != NaN).
          2) For undecided rows (hard_decision is NaN), load a trained ML pipeline and predict.
          3) Combine both buckets, write a BI-friendly CSV, and return the full combined DataFrame.

        Parameters
        ----------
        fe_df : pd.DataFrame
            Feature-engineered dataframe produced by `feature_engg_df` (must include:
            hard_decision, hard_reason, and all ML feature columns the model expects).
        model_path : str | Path
            Path to a previously trained and saved sklearn Pipeline (joblib .pkl).
        output_file_path : str | Path
            Destination CSV for a compact, BI-friendly view of predictions.

        Returns
        -------
        pd.DataFrame
            The combined DataFrame (rules + model) including `final_pred`, `final_proba`,
            and `final_reason` for every row.
        """


    # --- Basic sanity checks ---------------------------------------------------
    if "hard_decision" not in fe_df.columns or "hard_reason" not in fe_df.columns:
        raise ValueError(
            "fe_df must include 'hard_decision' and 'hard_reason'. "
            "Make sure you ran the full feature engineering step first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model pipeline not found at {model_path}. Train and save it first."
        )

    # --- Split: rule-decided vs undecided -------------------------------------
    df_rules = fe_df[fe_df["hard_decision"].notna()].copy() #filtered by hard rules
    df_ml    = fe_df[fe_df["hard_decision"].isna()].copy() #will be pass to model

    # ---Model loading-------------------------------------------------------------
    pipe = joblib.load(model_path)

    # rules → final
    # --- Finalize rule-decided rows -------------------------------------------
    decided = df_rules.copy()
    decided["final_pred"]   = decided["hard_decision"].astype(int)
    decided["final_proba"]  = np.where(decided["final_pred"] == 1, 1.0, 0.0)
    decided["final_reason"] = decided["hard_reason"].fillna("rule")

    # --- Model prediction for undecided rows ----------------------------------
    # models → final
    if len(df_ml) > 0:
        preds  = pipe.predict(df_ml)
        probas = pipe.predict_proba(df_ml)[:, 1]
        ml_out = df_ml.copy()
        ml_out["final_pred"]   = preds
        ml_out["final_proba"]  = probas
        ml_out["final_reason"] = "ml_model"
        combined = pd.concat([decided, ml_out]).sort_index()

    else:
        combined = decided

    # --- Deduplicate just in case (keeps first occurrence by index order) -----
    combined = combined.drop_duplicates()

    # --- Final output view ----------------------------------------------

    final_cols = ['page_part_title', 'page_part_number', 'page_part_description',
                  'page_part_product_group', 'page_part_manufacturer',
                  'client_part_manufacturer', 'client_part_number', 'client_part_type',
                  'client_part_internal_number', 'client_part_product_group',
                  'client_part_price', 'page_part_price', 'manu_first_match', 'final_pred', 'final_proba',
                  'final_reason','hard_decision', 'hard_reason']

    final_df_pred = combined[final_cols].copy()
    final_df_pred.to_csv(output_file_path, index=False)
    print('final prediction saved to ....', output_file_path)
    return combined