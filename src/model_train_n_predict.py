import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import lightgbm as lgb
import os
from model import build_ml_pipe

def train_with_rules_and_model(X,y,output_file_path,df_rules,model_path):
    """
        Train the ML pipeline on the *undecided* rows (rows that were NOT decided by hard rules),
        then combine ML predictions with the already decided rule-based rows and save a BI-friendly CSV.

        Parameters
        ----------
        X : pd.DataFrame
            Engineered feature table for the ML subset (i.e., undecided rows). Should include all
            columns expected by the model's selector (in build_ml_pipe).
        y : pd.Series
            Binary target for the ML subset (0/1). Must align by index with X.
        output_file_path : str | Path
            Where to write the final combined predicted CSV.
        df_rules : pd.DataFrame
            DataFrame of rows already decided by hard rules (hard_decision notna).
        model_path : str | Path
            Where to save the trained ML pipeline (.pkl). The parent directory will be created.

        Returns
        -------
        pd.DataFrame
            The final Predicted DataFrame that was saved to `output_file_path`.

        Workflow
        --------
        1) Split X/y into train/test and fit the ML pipeline (only for undecided rows).
        2) Evaluate on test split (prints classification report, AUC, confusion matrix).
        3) Create train/test result frames (true_label, pred_label, pred_proba, source).
        4) Normalize rule-decided rows to have the same service columns (true_label, pred_* , source).
        5) Concatenate rule-decided with ML results (outer-join semantics, keep all info).
        6) Write .
        """
    # ---- Safety checks --------------------------------------------------------
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or single-column DataFrame.")
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("If y is a DataFrame, it must have exactly one column.")
        y = y.iloc[:, 0]
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    # ---- 1) Train ML pipeline on undecided rows ------------------------------
    pipe = build_ml_pipe()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe.fit(X_train, y_train)

    # ---- 2) Quick evaluation --------------------------------------------------
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print(" ML on undecided: test metrics ")
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print(report)
    print("AUC:", auc)
    print("Confusion:\n", cm)
    
    # --- Save evaluation to file ---
    stats_path = os.path.join(os.path.dirname(output_file_path), "model_stats.txt")
    with open(stats_path, "w") as f:
        f.write("Classification Report \n")
        f.write(report + "\n\n")
        f.write("AUC \n")
        f.write(str(auc) + "\n\n")
        f.write("Confusion Matrix \n")
        f.write(str(cm) + "\n")

    print(f"Model evaluation saved → {stats_path}")

    # ---- 3) Create results for test and train splits -------------------------
    res_te = X_test.copy()
    res_te["true_label"] = y_test.values
    res_te["pred_label"] = y_pred
    res_te["pred_proba"] = y_proba
    res_te["source"] = "test"

    y_pred_tr  = pipe.predict(X_train)
    y_proba_tr = pipe.predict_proba(X_train)[:, 1]
    res_tr = X_train.copy()
    res_tr["true_label"] = y_train.values
    res_tr["pred_label"] = y_pred_tr
    res_tr["pred_proba"] = y_proba_tr
    res_tr["source"] = "train"

    results_all_ml = pd.concat([res_tr, res_te]).sort_index()

    # ---- 4) Normalize rule-decided rows' schema ------------------------------
    # mark rule-decided rows
    df_rules = df_rules.copy()
    df_rules["source"]      = "hard_rules"
    df_rules["true_label"]  = df_rules["hard_decision"].astype(int)
    df_rules["pred_label"]  = np.nan
    df_rules["pred_proba"]  = np.nan

    # ---- 5) Concatenate with outer semantics ---------------------------------
    common = sorted(set(df_rules.columns) & set(results_all_ml.columns))
    combined = pd.concat([df_rules[common], results_all_ml[common]], ignore_index=True).drop_duplicates()

    # Persist the trained model
    joblib.dump(pipe, model_path)
    print(f'Saved ML pipeline →',{model_path})

    # ---- 6) Deduplication of final dataframe ---------------------------------
    combined = combined.drop_duplicates()

    # ---- 7) Final Columns selection --------------------------------------
    final_cols = ['page_part_title', 'page_part_number', 'page_part_description',
                  'page_part_product_group', 'page_part_manufacturer',
                  'client_part_manufacturer', 'client_part_number', 'client_part_type',
                  'client_part_internal_number', 'client_part_product_group',
                  'client_part_price', 'page_part_price', 'manu_first_match', 'true_label', 'pred_label', 'pred_proba',
                  'hard_decision', 'hard_reason', 'source']


    final_df_pred = combined[final_cols].copy()

    # ---- 7) Saving Prediction to CSV file --------------------------------------
    final_df_pred.to_csv(output_file_path, index=False)
    print("output file saved:", output_file_path)
    return final_df_pred