from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb



def select_features(X, features=None):
    if features is None:
        return X
    return X[features]

def build_ml_pipe():
    features = [
        "group_exact", "manu_first_match", "manu_in_text", "pn_in_text",
        "client_part_price", "page_part_price", "price_diff_abs", "price_diff_rel",
        "client_pn_len", "page_pn_len", "pn_len_diff",
        "pn_exact", "pn_sim", "pn_prefix_match",
        "manu_sim", "group_jaccard", "price_ratio"
    ]
    selector = FunctionTransformer(select_features, kw_args={"features": features}, validate=False)
    clf = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42
    )
    return Pipeline([("select", selector), ("clf", clf)])