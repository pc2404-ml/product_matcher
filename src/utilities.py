import re
from rapidfuzz import fuzz

def norm_pn(s):
     if not s:  # handle None or empty
        return ""
     s = str(s).upper()
     # remove dashes, spaces, slashes, dots, etc.
     s = re.sub(r"[-_/.\s,#:]", "", s)
     return s


def first_word_manu(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().upper()
    return s.split()[0]


def text_contains(text, keyword):
    return int(str(keyword) in str(text))




def enrich_features(df):
    # pn_exact if missing
    if "pn_exact" not in df.columns:
        df["pn_exact"] = (df["client_part_number"].fillna("").str.upper().str.replace(r"[-_/.\s]","",regex=True) ==
                          df["page_part_number"].fillna("").str.upper().str.replace(r"[-_/.\s]","",regex=True)).astype(int)

    # pn_sim if missing
    if "pn_sim" not in df.columns:
        df["pn_sim"] = df.apply(lambda r: fuzz.ratio(str(r.client_part_number), str(r.page_part_number))/100, axis=1)

    # pn_prefix_match if missing
    if "pn_prefix_match" not in df.columns:
        df["pn_prefix_match"] = df.apply(lambda r: int(str(r.client_part_number)[:4] == str(r.page_part_number)[:4]), axis=1)

    # manu_sim if missing
    if "manu_sim" not in df.columns:
        df["manu_sim"] = df.apply(lambda r: fuzz.ratio(str(r.client_part_manufacturer), str(r.page_part_manufacturer))/100, axis=1)

    # group_jaccard if missing
    if "group_jaccard" not in df.columns:
        df["group_jaccard"] = df.apply(
            lambda r: len(set(str(r.client_part_product_group).split()) & set(str(r.page_part_product_group).split())) /
                      (len(set(str(r.client_part_product_group).split()) | set(str(r.page_part_product_group).split())) + 1e-6),
            axis=1
        )

    # price_ratio if missing
    if "price_ratio" not in df.columns:
        df["price_ratio"] = df.apply(
            lambda r: r.client_part_price / r.page_part_price if r.page_part_price and r.client_part_price else 0,
            axis=1
        )
        df["price_ratio"] = df["price_ratio"].clip(0, 10)

    # pn_len_diff if missing
    if "pn_len_diff" not in df.columns:
        df["pn_len_diff"] = abs(df["client_pn_len"] - df["page_pn_len"])

    return df


def apply_hard_rules(row):
    # Hard match
    if row["client_part_number"] == row["page_part_number"] and row["manu_first_match"] == 1:
        return 1, "rule_pn_brand_exact"
    # Hard non-match
    if row["manu_first_match"] == 0 and row["price_diff_rel"] > 0.95:
        return 0, "rule_brand_conflict_price_far"
    return None, None

