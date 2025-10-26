import pandas as pd
import numpy as np
import os
from utilities import *


def feature_engg(input_file_path,output_file_path):
    #data read
    df = pd.read_json(input_file_path)

    #preprocessing
    df["client_part_manufacturer"] = df["client_part_manufacturer"].fillna("NO_MANUFACTURER")
    df["page_part_manufacturer"]   = df["page_part_manufacturer"].fillna("NO_MANUFACTURER")

    df["client_part_number"] = df["client_part_number"].fillna("NO_PARTNUMBER")
    df["page_part_number"]   = df["page_part_number"].fillna("NO_PARTNUMBER")
    df=df.fillna("UNKNOWN")


    df["client_part_number"] = df["client_part_number"].map(norm_pn)
    df["page_part_number"]   = df["page_part_number"].map(norm_pn)
    df["page_part_product_group"]   = df["page_part_product_group"].map(norm_pn)
    df["client_part_product_group"]   = df["client_part_product_group"].map(norm_pn)
    df["group_exact"] = (df["client_part_product_group"] == df["page_part_product_group"]).astype(int)

    df["client_manu_first"] = df["client_part_manufacturer"].fillna("").map(first_word_manu)
    df["page_manu_first"]   = df["page_part_manufacturer"].fillna("").map(first_word_manu)
    df["manu_first_match"] = (df["client_manu_first"] == df["page_manu_first"]).astype(int)


    df["price_diff_abs"] = abs(df["client_part_price"] - df["page_part_price"])
    df["price_diff_rel"] = df["price_diff_abs"] / df[["client_part_price","page_part_price"]].max(axis=1)
    df["price_diff_rel"] = df["price_diff_rel"].fillna(1.0)



    df["manu_in_text"] = df.apply(lambda r: text_contains(r.page_part_title, r.client_part_manufacturer), axis=1)
    df["pn_in_text"]   = df.apply(lambda r: text_contains(r.page_part_title, r.page_part_manufacturer), axis=1)


    df["client_pn_len"] = df["client_part_number"].fillna("").astype(str).str.len()
    df["page_pn_len"]   = df["page_part_number"].fillna("").astype(str).str.len()

    #outlier handing
    df["page_part_price"] = df["page_part_price"].clip(lower=1, upper=1_000_000)

    df_enriched = enrich_features(df)

    # print(df_enriched)

    decisions, reasons = [], []
    for _, r in df_enriched.iterrows():
        d, why = apply_hard_rules(r)
        decisions.append(d)
        reasons.append(why)

    df_enriched["hard_decision"] = decisions
    df_enriched["hard_reason"] = reasons
    print(df_enriched.head())
    df_enriched.to_csv(output_file_path, index=False)
    return df_enriched

