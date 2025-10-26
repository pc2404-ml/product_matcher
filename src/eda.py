"""
Exploratory Data Analysis (EDA) for Product Matcher Dataset
-----------------------------------------------------------

This module provides `eda_plots`, which:
1. Loads raw dataset (JSON format).
2. Performs basic checks (shape, missing values, duplicates).
3. Generates descriptive statistics and sanity-check prints.
4. Creates diagnostic plots and saves them to `output_stat_path`.

Plots produced:
---------------
- Client vs Page Price distributions (histograms, KDE, log scale, boxplots).
- Label distribution (0 = non-match, 1 = match).
- Part number length distributions (client vs page).
- Scatter plots of prices for matches vs non-matches.
- Manufacturer, part number, and product group match rates.
- Top 10 manufacturers (client vs page).
- Suspicious records (large price differences but labeled as match).

Artifacts:
----------
All plots are saved as `.png` files under the provided `output_stat_path`.
Suspicious record samples are printed to console for quick inspection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def eda_plots(input_path,output_stat_path):
    """
       Perform exploratory data analysis on the dataset and save plots.

       Parameters
       ----------
       input_path : str
           Path to the input JSON file containing the dataset.
       output_stat_path : str
           Directory where output plots/statistics will be saved.
       """

    # ------------------------------------------------------------------
    # 1. Load and Inspect Dataset
    # ------------------------------------------------------------------
    df = pd.read_json(input_path)
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print(df.head())
    print(df.describe(include="all"))
    print(df.info())
    print("Missing values per column:\n", df.isnull().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # ------------------------------------------------------------------
    # 2. Basic Label Counts
    # ------------------------------------------------------------------
    n_matches = (df["label"] == 1).sum()
    print(f"Number of matches: {n_matches}")
    print(f"Number of non-matches: {len(df) - n_matches}")

    # ------------------------------------------------------------------
    # 3. Price Distributions
    # ------------------------------------------------------------------
    plt.figure(figsize=(12,5))
    # Client prices
    plt.subplot(1,2,1)
    df["client_part_price"].dropna().hist(bins=50)
    plt.title("Distribution of Client Part Prices")
    plt.xlabel("Price")
    plt.ylabel("Count")
    
    # Page prices
    plt.subplot(1,2,2)
    df["page_part_price"].dropna().hist(bins=50, color="orange")
    plt.title("Distribution of Page Part Prices")
    plt.xlabel("Price")
    plt.ylabel("Count")
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_stat_path+'/ClientVsPagePrice.png')
    
    
    # Count labels
    label_counts = df["label"].value_counts()
    
    # Bar plot
    plt.figure(figsize=(5,4))
    label_counts.plot(kind="bar", color=["orange","skyblue"])
    plt.title("Distribution of Labels")
    plt.xlabel("Label (0 = No Match, 1 = Match)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig(output_stat_path+'/LabelCount.png')
    
    # ------------------------------------------------------------------
    # 4. Part Number Lengths
    # ------------------------------------------------------------------
    df["client_pn_length"] = df["client_part_number"].fillna("").astype(str).str.len()
    df["page_pn_length"]   = df["page_part_number"].fillna("").astype(str).str.len()
    
    # Plot histograms side by side
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    df["client_pn_length"].hist(bins=30, color="skyblue")
    plt.title("Client Part Number Lengths")
    plt.xlabel("Length of string")
    plt.ylabel("Count")
    
    plt.subplot(1,2,2)
    df["page_pn_length"].hist(bins=30, color="orange")
    plt.title("Page Part Number Lengths")
    plt.xlabel("Length of string")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_stat_path+'/ClientVsPagePartLength.png')
    
    
    plt.figure(figsize=(12,5))

    # ------------------------------------------------------------------
    # 5. Advanced Price Distributions (KDE, log, boxplot)
    # ------------------------------------------------------------------
    plt.subplot(1,2,1)
    df["client_part_price"].dropna().plot(kind="hist", bins=50, density=True, alpha=0.5, color="blue")
    df["client_part_price"].dropna().plot(kind="kde")
    plt.title("Client Part Price Distribution")
    
    # Page prices
    plt.subplot(1,2,2)
    df["page_part_price"].dropna().plot(kind="hist", bins=50, density=True, alpha=0.5, color="orange")
    df["page_part_price"].dropna().plot(kind="kde", color="red")
    plt.title("Page Part Price Distribution")
    plt.savefig(output_stat_path+'/ClientVsPagePriceDistribution.png')
    
    #  Log-transformed histograms
    plt.subplot(1,2,1)
    df["client_part_price"].dropna().hist(bins=50, color="skyblue")
    plt.title("Client Part Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    
    # Page prices
    plt.subplot(1,2,2)
    df["page_part_price"].dropna().hist(bins=50, color="orange")
    plt.title("Page Part Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_stat_path+'/ClientVsPagePriceDistributionNormalised.png')
    

    # Client prices
    plt.subplot(1,2,1)
    np.log1p(df["client_part_price"].dropna()).hist(bins=50, color="skyblue")
    plt.title("Client Prices (log scale)")
    plt.xlabel("log(1+price)")
    plt.ylabel("Count")
    
    # Page prices
    plt.subplot(1,2,2)
    np.log1p(df["page_part_price"].dropna()).hist(bins=50, color="orange")
    plt.title("Page Prices (log scale)")
    plt.xlabel("log(1+price)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_stat_path+'/ClientVsPagePriceDistributionLogNormalised.png')

    # Boxplot spread
    plt.figure(figsize=(8,5))
    plt.boxplot(
        [df["client_part_price"].dropna(), df["page_part_price"].dropna()],
        labels=["Client", "Page"],
        showfliers=False   # hides extreme dots so you can see the box
    )
    plt.title("Price Spread: Client vs Page")
    plt.ylabel("Price")
    plt.savefig(output_stat_path+'/ClientVsPagePriceDistributionBoxPlot.png')
    
    df["price_diff_rel"] = abs(df["client_part_price"] - df["page_part_price"]) / \
                           df[["client_part_price", "page_part_price"]].max(axis=1)
    
    plt.figure(figsize=(8,5))
    df.boxplot(column="price_diff_rel", by="label")
    plt.title("Relative Price Difference by Match/Non-Match")
    plt.suptitle("")   # remove automatic pandas title
    plt.xlabel("Label (0 = No Match, 1 = Match)")
    plt.ylabel("Relative Price Difference")
    plt.savefig(output_stat_path+'/PriceDifference.png')

    # ------------------------------------------------------------------
    # 6. Scatter Plots: Matches vs Non-Matches
    # ------------------------------------------------------------------
    #Matches only
    df_match = df[df["label"] == 1]
    
    plt.figure(figsize=(6,6))
    plt.scatter(df_match["client_part_price"], df_match["page_part_price"],
                alpha=0.5, color="green")
    
    plt.title("Client vs Page Price (Matches only)")
    plt.xlabel("Client Price")
    plt.ylabel("Page Price")
    plt.xlim(0, 2000)   # adjust limits for readability
    plt.ylim(0, 2000)
    plt.plot([0,2000], [0,2000], color="red", linestyle="--")  # perfect equality line
    plt.savefig(output_stat_path+'/ClientVsPagePriceMatched.png')
    
    
    corr = df_match[["client_part_price", "page_part_price"]].corr()
    print(corr)
    
    df_match["price_diff_rel"] = abs(df_match["client_part_price"] - df_match["page_part_price"]) / \
                                 df_match[["client_part_price","page_part_price"]].max(axis=1)
    
    print(df_match["price_diff_rel"].describe())
    
    
    # Filter for non-matches
    df_nonmatch = df[df["label"] == 0]
    
    plt.figure(figsize=(6,6))
    plt.scatter(df_nonmatch["client_part_price"],
                df_nonmatch["page_part_price"],
                alpha=0.5, color="red")
    
    plt.title("Client vs Page Price (Non-Matches only)")
    plt.xlabel("Client Price")
    plt.ylabel("Page Price")
    plt.xlim(0, 2000)   # same limits as match plot for fair comparison
    plt.ylim(0, 2000)
    
    # Add diagonal line for reference
    plt.plot([0,2000], [0,2000], color="black", linestyle="--")
    plt.savefig(output_stat_path+'/ClientVsPagePriceNonMatched.png')

    # ------------------------------------------------------------------
    # 7. Manufacturer, PN, Group Match Rates
    # ------------------------------------------------------------------

    # Create a boolean column: manufacturer exact match
    df["manu_match"] = (df["client_part_manufacturer"].fillna("").str.upper()
                        == df["page_part_manufacturer"].fillna("").str.upper())
    
    # Group by label and compute mean (True=1, False=0 → mean = match rate)
    manu_match_rate = df.groupby("label")["manu_match"].mean()
    print(manu_match_rate)
    
    df["pn_match"] = (df["client_part_number"].fillna("").str.upper()
                      == df["page_part_number"].fillna("").str.upper())
    
    pn_match_rate = df.groupby("label")["pn_match"].mean()
    print(pn_match_rate)
    
    
    df["group_match"] = (df["client_part_product_group"].fillna("").str.upper() ==
                         df["page_part_product_group"].fillna("").str.upper())
    
    print(df.groupby("label")["group_match"].mean())
    
    
    # Ensure strings, replace NaN with empty
    df["client_pn_len"] = df["client_part_number"].fillna("").astype(str).str.len()
    df["page_pn_len"]   = df["page_part_number"].fillna("").astype(str).str.len()
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    df["client_pn_len"].hist(bins=30, color="skyblue")
    plt.title("Client Part Number Lengths")
    plt.xlabel("Length of client part number")
    plt.ylabel("Count")
    
    plt.subplot(1,2,2)
    df["page_pn_len"].hist(bins=30, color="orange")
    plt.title("Page Part Number Lengths")
    plt.xlabel("Length of page part number")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_stat_path+'/LengthHistogram.png')

    # ------------------------------------------------------------------
    # 8. Top Manufacturers
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    
    df["client_part_manufacturer"].value_counts().head(10).plot(
        kind="bar", ax=axes[0], color="skyblue", title="Client Manufacturers (Top 10)")
    
    df["page_part_manufacturer"].value_counts().head(10).plot(
        kind="bar", ax=axes[1], color="orange", title="Page Manufacturers (Top 10)")
    
    plt.tight_layout()
    plt.savefig(output_stat_path+'/ClientVsPageManufacturers.png')
    
    
    df["price_diff_rel"] = abs(df["client_part_price"] - df["page_part_price"]) / \
                           df[["client_part_price","page_part_price"]].max(axis=1)

    # ------------------------------------------------------------------
    # 9. Suspicious Matches (label=1 but big price diff)
    # ------------------------------------------------------------------

    # Look at matched pairs with big price differences
    suspicious = df[(df["label"]==1) & (df["price_diff_rel"] > 0.8)]
    print("Suspicious matches (label=1 but prices very different):\n",
          suspicious[["client_part_number","client_part_price","page_part_number","page_part_price","label"]].head())
    plt.figure(figsize=(6,5))
    plt.scatter(df["client_part_price"], df["page_part_price"], alpha=0.3, c=df["label"], cmap="coolwarm")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Client price (log)")
    plt.ylabel("Page price (log)")
    plt.title("Price relationship (log scale, colored by label)")
    plt.savefig(output_stat_path+'/SuspiciousRecordsDistribution.png')
    plt.close()
    print("EDA complete. Plots saved to:", output_stat_path)
