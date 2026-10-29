"""
preprocessing.py
================
Module responsible for data ingestion, inspection, cleaning, feature selection,
and feature scaling for the Customer Segmentation project.

Author: Academic ML Mentor
Course: University-Level Machine Learning
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer segmentation dataset
    if the external CSV file cannot be located.

    Parameters:
    -----------
    n_samples : int, default=200
        Number of customer records to generate.
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        Synthetic customer dataset matching the Mall Customers schema.
    """
    np.random.seed(random_state)
    customer_ids = np.arange(1, n_samples + 1)
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.44, 0.56])
    ages = np.random.randint(18, 71, size=n_samples)

    # Simulate 5 realistic customer behavior clusters for income and spending score
    cluster_proportions = [0.25, 0.20, 0.20, 0.15, 0.20]
    n_per_cluster = [int(p * n_samples) for p in cluster_proportions]
    n_per_cluster[-1] = n_samples - sum(n_per_cluster[:-1])  # ensure sum == n_samples

    incomes = []
    spending_scores = []

    # 1. Low Income, High Spending
    incomes.extend(np.random.normal(loc=25, scale=5, size=n_per_cluster[0]))
    spending_scores.extend(np.random.normal(loc=80, scale=8, size=n_per_cluster[0]))

    # 2. Low Income, Low Spending
    incomes.extend(np.random.normal(loc=26, scale=6, size=n_per_cluster[1]))
    spending_scores.extend(np.random.normal(loc=20, scale=8, size=n_per_cluster[1]))

    # 3. High Income, High Spending
    incomes.extend(np.random.normal(loc=86, scale=12, size=n_per_cluster[2]))
    spending_scores.extend(np.random.normal(loc=82, scale=8, size=n_per_cluster[2]))

    # 4. High Income, Low Spending
    incomes.extend(np.random.normal(loc=88, scale=14, size=n_per_cluster[3]))
    spending_scores.extend(np.random.normal(loc=18, scale=7, size=n_per_cluster[3]))

    # 5. Moderate Income, Moderate Spending
    incomes.extend(np.random.normal(loc=55, scale=8, size=n_per_cluster[4]))
    spending_scores.extend(np.random.normal(loc=50, scale=7, size=n_per_cluster[4]))

    incomes = np.clip(np.round(incomes), 15, 140).astype(int)
    spending_scores = np.clip(np.round(spending_scores), 1, 99).astype(int)

    df_synthetic = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": genders,
        "Age": ages,
        "Annual Income (k$)": incomes,
        "Spending Score (1-100)": spending_scores
    })
    return df_synthetic


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads customer data from a CSV file. If the file is not found,
    it gracefully falls back to generating a realistic synthetic dataset
    and warns the user.

    Parameters:
    -----------
    filepath : str
        Relative or absolute path to the customers CSV dataset.

    Returns:
    --------
    pd.DataFrame
        Loaded and standardized DataFrame.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
    else:
        print(f"[WARNING] Dataset file not found at '{filepath}'.")
        print("[INFO] Generating synthetic customer dataset for complete offline execution...")
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"[INFO] Synthetic dataset successfully saved to: {filepath}")

    # Standardize column naming across variations (e.g., 'Genre' -> 'Gender')
    rename_dict = {}
    for col in df.columns:
        if col.strip().lower() in ["genre", "sex"]:
            rename_dict[col] = "Gender"
        elif "annual" in col.lower() and "income" in col.lower():
            rename_dict[col] = "Annual Income (k$)"
        elif "spending" in col.lower() and "score" in col.lower():
            rename_dict[col] = "Spending Score (1-100)"
        elif "customer" in col.lower() and "id" in col.lower():
            rename_dict[col] = "CustomerID"
        elif col.strip().lower() == "age":
            rename_dict[col] = "Age"

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Displays dataset dimensions, column names, datatypes, and first five records.

    Parameters:
    -----------
    df : pd.DataFrame
        The loaded customer DataFrame.
    """
    print("\n" + "=" * 60)
    print("                DATASET INSPECTION OVERVIEW")
    print("=" * 60)
    print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn Information & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<25}: {dtype}")

    print("\nFirst 5 Records:")
    print(df.head().to_string(index=False))

    print("\nDescriptive Summary Statistics:")
    print(df.describe().to_string())
    print("=" * 60 + "\n")


def check_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing (null) values and duplicate records,
    handling them appropriately.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw customer DataFrame.

    Returns:
    --------
    pd.DataFrame
        Cleaned customer DataFrame without duplicates or unhandled nulls.
    """
    print("[INFO] Checking for missing values and duplicates...")
    df_clean = df.copy()

    # 1. Check Missing Values
    missing_counts = df_clean.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        print(f"[WARNING] Found {total_missing} missing values:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  - {col}: {count} missing")
        # Impute numeric columns with median, categorical with mode
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"    -> Imputed {col} with median ({median_val})")
                else:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"    -> Imputed {col} with mode ({mode_val})")
    else:
        print("[SUCCESS] Zero missing values detected.")

    # 2. Check Duplicates
    num_duplicates = df_clean.duplicated().sum()
    if num_duplicates > 0:
        print(f"[WARNING] Found {num_duplicates} duplicate records. Removing duplicates...")
        df_clean.drop_duplicates(inplace=True)
        print(f"[SUCCESS] Duplicates removed. New row count: {len(df_clean)}")
    else:
        print("[SUCCESS] Zero duplicate records detected.")

    return df_clean


def select_features(
    df: pd.DataFrame,
    feature_cols: list = None
) -> tuple[pd.DataFrame, list]:
    """
    Selects relevant numerical features for clustering, excluding CustomerID
    and non-informative primary keys.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned customer dataset.
    feature_cols : list, optional
        List of column names to select for clustering.
        Default: ['Annual Income (k$)', 'Spending Score (1-100)']

    Returns:
    --------
    tuple[pd.DataFrame, list]
        Subset DataFrame containing only selected features and feature names list.
    """
    if feature_cols is None:
        feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]

    # Validate that all requested features exist in the DataFrame
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Requested feature columns {missing_cols} not present in dataset.")

    # Explicitly ensure CustomerID or identifier columns are excluded
    filtered_cols = [c for c in feature_cols if c.lower() not in ["customerid", "id"]]
    X_df = df[filtered_cols].copy()

    print(f"[INFO] Selected clustering features: {filtered_cols}")
    return X_df, filtered_cols


def scale_features(X_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Applies StandardScaler to scale features to zero mean and unit variance.

    Theoretical Significance:
    -------------------------
    K-Means is a distance-based clustering algorithm that computes the Euclidean
    distance between data points:
        d(p, q) = sqrt( sum_{i=1}^n (p_i - q_i)^2 )
    If features have drastically different magnitudes or variances (e.g., Annual Income
    in tens of thousands vs Spending Score in 1-100), the feature with larger scale
    will artificially dominate the Euclidean distance metric. StandardScaler transforms
    each feature to have:
        Mean (mu) = 0
        Standard Deviation (sigma) = 1
    This ensures that each customer attribute contributes equitably to distance calculations.

    Parameters:
    -----------
    X_df : pd.DataFrame
        Numerical features selected for clustering.

    Returns:
    --------
    tuple[np.ndarray, StandardScaler]
        Scaled numerical array (X_scaled) and fitted StandardScaler instance.
    """
    print("\n[INFO] Applying StandardScaler for feature normalization...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    print(f"  - Original Feature Means: {dict(zip(X_df.columns, np.round(X_df.mean().values, 2)))}")
    print(f"  - Scaled Feature Means:   {np.round(X_scaled.mean(axis=0), 4)} (approx 0.0)")
    print(f"  - Scaled Feature StdDev:  {np.round(X_scaled.std(axis=0), 4)} (approx 1.0)")
    print("[SUCCESS] Feature scaling completed successfully.")

    return X_scaled, scaler
