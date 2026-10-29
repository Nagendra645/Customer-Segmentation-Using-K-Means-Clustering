"""
clustering.py
=============
Module implementing the K-Means clustering algorithm, Elbow Method,
Silhouette Score analysis, cluster profiling, and business persona mapping.

Author: Academic ML Mentor
Course: University-Level Machine Learning
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def explain_kmeans_theory() -> str:
    """
    Returns an academic summary explaining the mathematical foundations
    of the K-Means clustering algorithm.
    """
    theory_text = """
    ========================================================================
                      K-MEANS CLUSTERING ALGORITHM FOUNDATIONS
    ========================================================================
    1. Core Mechanism:
       K-Means is an iterative, partition-based unsupervised learning algorithm.
       Given a dataset X of N observations, it partitions them into K distinct,
       non-overlapping clusters (C_1, C_2, ..., C_K).

    2. Key Terminology:
       - Cluster: A collection of data points grouped together based on similarity.
       - Centroid (mu_k): The geometric center of a cluster, computed as the arithmetic
         mean of all points assigned to that cluster:
             mu_k = (1 / |C_k|) * sum_{x in C_k} x
       - Within-Cluster Sum of Squares (WCSS / Inertia): The sum of squared Euclidean
         distances between each point and its assigned cluster centroid:
             WCSS = sum_{k=1}^K sum_{x in C_k} ||x - mu_k||^2

    3. Algorithm Workflow:
       Step 1: Initialize K centroids (using 'k-means++' to avoid poor local minima).
       Step 2: Assign each data point to its nearest centroid using Euclidean distance:
               c(i) = argmin_k ||x_i - mu_k||^2
       Step 3: Recompute the centroid of each cluster based on current assignments.
       Step 4: Repeat Steps 2 and 3 until convergence (centroids no longer shift, or
               maximum iterations are reached).
    ========================================================================
    """
    return theory_text


def compute_elbow(
    X_scaled: np.ndarray,
    k_range: range = range(1, 11),
    random_state: int = 42
) -> tuple[list[int], list[float]]:
    """
    Computes WCSS (Inertia) for candidate values of K to apply the Elbow Method.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range, default=range(1, 11)
        Range of K values to evaluate.
    random_state : int, default=42
        Seed for centroid initialization reproducibility.

    Returns:
    --------
    tuple[list[int], list[float]]
        List of evaluated K values and their corresponding inertia values.
    """
    print("\n[INFO] Evaluating Elbow Method across K =", list(k_range))
    inertias = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        kmeans.fit(X_scaled)
        inertias.append(float(kmeans.inertia_))
        print(f"  - K = {k:>2}: Inertia (WCSS) = {kmeans.inertia_:10.3f}")

    return k_values, inertias


def compute_silhouette_scores(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42
) -> tuple[list[int], list[float]]:
    """
    Computes the Mean Silhouette Coefficient for candidate values of K.

    Mathematical Definition:
    ------------------------
    For each sample i:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
        a(i) = mean intra-cluster distance (cohesion)
        b(i) = mean nearest-cluster distance (separation)
    The overall Silhouette Score is the mean over all samples.
    Range: [-1, +1]. Values closer to +1 indicate dense, well-separated clusters.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range, default=range(2, 11)
        Candidate K values (Silhouette requires at least 2 clusters).
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    tuple[list[int], list[float]]
        List of candidate K values and their corresponding silhouette scores.
    """
    print("\n[INFO] Evaluating Silhouette Scores across K =", list(k_range))
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(float(score))
        print(f"  - K = {k:>2}: Silhouette Score = {score:.4f}")

    return k_values, silhouette_scores


def train_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> tuple[KMeans, np.ndarray]:
    """
    Trains the final K-Means model with chosen cluster count K.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_clusters : int, default=5
        Chosen number of clusters.
    random_state : int, default=42
        Seed for centroid initialization reproducibility.

    Returns:
    --------
    tuple[KMeans, np.ndarray]
        Fitted KMeans model and array of predicted cluster assignments (0 to K-1).
    """
    print(f"\n[INFO] Training final K-Means model with K = {n_clusters} clusters...")
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = model.fit_predict(X_scaled)
    print(f"[SUCCESS] Model training complete. Final Inertia: {model.inertia_:.3f}")
    return model, labels


def get_cluster_centroids(
    model: KMeans,
    scaler: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts cluster centroids in both standardized scale and original feature scale.

    Parameters:
    -----------
    model : KMeans
        Trained KMeans estimator.
    scaler : StandardScaler
        Fitted scaler used for inverse transformation.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Centroids in scaled space and centroids inverse-transformed to original units.
    """
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    return centroids_scaled, centroids_original


def assign_clusters(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Attaches cluster label identifiers to the customer DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Original customer dataset.
    cluster_labels : np.ndarray
        Integer array of cluster predictions.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with 'Cluster' column.
    """
    df_result = df.copy()
    df_result["Cluster"] = cluster_labels
    return df_result


def generate_cluster_summary(
    df_segmented: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Calculates detailed descriptive metrics (count, percentage, mean, median,
    min, max) for each cluster.

    Parameters:
    -----------
    df_segmented : pd.DataFrame
        Customer DataFrame containing 'Cluster' column.
    feature_cols : list
        List of numerical feature column names analyzed.

    Returns:
    --------
    pd.DataFrame
        Structured summary DataFrame aggregated by cluster.
    """
    total_customers = len(df_segmented)
    summary_rows = []

    for cluster_id in sorted(df_segmented["Cluster"].unique()):
        cluster_df = df_segmented[df_segmented["Cluster"] == cluster_id]
        count = len(cluster_df)
        pct = (count / total_customers) * 100

        row = {
            "Cluster": cluster_id,
            "Customer Count": count,
            "Percentage (%)": round(pct, 2)
        }

        for col in feature_cols:
            row[f"{col} Mean"] = round(cluster_df[col].mean(), 2)
            row[f"{col} Median"] = round(cluster_df[col].median(), 2)
            row[f"{col} Min"] = round(cluster_df[col].min(), 2)
            row[f"{col} Max"] = round(cluster_df[col].max(), 2)

        if "Age" in df_segmented.columns and "Age" not in feature_cols:
            row["Age Mean"] = round(cluster_df["Age"].mean(), 2)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def interpret_and_label_clusters(
    summary_df: pd.DataFrame,
    income_col_mean: str = "Annual Income (k$) Mean",
    spending_col_mean: str = "Spending Score (1-100) Mean"
) -> tuple[pd.DataFrame, dict]:
    """
    Examines empirical cluster averages and assigns descriptive, evidence-based
    business labels and tailored marketing recommendations.

    Important Academic Note:
    ------------------------
    K-Means is an unsupervised algorithm that only outputs numerical cluster
    indices (0, 1, 2, ...). Business interpretations and personas are derived
    post-hoc by domain experts by analyzing the centroid positions and feature
    distributions of each partition.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        Cluster summary table with feature means.
    income_col_mean : str
        Column name for mean income in summary_df.
    spending_col_mean : str
        Column name for mean spending score in summary_df.

    Returns:
    --------
    tuple[pd.DataFrame, dict]
        Updated summary table with 'Segment Label' and dictionary of marketing strategies.
    """
    median_income = summary_df[income_col_mean].median()
    median_spending = summary_df[spending_col_mean].median()

    segment_labels = []
    strategies = {}

    for _, row in summary_df.iterrows():
        cid = int(row["Cluster"])
        inc = row[income_col_mean]
        spn = row[spending_col_mean]

        # Classify based on empirical centroid position relative to thresholds
        # Standard Mall Customers centers around:
        # High Income (> ~70), Low Income (< ~40), Moderate (~45-65)
        # High Spending (> ~65), Low Spending (< ~35), Moderate (~40-60)
        if inc >= 65 and spn >= 60:
            label = "High Income, High Spending"
            persona = "VIP / Elite Brand Advocates"
            strategy = (
                "Provide white-glove VIP concierge service, exclusive previews of luxury collections, "
                "loyalty rewards, and premium personalized experiences. Price elasticity is low; focus on prestige."
            )
        elif inc >= 65 and spn < 45:
            label = "High Income, Low Spending"
            persona = "Careful / Value-Conscious Savers"
            strategy = (
                "Target with high-utility, investment-grade items, cashback promotions, financial/educational "
                "marketing highlighting quality, durability, and return on investment."
            )
        elif inc < 45 and spn >= 60:
            label = "Low Income, High Spending"
            persona = "Careless / Trend Enthusiasts"
            strategy = (
                "Engage with trendy, viral, social-media-driven campaigns, flexible buy-now-pay-later (BNPL) options, "
                "flash sales, and rewards for social engagement and word-of-mouth."
            )
        elif inc < 45 and spn < 45:
            label = "Low Income, Low Spending"
            persona = "Budget / Sensible Economizers"
            strategy = (
                "Focus on essential, everyday value items, clear price discounts, bundle offers, and budget-friendly "
                "coupons. Highlight savings and practical utility."
            )
        else:
            label = "Moderate Income, Moderate Spending"
            persona = "Standard / Mainstream Customers"
            strategy = (
                "Maintain steady, multi-channel engagement with seasonal promotions, family bundles, membership points, "
                "and competitive everyday pricing across popular brand categories."
            )

        segment_labels.append(label)
        strategies[cid] = {
            "Label": label,
            "Persona": persona,
            "Strategy": strategy
        }

    summary_df["Segment Label"] = segment_labels
    return summary_df, strategies
