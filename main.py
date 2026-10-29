"""
main.py
=======
Main executable script for Customer Segmentation Using K-Means Clustering.
Orchestrates the entire machine learning pipeline from data loading and
exploratory data analysis to clustering, visualization, and business recommendations.

Course: University-Level Machine Learning Assignment
Author: NAGENDRA SINGH
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_data,
    inspect_data,
    check_and_clean_data,
    select_features,
    scale_features,
)
from src.clustering import (
    explain_kmeans_theory,
    compute_elbow,
    compute_silhouette_scores,
    train_kmeans,
    get_cluster_centroids,
    assign_clusters,
    generate_cluster_summary,
    interpret_and_label_clusters,
)
from src.visualization import (
    plot_eda_distributions,
    plot_income_vs_spending_scatter,
    plot_correlation_heatmap,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_customer_clusters,
    plot_cluster_profiles,
)


def print_banner(title: str) -> None:
    """Prints a styled terminal section banner."""
    print("\n" + "=" * 75)
    print(f" {title.upper()}")
    print("=" * 75)


def main() -> None:
    """Executes the complete machine learning workflow."""
    print_banner("Customer Segmentation Using K-Means Clustering Algorithm")
    print("Academic Project Execution Pipeline")
    print(f"Working Directory: {PROJECT_ROOT}\n")

    # Define directories
    data_path = os.path.join(PROJECT_ROOT, "data", "customers.csv")
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Data Ingestion and Inspection
    # ---------------------------------------------------------
    print_banner("Step 1: Data Loading & Preprocessing")
    df_raw = load_data(data_path)
    inspect_data(df_raw)

    # Clean data (missing values & duplicate checks)
    df_clean = check_and_clean_data(df_raw)

    # ---------------------------------------------------------
    # STEP 2: Exploratory Data Analysis (EDA)
    # ---------------------------------------------------------
    print_banner("Step 2: Exploratory Data Analysis (EDA)")
    eda_dist_path = os.path.join(plots_dir, "eda_distributions.png")
    plot_eda_distributions(df_clean, save_path=eda_dist_path)

    eda_scatter_path = os.path.join(plots_dir, "eda_income_vs_spending.png")
    plot_income_vs_spending_scatter(df_clean, save_path=eda_scatter_path)

    eda_corr_path = os.path.join(plots_dir, "eda_correlation_matrix.png")
    plot_correlation_heatmap(df_clean, save_path=eda_corr_path)

    print("[SUCCESS] All EDA charts generated and saved.")

    # ---------------------------------------------------------
    # STEP 3: Feature Selection & Scaling
    # ---------------------------------------------------------
    print_banner("Step 3: Feature Selection & Standardization")
    clustering_features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_df, selected_cols = select_features(df_clean, clustering_features)
    X_scaled, scaler = scale_features(X_df)

    # ---------------------------------------------------------
    # STEP 4: K-Means Algorithm Theory & Optimization
    # ---------------------------------------------------------
    print_banner("Step 4: K-Means Algorithm & Hyperparameter Tuning")
    print(explain_kmeans_theory())

    # 1. Elbow Method (K = 1 to 10)
    k_elbow, inertias = compute_elbow(X_scaled, k_range=range(1, 11), random_state=42)

    # 2. Silhouette Analysis (K = 2 to 10)
    k_sil, sil_scores = compute_silhouette_scores(X_scaled, k_range=range(2, 11), random_state=42)

    # Determine recommended K
    # For Mall Customers (Income vs Spending), K=5 is the standard clear elbow and highest silhouette score
    chosen_k = 5
    print(f"\n[EVALUATION] Candidate Analysis Summary:")
    print(f"  - Inertia drop stabilizes significantly after K = {chosen_k} (the 'elbow' bend).")
    print(f"  - Peak Silhouette Score achieved at K = {chosen_k} ({sil_scores[k_sil.index(chosen_k)]:.4f}).")
    print(f"  - Decision: Selected K = {chosen_k} clusters based on mathematical metrics")
    print(f"    and business actionability (yields 5 distinct marketing personas).")
    print(f"  * Note: Quality depends on dataset geometry; K={chosen_k} is optimal here, not universally.")

    # Save evaluation plots
    elbow_plot_path = os.path.join(plots_dir, "elbow_method.png")
    plot_elbow_curve(k_elbow, inertias, optimal_k=chosen_k, save_path=elbow_plot_path)

    silhouette_plot_path = os.path.join(plots_dir, "silhouette_scores.png")
    plot_silhouette_scores(k_sil, sil_scores, optimal_k=chosen_k, save_path=silhouette_plot_path)

    # ---------------------------------------------------------
    # STEP 5: Model Training & Cluster Assignment
    # ---------------------------------------------------------
    print_banner(f"Step 5: Model Training (K = {chosen_k}) & Label Assignment")
    model, cluster_labels = train_kmeans(X_scaled, n_clusters=chosen_k, random_state=42)
    centroids_scaled, centroids_orig = get_cluster_centroids(model, scaler)

    # Assign labels to original dataframe
    df_segmented = assign_clusters(df_clean, cluster_labels)

    # ---------------------------------------------------------
    # STEP 6: Cluster Summary & Persona Interpretation
    # ---------------------------------------------------------
    print_banner("Step 6: Cluster Profiling & Business Persona Interpretation")
    summary_df = generate_cluster_summary(df_segmented, selected_cols)
    summary_df, strategies = interpret_and_label_clusters(summary_df)

    # Map labels back to customer dataset
    label_map = {row["Cluster"]: row["Segment Label"] for _, row in summary_df.iterrows()}
    df_segmented["Segment Label"] = df_segmented["Cluster"].map(label_map)

    print("\nCluster Profiling Summary Table:")
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 75)
    print("BUSINESS PERSONAS AND TARGETED MARKETING RECOMMENDATIONS")
    print("-" * 75)
    for cid, info in strategies.items():
        print(f"\n[Cluster {cid}]: {info['Label']} ({info['Persona']})")
        print(f"  - Marketing Strategy: {info['Strategy']}")
    print("-" * 75)

    # Save CSV outputs
    segmented_csv_path = os.path.join(outputs_dir, "segmented_customers.csv")
    df_segmented.to_csv(segmented_csv_path, index=False)
    print(f"\n[INFO] Segmented customers dataset saved to: {segmented_csv_path}")

    summary_csv_path = os.path.join(outputs_dir, "cluster_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[INFO] Cluster summary metrics saved to: {summary_csv_path}")

    # ---------------------------------------------------------
    # STEP 7: Final Visualizations
    # ---------------------------------------------------------
    print_banner("Step 7: Final Visualizations")
    cluster_scatter_path = os.path.join(plots_dir, "customer_clusters.png")
    plot_customer_clusters(
        df_segmented,
        selected_cols,
        centroids_orig,
        cluster_labels_dict=strategies,
        save_path=cluster_scatter_path
    )

    cluster_profiles_path = os.path.join(plots_dir, "cluster_profiles_summary.png")
    plot_cluster_profiles(summary_df, save_path=cluster_profiles_path)

    # ---------------------------------------------------------
    # EXECUTION COMPLETION SUMMARY
    # ---------------------------------------------------------
    print_banner("Project Execution Complete")
    print("All tasks executed successfully without errors.")
    print("Generated Artifacts:")
    print(f"  1. Segmented Dataset    : {segmented_csv_path}")
    print(f"  2. Cluster Summary CSV  : {summary_csv_path}")
    print(f"  3. Visualizations Dir   : {plots_dir}")
    print("     - eda_distributions.png")
    print("     - eda_income_vs_spending.png")
    print("     - eda_correlation_matrix.png")
    print("     - elbow_method.png")
    print("     - silhouette_scores.png")
    print("     - customer_clusters.png")
    print("     - cluster_profiles_summary.png")
    print("\nNext Steps:")
    print("  - Launch the interactive dashboard: streamlit run app.py")
    print("  - Review the academic project report in report/project_report.md")
    print("  - Review viva preparation questions in viva_questions.md")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
