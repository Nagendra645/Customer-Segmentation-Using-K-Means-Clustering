"""
visualization.py
================
Module providing professional, publication-quality visualizations
for Exploratory Data Analysis (EDA) and K-Means Clustering Results.

Author: Academic ML Mentor
Course: University-Level Machine Learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global aesthetic style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})


def plot_eda_distributions(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plots demographic and behavioral distributions for Age, Annual Income,
    Spending Score, and Gender.

    Parameters:
    -----------
    df : pd.DataFrame
        Customer DataFrame.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis: Customer Feature Distributions", fontsize=16, fontweight="bold", y=0.98)

    # 1. Annual Income Distribution
    sns.histplot(df["Annual Income (k$)"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=15, edgecolor="black")
    axes[0, 0].set_title("Distribution of Annual Income (k$)", fontweight="semibold")
    axes[0, 0].set_xlabel("Annual Income (k$)")
    axes[0, 0].set_ylabel("Customer Count")
    median_inc = df["Annual Income (k$)"].median()
    axes[0, 0].axvline(median_inc, color="red", linestyle="--", label=f"Median: {median_inc:.1f}k$")
    axes[0, 0].legend()

    # 2. Spending Score Distribution
    sns.histplot(df["Spending Score (1-100)"], kde=True, ax=axes[0, 1], color="#e07a5f", bins=15, edgecolor="black")
    axes[0, 1].set_title("Distribution of Spending Score (1-100)", fontweight="semibold")
    axes[0, 1].set_xlabel("Spending Score (1-100)")
    axes[0, 1].set_ylabel("Customer Count")
    median_spn = df["Spending Score (1-100)"].median()
    axes[0, 1].axvline(median_spn, color="red", linestyle="--", label=f"Median: {median_spn:.1f}")
    axes[0, 1].legend()

    # 3. Age Distribution
    if "Age" in df.columns:
        sns.histplot(df["Age"], kde=True, ax=axes[1, 0], color="#3d85c6", bins=15, edgecolor="black")
        axes[1, 0].set_title("Distribution of Customer Age", fontweight="semibold")
        axes[1, 0].set_xlabel("Age (Years)")
        axes[1, 0].set_ylabel("Customer Count")
        median_age = df["Age"].median()
        axes[1, 0].axvline(median_age, color="red", linestyle="--", label=f"Median: {median_age:.1f}")
        axes[1, 0].legend()

    # 4. Gender Breakdown
    if "Gender" in df.columns:
        gender_counts = df["Gender"].value_counts()
        colors = ["#457b9d", "#e63946"]
        axes[1, 1].pie(
            gender_counts,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        axes[1, 1].set_title("Gender Breakdown of Customer Base", fontweight="semibold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] EDA distributions plot saved to: {save_path}")

    plt.close()


def plot_income_vs_spending_scatter(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Generates a scatter plot of Annual Income vs Spending Score
    with Gender distinction.

    Parameters:
    -----------
    df : pd.DataFrame
        Customer DataFrame.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    plt.figure(figsize=(9, 6))

    hue_param = "Gender" if "Gender" in df.columns else None
    palette_param = {"Male": "#1d3557", "Female": "#e63946"} if hue_param else None

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue=hue_param,
        palette=palette_param,
        s=85,
        alpha=0.85,
        edgecolor="black"
    )

    plt.title("Bivariate Analysis: Annual Income vs. Spending Score", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Annotate apparent natural clusters
    plt.annotate(
        "Low Income,\nHigh Spending",
        xy=(25, 80),
        xytext=(20, 95),
        arrowprops=dict(facecolor="gray", arrowstyle="->", lw=1),
        fontsize=9,
        fontweight="semibold"
    )
    plt.annotate(
        "High Income,\nHigh Spending",
        xy=(85, 85),
        xytext=(80, 95),
        arrowprops=dict(facecolor="gray", arrowstyle="->", lw=1),
        fontsize=9,
        fontweight="semibold"
    )
    plt.annotate(
        "Moderate Income,\nModerate Spending",
        xy=(55, 50),
        xytext=(50, 30),
        arrowprops=dict(facecolor="gray", arrowstyle="->", lw=1),
        fontsize=9,
        fontweight="semibold"
    )

    if hue_param:
        plt.legend(title="Customer Gender", frameon=True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Income vs Spending scatter plot saved to: {save_path}")

    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Generates a correlation heatmap for numerical attributes.

    Parameters:
    -----------
    df : pd.DataFrame
        Customer DataFrame.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    plt.figure(figsize=(7, 5))
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude CustomerID from correlation analysis
    num_cols = [c for c in num_cols if c.lower() not in ["customerid", "id"]]
    corr = df[num_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        vmin=-1,
        vmax=1,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Pearson Correlation"}
    )
    plt.title("Correlation Matrix of Customer Features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Correlation heatmap saved to: {save_path}")

    plt.close()


def plot_elbow_curve(
    k_values: list[int],
    inertias: list[float],
    optimal_k: int = 5,
    save_path: str = None
) -> None:
    """
    Plots the Elbow Curve (WCSS / Inertia vs. Number of Clusters K).

    Parameters:
    -----------
    k_values : list[int]
        Evaluated cluster numbers (e.g. 1 through 10).
    inertias : list[float]
        Calculated inertia values.
    optimal_k : int, default=5
        Identified elbow point to highlight.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o", markersize=8, color="#1f77b4", linewidth=2.5, label="WCSS (Inertia)")

    # Highlight chosen optimal K
    if optimal_k in k_values:
        idx = k_values.index(optimal_k)
        plt.scatter(
            [optimal_k],
            [inertias[idx]],
            color="#d62728",
            s=160,
            zorder=5,
            edgecolor="black",
            linewidth=1.5,
            label=f"Selected Elbow Point (K={optimal_k})"
        )
        plt.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.7)
        plt.annotate(
            f"Elbow Point (K={optimal_k})\nInertia = {inertias[idx]:.1f}",
            xy=(optimal_k, inertias[idx]),
            xytext=(optimal_k + 0.7, inertias[idx] + (max(inertias) * 0.08)),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=8),
            fontweight="bold",
            color="#990000"
        )

    plt.title("The Elbow Method for Optimal K", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Within-Cluster Sum of Squares (WCSS / Inertia)", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Elbow method plot saved to: {save_path}")

    plt.close()


def plot_silhouette_scores(
    k_values: list[int],
    silhouette_scores: list[float],
    optimal_k: int = 5,
    save_path: str = None
) -> None:
    """
    Plots the Silhouette Score across candidate values of K.

    Parameters:
    -----------
    k_values : list[int]
        Evaluated cluster numbers (2 through 10).
    silhouette_scores : list[float]
        Mean Silhouette Coefficients.
    optimal_k : int, default=5
        Selected cluster number to annotate.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    plt.figure(figsize=(8, 5))

    bar_colors = ["#2b5c8f" if k != optimal_k else "#d62728" for k in k_values]
    bars = plt.bar(k_values, silhouette_scores, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)

    # Add numeric labels on top of bars
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.plot(k_values, silhouette_scores, color="#1d3557", marker="o", linewidth=1.5, linestyle=":")
    plt.title("Silhouette Score Comparison Across Candidate K Values", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Mean Silhouette Coefficient", fontsize=11)
    plt.xticks(k_values)
    plt.ylim(0, max(silhouette_scores) + 0.1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotation
    if optimal_k in k_values:
        idx = k_values.index(optimal_k)
        score_opt = silhouette_scores[idx]
        plt.annotate(
            f"Chosen K={optimal_k}\n(Score={score_opt:.3f})",
            xy=(optimal_k, score_opt),
            xytext=(optimal_k + 0.6, score_opt + 0.04),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.2, headwidth=6),
            fontweight="bold",
            color="#990000"
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Silhouette scores plot saved to: {save_path}")

    plt.close()


def plot_customer_clusters(
    df_segmented: pd.DataFrame,
    feature_cols: list[str],
    centroids_original: np.ndarray,
    cluster_labels_dict: dict = None,
    save_path: str = None
) -> None:
    """
    Renders the final customer clusters in a 2D scatter plot with marked centroids.

    Academic Dimensionality Note:
    ----------------------------
    When clustering is performed on two features, this scatter plot displays the full
    clustering feature space. If three or more features are used in training, this 2D
    chart depicts a two-dimensional projection along the primary selected axes and
    does not represent all dimensions of the underlying model.

    Parameters:
    -----------
    df_segmented : pd.DataFrame
        DataFrame with cluster assignments.
    feature_cols : list[str]
        Two feature column names for axes (e.g. Annual Income and Spending Score).
    centroids_original : np.ndarray
        Unscaled cluster centroids corresponding to the two feature columns.
    cluster_labels_dict : dict, optional
        Mapping of cluster integer to descriptive business name.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    plt.figure(figsize=(10, 7))

    x_col, y_col = feature_cols[0], feature_cols[1]
    clusters = sorted(df_segmented["Cluster"].unique())

    # Distinct vibrant color palette
    palette = sns.color_palette("bright", n_colors=len(clusters))

    for idx, cid in enumerate(clusters):
        subset = df_segmented[df_segmented["Cluster"] == cid]
        label_text = f"Cluster {cid}"
        if cluster_labels_dict and cid in cluster_labels_dict:
            label_text += f": {cluster_labels_dict[cid]['Label']}"

        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=75,
            color=palette[idx],
            label=label_text,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5
        )

    # Plot Centroids prominently
    for idx, cid in enumerate(clusters):
        cx, cy = centroids_original[cid, 0], centroids_original[cid, 1]
        plt.scatter(
            cx,
            cy,
            s=280,
            marker="*",
            color="#ffbe0b",
            edgecolor="black",
            linewidth=1.8,
            zorder=10,
            label="Cluster Centroids" if idx == 0 else ""
        )
        plt.text(
            cx,
            cy + 2.5,
            f"C{cid}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray")
        )

    plt.title(
        f"Customer Segments Identified by K-Means (K={len(clusters)})",
        fontsize=14,
        fontweight="bold",
        pad=14
    )
    plt.suptitle(
        "Two-dimensional feature projection (Annual Income vs Spending Score)",
        fontsize=10,
        style="italic",
        y=0.915
    )
    plt.xlabel(x_col, fontsize=11, fontweight="semibold")
    plt.ylabel(y_col, fontsize=11, fontweight="semibold")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Position legend cleanly outside or in suitable corner
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, title="Customer Segments")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Customer clusters plot saved to: {save_path}")

    plt.close()


def plot_cluster_profiles(
    summary_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Renders a grouped comparative bar chart comparing mean Annual Income
    and mean Spending Score across all discovered clusters.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        Cluster summary table with feature means and labels.
    save_path : str, optional
        Filepath to save the plot figure.
    """
    plt.figure(figsize=(10, 6))

    x_indices = np.arange(len(summary_df))
    bar_width = 0.35

    income_means = summary_df["Annual Income (k$) Mean"]
    spending_means = summary_df["Spending Score (1-100) Mean"]

    plt.bar(
        x_indices - bar_width / 2,
        income_means,
        width=bar_width,
        label="Mean Annual Income (k$)",
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.9
    )
    plt.bar(
        x_indices + bar_width / 2,
        spending_means,
        width=bar_width,
        label="Mean Spending Score (1-100)",
        color="#e07a5f",
        edgecolor="black",
        alpha=0.9
    )

    # Add value annotations on bars
    for i in x_indices:
        plt.text(i - bar_width / 2, income_means.iloc[i] + 1.2, f"{income_means.iloc[i]:.1f}", ha="center", fontsize=9)
        plt.text(i + bar_width / 2, spending_means.iloc[i] + 1.2, f"{spending_means.iloc[i]:.1f}", ha="center", fontsize=9)

    tick_labels = [f"Cluster {c}\n({row.get('Segment Label', '')[:15]})" for c, row in summary_df.iterrows()]
    plt.xticks(x_indices, tick_labels, rotation=0, fontsize=9.5)
    plt.ylabel("Mean Score / Income (k$)", fontsize=11)
    plt.title("Cluster Comparison: Mean Annual Income vs. Mean Spending Score", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Cluster profiles summary plot saved to: {save_path}")

    plt.close()
