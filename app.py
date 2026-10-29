"""
app.py
======
Interactive Streamlit Dashboard for Customer Segmentation Using K-Means Clustering.

Author: NAGENDRA SINGH
Course: University-Level Machine Learning Assignment
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_data,
    check_and_clean_data,
    select_features,
    scale_features,
)
from src.clustering import (
    compute_elbow,
    compute_silhouette_scores,
    train_kmeans,
    get_cluster_centroids,
    assign_clusters,
    generate_cluster_summary,
    interpret_and_label_clusters,
)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation | K-Means Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-left: 5px solid #2563eb;
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_dataset():
    """Cached loader for dataset and scaling."""
    data_path = os.path.join(PROJECT_ROOT, "data", "customers.csv")
    df = load_data(data_path)
    df_clean = check_and_clean_data(df)
    feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_df, _ = select_features(df_clean, feature_cols)
    X_scaled, scaler = scale_features(X_df)
    return df_clean, feature_cols, X_scaled, scaler


def main():
    # Header Banner
    st.markdown('<div class="main-title">🛍️ Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Unsupervised Machine Learning with K-Means Clustering | Academic Project</div>', unsafe_allow_html=True)

    # Load data
    df_clean, feature_cols, X_scaled, scaler = load_and_preprocess_dataset()

    # Sidebar controls
    st.sidebar.image("https://images.unsplash.com/photo-1555421689-491a97ff2040?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.sidebar.title("🎛️ Model Controls")
    st.sidebar.markdown("Configure the K-Means hyperparameter below:")

    use_recommended = st.sidebar.checkbox("Use Recommended K (K = 5)", value=True)
    if use_recommended:
        k_clusters = 5
        st.sidebar.info("💡 **K = 5** is selected based on the Elbow inflection and peak Silhouette score (0.555).")
    else:
        k_clusters = st.sidebar.slider(
            "Select Number of Clusters (K):",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Choose K to explore different partitioning granularities."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Navigation")
    section = st.sidebar.radio(
        "Jump to Section:",
        [
            "1. Project Overview",
            "2. Dataset & Exploratory Analysis",
            "3. Elbow Method & Silhouette Analysis",
            "4. Cluster Visualization & Profiles",
            "5. Business Strategies & Marketing",
            "6. Live Customer Segment Predictor",
        ]
    )

    # -------------------------------------------------------------
    # SECTION 1: PROJECT OVERVIEW
    # -------------------------------------------------------------
    if section == "1. Project Overview":
        st.header("📌 1. Project Objective & Methodology")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### Executive Summary
            This machine learning application demonstrates **Customer Segmentation** using the **K-Means Clustering Algorithm**,
            a foundational unsupervised learning technique.

            In competitive modern retail and commerce, treating every customer identically results in wasted marketing expenditure
            and suboptimal customer satisfaction. By grouping customers based on their **Annual Income** and **Spending Score**,
            businesses can:
            - **Personalize Marketing Campaigns:** Target the right offers to the right customer profiles.
            - **Optimize Customer Lifetime Value (CLV):** Nurture high-potential customers into loyal brand advocates.
            - **Minimize Customer Churn:** Re-engage disengaged or price-sensitive shoppers with appropriate value propositions.
            - **Improve Resource Allocation:** Allocate marketing budget efficiently across high-ROI customer groups.

            #### Theoretical Foundation of K-Means:
            - **Centroid:** Arithmetic mean of all points assigned to a cluster.
            - **Objective Function:** Minimize Within-Cluster Sum of Squares (WCSS / Inertia):
            """)
            st.latex(r"\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} ||x - \mu_k||^2")
            st.markdown("""
            - **Feature Standardization:** Uses `StandardScaler` to ensure Euclidean distance is not distorted by differing feature scales.
            """)

        with col2:
            st.subheader("📊 Dataset At A Glance")
            st.metric("Total Customers", f"{len(df_clean)}")
            st.metric("Features Analyzed", "2 Key Metrics")
            st.metric("Clustering Algorithm", "K-Means++")
            st.metric("Recommended Clusters (K)", "5 Segments")
            st.metric("Peak Silhouette Score", "0.5547")

    # -------------------------------------------------------------
    # SECTION 2: DATASET & EXPLORATORY ANALYSIS
    # -------------------------------------------------------------
    elif section == "2. Dataset & Exploratory Analysis":
        st.header("📊 2. Dataset Exploration & Preprocessing")

        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Feature Distributions", "🔥 Correlation Analysis"])

        with tab1:
            st.subheader("Customer Data Records")
            st.dataframe(df_clean.head(15), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Dimensions:**", f"{df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
                st.write("**Missing Values:**", f"{df_clean.isnull().sum().sum()} null values detected")
                st.write("**Duplicate Records:**", f"{df_clean.duplicated().sum()} duplicates detected")
            with col2:
                st.write("**Statistical Summary:**")
                st.dataframe(df_clean.describe().round(2), use_container_width=True)

        with tab2:
            st.subheader("Feature Distributions")
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            sns.histplot(df_clean["Annual Income (k$)"], kde=True, ax=axes[0], color="#2b5c8f")
            axes[0].set_title("Annual Income (k$) Distribution")

            sns.histplot(df_clean["Spending Score (1-100)"], kde=True, ax=axes[1], color="#e07a5f")
            axes[1].set_title("Spending Score (1-100) Distribution")

            if "Age" in df_clean.columns:
                sns.histplot(df_clean["Age"], kde=True, ax=axes[2], color="#3d85c6")
                axes[2].set_title("Customer Age Distribution")

            st.pyplot(fig)
            plt.close()

            st.subheader("Annual Income vs. Spending Score (Raw Observations)")
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            hue_col = "Gender" if "Gender" in df_clean.columns else None
            sns.scatterplot(
                data=df_clean,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                hue=hue_col,
                palette="Set1",
                s=70,
                alpha=0.85,
                ax=ax2
            )
            ax2.set_title("Income vs. Spending Score Scatter Plot")
            st.pyplot(fig2)
            plt.close()

        with tab3:
            st.subheader("Correlation Heatmap")
            num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c.lower() not in ["customerid", "id"]]
            corr = df_clean[num_cols].corr()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, ax=ax, linewidths=1)
            ax.set_title("Pearson Correlation Matrix")
            st.pyplot(fig)
            plt.close()
            st.info("Insight: Annual Income and Spending Score exhibit very low linear correlation (r ≈ 0.01), making them orthogonal behavioral dimensions ideal for clustering.")

    # -------------------------------------------------------------
    # SECTION 3: ELBOW METHOD & SILHOUETTE ANALYSIS
    # -------------------------------------------------------------
    elif section == "3. Elbow Method & Silhouette Analysis":
        st.header("🎯 3. Determining the Optimal Number of Clusters (K)")
        st.markdown("""
        Selecting an appropriate value for $K$ is a fundamental challenge in unsupervised learning.
        We employ two complementary techniques:
        1. **The Elbow Method (Inertia / WCSS):** Identifies the point of diminishing returns where adding more clusters yields negligible reduction in variance.
        2. **Silhouette Analysis:** Quantifies cluster cohesion and separation. A higher score (closer to +1.0) reflects well-separated and cohesive clusters.
        """)

        k_elbow, inertias = compute_elbow(X_scaled, range(1, 11), random_state=42)
        k_sil, sil_scores = compute_silhouette_scores(X_scaled, range(2, 11), random_state=42)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("The Elbow Curve")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(k_elbow, inertias, marker="o", color="#1f77b4", linewidth=2)
            ax.scatter([5], [inertias[k_elbow.index(5)]], color="red", s=130, zorder=5, label="Elbow (K=5)")
            ax.set_title("Elbow Method: WCSS vs. K")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Inertia (WCSS)")
            ax.set_xticks(k_elbow)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Silhouette Score Comparison")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            colors = ["#2b5c8f" if k != 5 else "#d62728" for k in k_sil]
            ax.bar(k_sil, sil_scores, color=colors, edgecolor="black", alpha=0.85)
            ax.plot(k_sil, sil_scores, marker="o", color="#1d3557", linestyle=":")
            ax.set_title("Silhouette Scores for Candidate K")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Mean Silhouette Coefficient")
            ax.set_xticks(k_sil)
            ax.set_ylim(0, max(sil_scores) + 0.1)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            st.pyplot(fig)
            plt.close()

        st.success(f"""
        **Mathematical Conclusion:**
        - K = 5 produces a prominent elbow where the rate of inertia reduction levels off.
        - K = 5 achieves the **highest Silhouette Score (0.5547)** among all candidates $K \\in [2, 10]$.
        - This provides strong empirical justification for choosing 5 customer segments.
        """)

    # -------------------------------------------------------------
    # SECTION 4: CLUSTER VISUALIZATION & PROFILES
    # -------------------------------------------------------------
    elif section == "4. Cluster Visualization & Profiles":
        st.header(f"🔍 4. Customer Clusters Visualization (K = {k_clusters})")

        # Train model with selected K
        model, cluster_labels = train_kmeans(X_scaled, n_clusters=k_clusters, random_state=42)
        _, centroids_orig = get_cluster_centroids(model, scaler)
        df_seg = assign_clusters(df_clean, cluster_labels)
        summary_df = generate_cluster_summary(df_seg, feature_cols)
        summary_df, strategies = interpret_and_label_clusters(summary_df)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Customer Cluster Map")
            fig, ax = plt.subplots(figsize=(9, 6))
            palette = sns.color_palette("bright", n_colors=k_clusters)

            for idx, cid in enumerate(sorted(df_seg["Cluster"].unique())):
                sub = df_seg[df_seg["Cluster"] == cid]
                lbl = f"Cluster {cid}: {strategies[cid]['Label']}" if cid in strategies else f"Cluster {cid}"
                ax.scatter(
                    sub["Annual Income (k$)"],
                    sub["Spending Score (1-100)"],
                    s=70,
                    color=palette[idx],
                    label=lbl,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5
                )

            # Centroids
            for idx, cid in enumerate(sorted(df_seg["Cluster"].unique())):
                cx, cy = centroids_orig[cid, 0], centroids_orig[cid, 1]
                ax.scatter(cx, cy, s=260, marker="*", color="#ffbe0b", edgecolor="black", linewidth=1.5, zorder=10)
                ax.text(cx, cy + 2.5, f"C{cid}", fontsize=9, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

            ax.set_title(f"K-Means Clustering Result (K = {k_clusters})", fontsize=13, fontweight="bold")
            ax.set_xlabel("Annual Income (k$)")
            ax.set_ylabel("Spending Score (1-100)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Identified Clusters")
            st.pyplot(fig)
            plt.close()

            st.caption("⚠️ Note: This 2D scatter plot displays the full feature space used for clustering (Annual Income vs Spending Score).")

        with col2:
            st.subheader("Cluster Summary Table")
            display_cols = ["Cluster", "Customer Count", "Percentage (%)", "Annual Income (k$) Mean", "Spending Score (1-100) Mean", "Segment Label"]
            st.dataframe(summary_df[display_cols], use_container_width=True)

            st.subheader("Comparative Profiles Chart")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_idx = np.arange(len(summary_df))
            width = 0.35
            ax.bar(x_idx - width/2, summary_df["Annual Income (k$) Mean"], width=width, label="Income Mean", color="#2b5c8f")
            ax.bar(x_idx + width/2, summary_df["Spending Score (1-100) Mean"], width=width, label="Spending Mean", color="#e07a5f")
            ax.set_xticks(x_idx)
            ax.set_xticklabels([f"C{c}" for c in summary_df["Cluster"]])
            ax.set_ylabel("Mean Value")
            ax.set_title("Mean Income vs Spending by Cluster")
            ax.legend()
            st.pyplot(fig)
            plt.close()

    # -------------------------------------------------------------
    # SECTION 5: BUSINESS STRATEGIES & MARKETING
    # -------------------------------------------------------------
    elif section == "5. Business Strategies & Marketing":
        st.header("💼 5. Segment Personas & Targeted Marketing Strategies")
        st.markdown("""
        Clustering algorithms output mathematical partitions. Business value is unlocked when domain experts translate
        these partitions into actionable marketing personas. Below are the strategic recommendations derived from our empirical clusters:
        """)

        # Train model with K=5 for definitive recommendations
        model, cluster_labels = train_kmeans(X_scaled, n_clusters=5, random_state=42)
        df_seg = assign_clusters(df_clean, cluster_labels)
        summary_df = generate_cluster_summary(df_seg, feature_cols)
        summary_df, strategies = interpret_and_label_clusters(summary_df)

        for cid, info in strategies.items():
            cluster_data = summary_df[summary_df["Cluster"] == cid].iloc[0]
            with st.expander(f"📍 **Cluster {cid}: {info['Label']} ({info['Persona']})** — {cluster_data['Percentage (%)']}% of customers", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Customer Count:** {int(cluster_data['Customer Count'])} ({cluster_data['Percentage (%)']}%)")
                    st.markdown(f"**Mean Annual Income:** ${cluster_data['Annual Income (k$) Mean']:.1f}k")
                    st.markdown(f"**Mean Spending Score:** {cluster_data['Spending Score (1-100) Mean']:.1f} / 100")
                    if "Age Mean" in cluster_data:
                        st.markdown(f"**Mean Age:** {cluster_data['Age Mean']:.1f} years")
                with col2:
                    st.markdown("**🎯 Strategic Marketing Recommendation:**")
                    st.write(info["Strategy"])

    # -------------------------------------------------------------
    # SECTION 6: LIVE CUSTOMER SEGMENT PREDICTOR
    # -------------------------------------------------------------
    elif section == "6. Live Customer Segment Predictor":
        st.header("🔮 6. Interactive Customer Segment Predictor")
        st.markdown("Test the trained K-Means model by entering an individual customer's attributes below:")

        model, _ = train_kmeans(X_scaled, n_clusters=5, random_state=42)
        df_seg = assign_clusters(df_clean, model.labels_)
        summary_df = generate_cluster_summary(df_seg, feature_cols)
        summary_df, strategies = interpret_and_label_clusters(summary_df)

        col1, col2 = st.columns(2)

        with col1:
            input_income = st.slider("Customer Annual Income (k$):", min_value=10, max_value=150, value=75, step=1)
            input_spending = st.slider("Customer Spending Score (1-100):", min_value=1, max_value=100, value=65, step=1)

            # Standardize user input using fitted scaler
            input_raw = np.array([[input_income, input_spending]])
            input_scaled = scaler.transform(input_raw)
            pred_cluster = int(model.predict(input_scaled)[0])

        with col2:
            st.subheader("Prediction Result")
            persona_info = strategies.get(pred_cluster, {})
            st.success(f"**Predicted Segment:** Cluster {pred_cluster} — **{persona_info.get('Label', 'Unknown')}**")
            st.info(f"**Customer Persona:** {persona_info.get('Persona', 'N/A')}")
            st.markdown(f"**Recommended Action:** {persona_info.get('Strategy', 'N/A')}")


if __name__ == "__main__":
    main()
