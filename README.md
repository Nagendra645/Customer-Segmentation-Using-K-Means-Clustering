# Customer Segmentation Using K-Means Clustering Algorithm

A comprehensive, university-level machine learning assignment project developed in Python that applies the **K-Means Clustering** unsupervised learning algorithm to segment retail mall customers based on their annual income and spending behavior.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [Dataset Description](#-dataset-description)
5. [Project Architecture](#-project-architecture)
6. [Installation & Setup](#-installation--setup)
7. [Running the Project](#-running-the-project)
8. [Generated Outputs & Results](#-generated-outputs--results)
9. [Identified Customer Segments](#-identified-customer-segments)
10. [Troubleshooting Guide](#-troubleshooting-guide)
11. [Academic Documentation](#-academic-documentation)

---

## 📌 Project Overview

In competitive retail environments, applying a uniform marketing strategy to all customers leads to inefficient budget allocation and low conversion rates. **Customer Segmentation** partitions a heterogeneous customer base into distinct, homogeneous subgroups sharing similar purchasing behaviors.

This project implements an end-to-end unsupervised machine learning workflow:
- **Data Ingestion & Cleaning:** Handles missing data, removes duplicates, and filters non-informative primary keys (`CustomerID`).
- **Feature Scaling:** Uses `StandardScaler` to normalize feature magnitudes so that Euclidean distance calculations are unbiased.
- **Hyperparameter Optimization:** Utilizes both the **Elbow Method** (WCSS / Inertia) and **Silhouette Analysis** to empirically identify the optimal number of clusters ($K=5$).
- **Cluster Profiling & Persona Mapping:** Translates mathematical clusters into actionable commercial personas and provides targeted marketing strategies.
- **Interactive Dashboard:** Features a full-fledged **Streamlit** web application with interactive hyperparameter tuning and a live customer segment predictor.

---

## 🚀 Key Features

- **Beginner-Friendly & Modular:** Clean, object-oriented, and modular architecture split into `src/preprocessing.py`, `src/clustering.py`, and `src/visualization.py`.
- **Failsafe Offline Execution:** If the dataset is moved or internet is unavailable, an automated synthetic generator seamlessly creates a realistic fallback dataset.
- **Publication-Grade Visualizations:** Generates 300 DPI figures using Matplotlib and Seaborn with clean typography, clear legends, annotated centroids, and custom palettes.
- **Empirical Rigor:** Combines Inertia and Silhouette Coefficient ($s = 0.5547$) to rigorously defend the cluster count without making unsubstantiated optimality claims.
- **Interactive Streamlit App:** Allows real-time experimentation with $K$, interactive cluster mapping, and individual customer persona prediction.
- **Complete Academic Report & Viva Prep:** Includes a 23-section university report and 15+ comprehensive viva questions with concise, accurate answers.

---

## 💻 Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Primary programming language |
| **Pandas** | Tabular data manipulation and summary aggregations |
| **NumPy** | Numerical computations and vector operations |
| **Scikit-learn** | K-Means clustering, `StandardScaler`, and `silhouette_score` |
| **Matplotlib & Seaborn** | Publication-quality statistical visualizations |
| **Streamlit** | Interactive web dashboard and live predictor interface |

---

## 📂 Dataset Description

- **Source:** Canonical Mall Customers Dataset (widely utilized benchmark in unsupervised learning).
- **Location:** `data/customers.csv`
- **Total Records:** 200 customer entries
- **Attributes:**
  - `CustomerID`: Unique numerical identifier assigned to each customer (excluded from clustering).
  - `Gender`: Biological gender (`Male`, `Female`).
  - `Age`: Customer age in years (Range: 18 – 70).
  - `Annual Income (k$)`: Customer annual income in thousands of USD (Range: 15k$ – 137k$).
  - `Spending Score (1-100)`: Proprietary behavioral score assigned by mall management based on transaction history and shopping frequency (Range: 1 – 99).

> **Academic Note on Limitations:** The dataset is cross-sectional and represents 200 customers from a single mall. It does not account for temporal shifts, inflation, e-commerce transactions, or specific product categories.

---

## 📁 Project Architecture

```
customer_segmentation_kmeans/
│
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # Command-line end-to-end execution pipeline
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation & instructions
├── viva_questions.md           # University oral exam / viva preparation guide
│
├── data/
│   └── customers.csv           # Cleaned customer dataset (200 records)
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── preprocessing.py        # Ingestion, cleaning, feature selection, scaling
│   ├── clustering.py           # K-Means, Elbow method, Silhouette analysis, profiling
│   └── visualization.py        # Publication-ready Matplotlib & Seaborn plots
│
├── outputs/
│   ├── cluster_summary.csv     # Summary statistics per cluster
│   ├── segmented_customers.csv # Dataset with assigned Cluster and Segment Label
│   └── plots/
│       ├── eda_distributions.png          # Distributions of Income, Spending, Age, Gender
│       ├── eda_income_vs_spending.png      # Income vs Spending scatter plot
│       ├── eda_correlation_matrix.png      # Pearson correlation matrix heatmap
│       ├── elbow_method.png                # WCSS vs K curve with annotated elbow
│       ├── silhouette_scores.png           # Silhouette score comparison across K
│       ├── customer_clusters.png           # 2D customer clusters with centroids
│       └── cluster_profiles_summary.png    # Comparative bar chart of cluster means
│
└── report/
    └── project_report.md       # Comprehensive 23-section academic project report
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed. Check via Windows Command Prompt or PowerShell:
```powershell
python --version
pip --version
```

### 2. Navigate to Project Directory
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\customer_segmentation_kmeans"
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A: Run the Automated CLI Pipeline
Executes data preprocessing, exploratory analysis, hyperparameter tuning, model training, and generates all summary tables and plots:
```powershell
python main.py
```

### Option B: Launch the Interactive Streamlit Dashboard
Launches the browser dashboard for visual exploration and live testing:
```powershell
streamlit run app.py
```
*(The dashboard will open automatically in your browser at `http://localhost:8501`)*

---

## 📊 Generated Outputs & Results

Running `main.py` generates the following files:

| File / Artifact | Location | Description |
| :--- | :--- | :--- |
| `segmented_customers.csv` | `outputs/` | Complete customer dataset annotated with assigned cluster IDs and descriptive business labels. |
| `cluster_summary.csv` | `outputs/` | Mean, median, min, max, count, and percentage breakdown for each cluster. |
| `eda_distributions.png` | `outputs/plots/` | Multi-panel distribution plots of numerical and categorical variables. |
| `eda_income_vs_spending.png` | `outputs/plots/` | Scatter plot demonstrating natural customer groupings. |
| `eda_correlation_matrix.png` | `outputs/plots/` | Feature correlation matrix proving feature independence. |
| `elbow_method.png` | `outputs/plots/` | Inertia vs K plot showing the inflection at $K=5$. |
| `silhouette_scores.png` | `outputs/plots/` | Silhouette score bar chart confirming $K=5$ as the peak ($0.5547$). |
| `customer_clusters.png` | `outputs/plots/` | 2D scatter plot of final clusters with prominent centroid markers. |
| `cluster_profiles_summary.png`| `outputs/plots/` | Comparative bar chart showing Income vs Spending across clusters. |

---

## 👥 Identified Customer Segments

Based on empirical clustering results ($K=5$):

| Cluster | Segment Label | Customer Share | Mean Income | Mean Spending | Business Persona | Recommended Marketing Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **0** | **Moderate Income, Moderate Spending** | 40.5% (81) | \$55.3k | 49.5 / 100 | *Standard / Mainstream* | Everyday value, seasonal sales, loyalty points, family bundles. |
| **1** | **High Income, High Spending** | 19.5% (39) | \$86.5k | 82.1 / 100 | *VIP / Elite Brand Advocates* | White-glove concierge, luxury previews, exclusive VIP loyalty perks. |
| **2** | **Low Income, High Spending** | 11.0% (22) | \$25.7k | 79.4 / 100 | *Careless / Trend Enthusiasts* | Viral social campaigns, trendy fashion drops, Buy-Now-Pay-Later (BNPL). |
| **3** | **High Income, Low Spending** | 17.5% (35) | \$88.2k | 17.1 / 100 | *Careful / Wealth Accumulators* | High-utility, premium durable goods, cashback, ROI-focused marketing. |
| **4** | **Low Income, Low Spending** | 11.5% (23) | \$26.3k | 20.9 / 100 | *Budget / Sensible Economizers* | Deep clearance discounts, value essentials, utility coupons. |

---

## 🔧 Troubleshooting Guide

- **ModuleNotFoundError: No module named 'src'**
  - Run the scripts directly from the project root directory or ensure `sys.path.insert(0, PROJECT_ROOT)` is present in your execution script.
- **Port 8501 Already in Use (Streamlit)**
  - Run streamlit with a different port:
    ```powershell
    streamlit run app.py --server.port 8502
    ```
- **Matplotlib GUI Backend Warnings on Windows**
  - All scripts explicitly call `plt.close()` after saving figures to avoid memory leaks or threading conflicts in headless/terminal environments.
- **Dataset Missing**
  - The script automatically generates a realistic synthetic dataset if `data/customers.csv` is absent, guaranteeing zero runtime interruptions.

---

## 📚 Academic Documentation

- **Full Project Report:** Located at [`report/project_report.md`](report/project_report.md) — includes complete mathematical proofs, literature review, methodology, results, and citations formatted for academic submission.
- **Viva Questions & Answers:** Located at [`viva_questions.md`](viva_questions.md) — contains 18 comprehensive viva questions with concise, technically rigorous answers for exam defense.

---

## 👨‍💻 Author

**NAGENDRA SINGH**  
- **GitHub:** [@Nagendra645](https://github.com/Nagendra645)  
- **Project Repository:** [Customer-Segmentation-Using-K-Means-Clustering](https://github.com/Nagendra645/Customer-Segmentation-Using-K-Means-Clustering)

