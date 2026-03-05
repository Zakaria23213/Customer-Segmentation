"""
clustering.py
=============
Supports four algorithms:
  1. K-Means
  2. Agglomerative (Hierarchical)
  3. DBSCAN
  4. Gaussian Mixture Model (GMM)

Each fit_* function returns a labels array aligned with rfm_scaled rows.
DBSCAN may return -1 labels (noise points).

compare_algorithms() runs all four and prints a metric table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cluster          import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture          import GaussianMixture
from sklearn.metrics          import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy  import dendrogram, linkage

sns.set_theme(style="whitegrid", palette="muted")
FEATURES = ["Recency", "Frequency", "Monetary"]
PALETTE  = sns.color_palette("tab10")


# ══════════════════════════════════════════════════════════════════════════════
# 1. K-MEANS
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_k(rfm_scaled: pd.DataFrame, k_range: range = range(2, 11),
                   save_path: str = None) -> None:
    X = rfm_scaled[FEATURES].values
    inertias, silhouettes = [], []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias,    marker="o", color="#4C72B0", linewidth=2)
    ax1.set_title("Elbow Method — Inertia",  fontsize=13, fontweight="bold")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")

    ax2.plot(list(k_range), silhouettes, marker="s", color="#C44E52", linewidth=2)
    ax2.set_title("Silhouette Score",         fontsize=13, fontweight="bold")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")

    plt.suptitle("K-Means: Choosing Optimal K", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def fit_kmeans(rfm_scaled: pd.DataFrame, n_clusters: int) -> np.ndarray:
    X      = rfm_scaled[FEATURES].values
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    print(f"  K-Means (k={n_clusters}) silhouette: {silhouette_score(X, labels):.4f}")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGGLOMERATIVE (HIERARCHICAL)
# ══════════════════════════════════════════════════════════════════════════════

def plot_dendrogram(rfm_scaled: pd.DataFrame, max_samples: int = 500,
                    save_path: str = None) -> None:
    """
    Plot a dendrogram on a random sample (full dataset is too large).
    Use this to visually pick the number of clusters before fitting.
    """
    X = rfm_scaled[FEATURES].values
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        X   = X[idx]

    Z = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=90, leaf_font_size=9, color_threshold=0)
    ax.set_title(f"Hierarchical Clustering Dendrogram (sample n={max_samples})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster / Sample index")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def fit_agglomerative(rfm_scaled: pd.DataFrame, n_clusters: int) -> np.ndarray:
    X      = rfm_scaled[FEATURES].values
    model  = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)
    print(f"  Agglomerative (k={n_clusters}) silhouette: {silhouette_score(X, labels):.4f}")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 3. DBSCAN
# ══════════════════════════════════════════════════════════════════════════════

def fit_dbscan(rfm_scaled: pd.DataFrame, eps: float = 0.5,
               min_samples: int = 10) -> np.ndarray:
    """
    DBSCAN — automatically finds clusters and marks outliers as -1.
    Tune eps (neighbourhood radius) and min_samples for your data density.
    A k-distance plot (below) helps choose eps.
    """
    X      = rfm_scaled[FEATURES].values
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  DBSCAN (eps={eps}, min_samples={min_samples})")
    print(f"    Clusters found : {n_clusters}")
    print(f"    Noise points   : {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    if n_clusters > 1:
        # Silhouette only valid when >1 cluster and excluding noise
        mask = labels != -1
        if mask.sum() > 1:
            print(f"    Silhouette (excl. noise): {silhouette_score(X[mask], labels[mask]):.4f}")
    return labels


def plot_kdistance(rfm_scaled: pd.DataFrame, k: int = 5,
                   save_path: str = None) -> None:
    """
    K-distance plot to help choose eps for DBSCAN.
    Look for the 'elbow' — that distance is a good eps value.
    """
    from sklearn.neighbors import NearestNeighbors
    X  = rfm_scaled[FEATURES].values
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances     = np.sort(distances[:, k - 1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color="#4C72B0", linewidth=1.5)
    ax.set_title(f"K-Distance Plot (k={k}) — find the elbow for eps",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{k}-NN Distance")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. GAUSSIAN MIXTURE MODEL (GMM)
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_gmm(rfm_scaled: pd.DataFrame, k_range: range = range(2, 11),
                     save_path: str = None) -> None:
    """
    Plot BIC and AIC scores to choose the optimal number of GMM components.
    Lower BIC = better model.
    """
    X    = rfm_scaled[FEATURES].values
    bics, aics = [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        print(f"  k={k}  BIC={gmm.bic(X):,.1f}  AIC={gmm.aic(X):,.1f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), bics, marker="o", label="BIC", color="#4C72B0", linewidth=2)
    ax.plot(list(k_range), aics, marker="s", label="AIC", color="#C44E52", linewidth=2)
    ax.set_title("GMM: BIC / AIC vs Number of Components",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Components (K)")
    ax.set_ylabel("Score (lower = better)")
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def fit_gmm(rfm_scaled: pd.DataFrame, n_components: int) -> np.ndarray:
    X      = rfm_scaled[FEATURES].values
    gmm    = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    labels = gmm.fit_predict(X)
    probs  = gmm.predict_proba(X).max(axis=1).mean()
    print(f"  GMM (k={n_components}) silhouette: {silhouette_score(X, labels):.4f}")
    print(f"  Avg max membership probability : {probs:.4f}")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 5. ALGORITHM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_algorithms(rfm_scaled: pd.DataFrame,
                       labels_dict: dict,
                       save_path: str = None) -> pd.DataFrame:
    """
    Compare multiple label sets using three metrics.

    labels_dict : { "Algorithm Name": labels_array, ... }
                  DBSCAN labels may contain -1 (noise); these are excluded.

    Returns a DataFrame sorted by Silhouette (desc).
    """
    X       = rfm_scaled[FEATURES].values
    records = []

    for name, labels in labels_dict.items():
        mask = labels != -1          # exclude DBSCAN noise
        X_valid, l_valid = X[mask], labels[mask]
        n_clusters = len(set(l_valid))

        if n_clusters < 2:
            print(f"  Skipping {name} — fewer than 2 valid clusters.")
            continue

        sil = silhouette_score(X_valid, l_valid)
        db  = davies_bouldin_score(X_valid, l_valid)
        ch  = calinski_harabasz_score(X_valid, l_valid)
        noise_pct = (labels == -1).mean() * 100

        records.append({
            "Algorithm":        name,
            "N Clusters":       n_clusters,
            "Silhouette ↑":     round(sil, 4),
            "Davies-Bouldin ↓": round(db,  4),
            "Calinski-Harabasz ↑": round(ch, 1),
            "Noise %":          round(noise_pct, 1),
        })
        print(f"  {name:30s}  sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}  noise={noise_pct:.1f}%")

    results = pd.DataFrame(records).sort_values("Silhouette ↑", ascending=False)

    # ── Plot comparison bar chart ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]
    colors  = ["#4C72B0", "#C44E52", "#55A868"]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.barh(results["Algorithm"], results[metric], color=color, edgecolor="white")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score")
        # Annotate bars
        for i, v in enumerate(results[metric]):
            ax.text(v * 0.98 if metric == "Davies-Bouldin ↓" else v * 1.01,
                    i, f"{v}", va="center", fontsize=9)

    plt.suptitle("Algorithm Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()

    print("\n" + results.to_string(index=False))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLUSTER PROFILE (shared utility)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_profile(rfm: pd.DataFrame, label_col: str = "Cluster") -> pd.DataFrame:
    features = ["Recency", "Frequency", "Monetary"]

    # Exclude DBSCAN noise points (-1) from profile
    rfm_clean = rfm[rfm[label_col] != -1]

    profile = (
        rfm_clean.groupby(label_col)[features]
        .mean().round(2).reset_index()
    )
    profile["CustomerCount"] = rfm_clean.groupby(label_col).size().values
    profile["Segment"] = _assign_segment_labels_ranked(profile)
    profile = profile.sort_values("Monetary", ascending=False).reset_index(drop=True)

    # Noise stats (DBSCAN only)
    n_noise = (rfm[label_col] == -1).sum()
    if n_noise > 0:
        print(f"  [DBSCAN] {n_noise} noise/outlier points excluded from profile")

    print(f"\nCluster Profile ({label_col}):")
    print(profile.to_string(index=False))
    return profile


def _assign_segment_labels_ranked(profile: pd.DataFrame) -> list:
    """
    Rank-based labelling: each cluster gets a unique label determined by
    its rank on a composite RFM score.

    Score = Frequency_rank + Monetary_rank - Recency_rank
            (higher F and M = good, lower R = good)

    Works for any number of clusters (2–8+) without collision.
    """
    n = len(profile)

    # Rank each dimension (rank 1 = best)
    r_rank = profile["Recency"].rank(ascending=True)    # low recency = best
    f_rank = profile["Frequency"].rank(ascending=False) # high freq   = best
    m_rank = profile["Monetary"].rank(ascending=False)  # high money  = best

    composite = f_rank + m_rank + r_rank  # lower composite = better overall
    order = composite.rank(method="first", ascending=True).astype(int) - 1

    # Label library — covers up to 8 clusters gracefully
    # Rank 0 = best composite RFM, rank 7 = worst
    label_library = [
        "Champions",           # rank 0 — frequent, high spend, recent
        "Loyal Customers",     # rank 1 — solid engagement
        "Occasional Buyers",   # rank 2 — moderate, some promise
        "At-Risk Customers",   # rank 3 — slipping away
        "Can't Lose Them",     # rank 4 — used to be valuable
        "Hibernating",         # rank 5 — low engagement, long gone
        "Lost Customers",      # rank 6 — very inactive
        "Low Value / Lost",    # rank 7 — worst on all dimensions
    ]

    labels = [""] * n
    for idx, rank in order.items():
        labels[idx] = label_library[min(rank, len(label_library) - 1)]
    return labels