"""
visualization.py
================
All plotting functions. Each accepts an optional save_path.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", palette="muted")
PALETTE  = sns.color_palette("tab10")
FEATURES = ["Recency", "Frequency", "Monetary"]


# ── PCA scatter (works for any algorithm) ─────────────────────────────────────

def plot_clusters_pca(rfm_scaled: pd.DataFrame, cluster_col: str = "Cluster",
                      title: str = "Customer Clusters (PCA)",
                      save_path: str = None) -> None:
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(rfm_scaled[FEATURES])
    var    = pca.explained_variance_ratio_ * 100

    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["Cluster"] = rfm_scaled[cluster_col].values

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster, grp in plot_df.groupby("Cluster"):
        color = "#aaaaaa" if cluster == -1 else PALETTE[int(cluster) % len(PALETTE)]
        label = "Noise" if cluster == -1 else f"Cluster {cluster}"
        ax.scatter(grp["PC1"], grp["PC2"], label=label,
                   s=15, alpha=0.5, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ── Side-by-side PCA for all algorithms ──────────────────────────────────────

def plot_all_pca(rfm_scaled: pd.DataFrame, labels_dict: dict,
                 save_path: str = None) -> None:
    """
    Grid of PCA scatter plots — one panel per algorithm.
    labels_dict: {"Algorithm Name": labels_array, ...}
    """
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(rfm_scaled[FEATURES])
    var    = pca.explained_variance_ratio_ * 100

    n      = len(labels_dict)
    ncols  = 2
    nrows  = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        unique = sorted(set(labels))
        for cluster in unique:
            mask   = labels == cluster
            color  = "#cccccc" if cluster == -1 else PALETTE[int(cluster) % len(PALETTE)]
            lbl    = "Noise" if cluster == -1 else f"C{cluster}"
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=lbl, s=10, alpha=0.45, color=color)
        noise_pct = (labels == -1).mean() * 100
        n_clust   = len(unique) - (1 if -1 in unique else 0)
        ax.set_title(f"{name}\n({n_clust} clusters, {noise_pct:.0f}% noise)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        ax.legend(fontsize=7, markerscale=2, loc="upper right")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("PCA Projection — All Algorithms", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Snake plot ─────────────────────────────────────────────────────────────────

def plot_snake(rfm: pd.DataFrame, cluster_col: str = "Cluster",
               save_path: str = None) -> None:
    rfm_norm = rfm.copy()
    for f in FEATURES:
        mn, mx = rfm[f].min(), rfm[f].max()
        rfm_norm[f] = (rfm[f] - mn) / (mx - mn) if mx > mn else 0

    melted = rfm_norm[rfm_norm[cluster_col] != -1].melt(
        id_vars=cluster_col, value_vars=FEATURES,
        var_name="Metric", value_name="NormValue")
    means = melted.groupby([cluster_col, "Metric"])["NormValue"].mean().reset_index()

    # Use segment names if available
    seg_map = {}
    if "Segment" in rfm.columns:
        seg_map = rfm[rfm[cluster_col] != -1].groupby(cluster_col)["Segment"].first().to_dict()

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (cluster, grp) in enumerate(means.groupby(cluster_col)):
        lbl = seg_map.get(cluster, f"Cluster {cluster}")
        ax.plot(grp["Metric"], grp["NormValue"], marker="o",
                label=lbl, color=PALETTE[i % len(PALETTE)], linewidth=2.5)

    ax.set_title("Snake Plot — Normalised RFM by Segment", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalised Value (0–1)"); ax.set_xlabel("RFM Metric")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ── Profile heatmap ────────────────────────────────────────────────────────────

def plot_profile_heatmap(profile: pd.DataFrame, save_path: str = None) -> None:
    label_col = "Segment" if "Segment" in profile.columns else "Cluster"
    heat      = profile.set_index(label_col)[FEATURES].copy()
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min())

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(heat) + 2))
    sns.heatmap(heat_norm, annot=heat.round(1), fmt="g",
                cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Normalised value"})
    ax.set_title("Cluster Profile Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ── Cluster sizes ──────────────────────────────────────────────────────────────

def plot_cluster_sizes(rfm: pd.DataFrame, cluster_col: str = "Cluster",
                       save_path: str = None) -> None:
    counts = rfm[cluster_col].value_counts().sort_index()
    colors = ["#aaaaaa" if c == -1 else PALETTE[int(c) % len(PALETTE)]
              for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=colors, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=11)
    ax.set_title("Customer Count per Cluster", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Customers")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ── RFM distributions ──────────────────────────────────────────────────────────

def plot_rfm_distributions(rfm: pd.DataFrame, save_path: str = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for ax, feat, color in zip(axes, FEATURES, colors):
        ax.hist(rfm[feat], bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
    fig.suptitle("RFM Feature Distributions (after log transform)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ── Monthly revenue ────────────────────────────────────────────────────────────

def plot_monthly_revenue(df: pd.DataFrame, save_path: str = None) -> None:
    monthly = (df.assign(Month=df["InvoiceDate"].dt.to_period("M"))
               .groupby("Month")["TotalPrice"].sum().reset_index())
    monthly["Month"] = monthly["Month"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly["Month"], monthly["TotalPrice"], color="#4C72B0", edgecolor="white")
    ax.set_title("Monthly Revenue", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# ── Top countries ──────────────────────────────────────────────────────────────

def plot_top_countries(df: pd.DataFrame, top_n: int = 10,
                       save_path: str = None) -> None:
    by_country = (df.groupby("Country")["TotalPrice"].sum()
                  .sort_values(ascending=False).head(top_n))
    fig, ax = plt.subplots(figsize=(10, 6))
    by_country[::-1].plot(kind="barh", ax=ax, color="#DD8452")
    ax.set_title(f"Top {top_n} Countries by Revenue", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()