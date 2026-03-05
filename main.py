"""
main.py — E-Commerce Customer Segmentation Pipeline
====================================================
Runs four clustering algorithms, compares them, and saves all outputs.

Usage:
    python main.py

Tune the CONFIG block below after reviewing the elbow / BIC / k-distance plots.
"""

import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
SRC     = ROOT / "src"
DATA    = ROOT / "data" / "data.csv"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
sys.path.insert(0, str(SRC))

from data_loader   import load_data
from preprocessing import clean_data, build_rfm, log_transform_rfm, scale_rfm
from eda           import summarize
from clustering    import (find_optimal_k, fit_kmeans,
                           plot_dendrogram, fit_agglomerative,
                           plot_kdistance,  fit_dbscan,
                           find_optimal_gmm, fit_gmm,
                           compare_algorithms, cluster_profile)
from visualization import (plot_monthly_revenue, plot_top_countries,
                            plot_rfm_distributions, plot_clusters_pca,
                            plot_all_pca, plot_snake,
                            plot_profile_heatmap, plot_cluster_sizes)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — adjust after reviewing the diagnostic plots
# ══════════════════════════════════════════════════════════════════════════════
K_MEANS_N       = 3       # from elbow / silhouette
AGGLO_N         = 3       # from dendrogram
GMM_N           = 3       # from BIC / AIC plot
DBSCAN_EPS      = 0.3     # tuned from k-distance elbow (elbow visible ~0.3)
DBSCAN_MIN_SAMP = 10
K_RANGE         = range(2, 9)
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("\n" + "=" * 60)
    print("  E-Commerce Customer Segmentation — Multi-Algorithm")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = load_data(DATA, encoding="latin1")
    print(f"  Raw: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    print("\n[2/7] EDA...")
    summarize(df)

    # ── 3. Clean + RFM ────────────────────────────────────────────────────────
    print("\n[3/7] Cleaning & building RFM...")
    df_clean = clean_data(df)
    plot_monthly_revenue(df_clean, save_path=str(OUTPUTS / "monthly_revenue.png"))
    plot_top_countries(df_clean,   save_path=str(OUTPUTS / "top_countries.png"))

    rfm = build_rfm(df_clean)

    # ── 4. Log-transform + Scale ──────────────────────────────────────────────
    print("\n[4/7] Log-transform & scaling...")
    rfm_log        = log_transform_rfm(rfm)
    rfm_scaled, _  = scale_rfm(rfm_log)
    plot_rfm_distributions(rfm_log, save_path=str(OUTPUTS / "rfm_distributions_log.png"))

    # ── 5. Diagnostic plots (run once, then tune CONFIG above) ────────────────
    print("\n[5/7] Diagnostic plots for K selection...")
    find_optimal_k(rfm_scaled, k_range=K_RANGE,
                   save_path=str(OUTPUTS / "kmeans_elbow_silhouette.png"))
    plot_dendrogram(rfm_scaled,
                    save_path=str(OUTPUTS / "dendrogram.png"))
    plot_kdistance(rfm_scaled, k=DBSCAN_MIN_SAMP,
                   save_path=str(OUTPUTS / "dbscan_kdistance.png"))
    find_optimal_gmm(rfm_scaled, k_range=K_RANGE,
                     save_path=str(OUTPUTS / "gmm_bic_aic.png"))

    # ── 6. Fit all algorithms ─────────────────────────────────────────────────
    print("\n[6/7] Fitting all algorithms...")

    km_labels    = fit_kmeans(rfm_scaled,      n_clusters=K_MEANS_N)
    agglo_labels = fit_agglomerative(rfm_scaled, n_clusters=AGGLO_N)
    dbscan_labels= fit_dbscan(rfm_scaled,       eps=DBSCAN_EPS,
                                                min_samples=DBSCAN_MIN_SAMP)
    gmm_labels   = fit_gmm(rfm_scaled,          n_components=GMM_N)

    labels_dict = {
        "K-Means":       km_labels,
        "Agglomerative": agglo_labels,
        "DBSCAN":        dbscan_labels,
        "GMM":           gmm_labels,
    }

    # ── 7. Compare + Visualise ────────────────────────────────────────────────
    print("\n[7/7] Comparing algorithms & saving visuals...")

    comparison = compare_algorithms(rfm_scaled, labels_dict,
                                    save_path=str(OUTPUTS / "algorithm_comparison.png"))
    comparison.to_csv(OUTPUTS / "algorithm_comparison.csv", index=False)

    # Side-by-side PCA for all algorithms
    rfm_scaled_copy = rfm_scaled.copy()
    for name, labels in labels_dict.items():
        rfm_scaled_copy[name] = labels
    plot_all_pca(rfm_scaled, labels_dict,
                 save_path=str(OUTPUTS / "all_pca_comparison.png"))

    # Best algorithm deep-dive (K-Means by default — swap if comparison says otherwise)
    best_labels = km_labels
    rfm["Cluster"]        = best_labels
    rfm_scaled["Cluster"] = best_labels

    profile = cluster_profile(rfm)
    profile.to_csv(OUTPUTS / "cluster_profile.csv", index=False)

    # Merge segment names back into rfm so snake plot uses labels
    rfm = rfm.merge(profile[["Cluster", "Segment"]], on="Cluster", how="left")
    rfm.to_csv(OUTPUTS / "rfm_with_clusters.csv", index=False)

    plot_clusters_pca(rfm_scaled,
                      title="K-Means Clusters (PCA)",
                      save_path=str(OUTPUTS / "pca_best.png"))
    plot_snake(rfm,             save_path=str(OUTPUTS / "snake_plot.png"))
    plot_profile_heatmap(profile, save_path=str(OUTPUTS / "profile_heatmap.png"))
    plot_cluster_sizes(rfm,     save_path=str(OUTPUTS / "cluster_sizes.png"))

    print("\n✅  Done. Outputs saved to:", OUTPUTS)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()