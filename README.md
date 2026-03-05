# E-Commerce Customer Segmentation

Customer segmentation project using RFM (Recency, Frequency, Monetary) analysis and multiple clustering algorithms on the UCI Online Retail Dataset.

---

## Project Structure

```
Cust_Seg/
├── main.py                  # Entry point — runs the full pipeline
├── data/
│   └── data.csv             # Raw transactional dataset (place here)
├── src/
│   ├── data_loader.py       # CSV loading with encoding/delimiter support
│   ├── preprocessing.py     # Cleaning, RFM engineering, log-transform, scaling
│   ├── eda.py               # Exploratory analysis and business charts
│   ├── clustering.py        # K-Means, Agglomerative, DBSCAN, GMM + comparison
│   └── visualization.py     # PCA scatter, snake plot, heatmap, cluster sizes
└── outputs/                 # All generated plots and CSVs saved here
```

---

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

Place `data.csv` in the `data/` folder, then run:

```bash
python main.py
```

---

## Pipeline Overview

1. **Load** — reads raw CSV with latin1 encoding
2. **Clean** — drops anonymous rows, cancellations, invalid quantities/prices
3. **RFM** — aggregates transactions to one row per customer (Recency, Frequency, Monetary)
4. **Transform** — log1p on Frequency and Monetary, then RobustScaler
5. **Diagnose** — elbow/silhouette plot, dendrogram, k-distance plot, BIC/AIC chart
6. **Cluster** — fits all four algorithms, compares on three metrics
7. **Profile** — labels segments, saves CSVs and visualisations

---

## Algorithms Compared

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| **K-Means** ✅ | **0.4218** | **0.8069** | **4,591** |
| Agglomerative | 0.3805 | 0.8580 | 3,876 |
| DBSCAN | 0.3210 | 1.1822 | 2,283 |
| GMM | 0.1935 | 1.5679 | 2,117 |

K-Means (K=3) was selected as the best-performing algorithm.

---

## Segments Identified

| Segment | Customers | Recency | Frequency | Avg. Spend |
|---|---|---|---|---|
| Champions | 1,367 (32%) | 30 days | 9.6 orders | £5,388 |
| Loyal Customers | 1,984 (46%) | 54 days | 2.0 orders | £577 |
| Occasional Buyers | 987 (23%) | 255 days | 1.4 orders | £406 |

DBSCAN additionally flagged **114 customers (2.6%) as outliers** — ultra-high-value accounts that fall outside normal segment patterns.

---

## Configuration

Tune these values in `main.py` after reviewing the diagnostic plots:

```python
K_MEANS_N       = 3      # elbow/silhouette plot
AGGLO_N         = 3      # dendrogram
GMM_N           = 3      # BIC/AIC plot
DBSCAN_EPS      = 0.3    # k-distance elbow
DBSCAN_MIN_SAMP = 10
```

---

## Outputs

| File | Description |
|---|---|
| `rfm_with_clusters.csv` | One row per customer with RFM values, cluster ID, and segment name |
| `cluster_profile.csv` | Mean RFM stats and customer count per segment |
| `algorithm_comparison.csv` | Metric scores for all four algorithms |
| `kmeans_elbow_silhouette.png` | K selection diagnostic for K-Means |
| `all_pca_comparison.png` | Side-by-side PCA view of all four algorithms |
| `snake_plot.png` | Normalised RFM comparison across segments |
| `profile_heatmap.png` | Heatmap of segment profiles |

---

## Dataset

**UCI Online Retail Dataset** — 541,909 transactions from a UK-based online retailer (Dec 2010 – Dec 2011).
After cleaning: **397,884 rows**, **4,338 unique customers**.
