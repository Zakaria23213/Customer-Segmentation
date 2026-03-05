import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Overview ───────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> None:
    """Print shape, dtypes, null counts, and basic describe."""
    print("=" * 55)
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nNull counts:")
    nulls = df.isnull().sum()
    pct   = (df.isnull().mean() * 100).round(2)
    print(pd.concat([nulls, pct], axis=1, keys=["count", "%"]))
    print("\nDescribe (numeric):")
    print(df.describe().round(2))
    print("=" * 55)


# ── 2. Revenue Over Time ─────────────────────────────────────────────────────

def plot_monthly_revenue(df: pd.DataFrame, save_path: str = None) -> None:
    """Bar chart of monthly revenue using TotalPrice column."""
    monthly = (
        df.assign(Month=df["InvoiceDate"].dt.to_period("M"))
        .groupby("Month")["TotalPrice"]
        .sum()
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly["Month"], monthly["TotalPrice"], color="#4C72B0", edgecolor="white")
    ax.set_title("Monthly Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 3. Top Countries ─────────────────────────────────────────────────────────

def plot_top_countries(df: pd.DataFrame, top_n: int = 10, save_path: str = None) -> None:
    """Horizontal bar chart of top countries by revenue (excluding UK to show others)."""
    by_country = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    by_country[::-1].plot(kind="barh", ax=ax, color="#DD8452")
    ax.set_title(f"Top {top_n} Countries by Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 4. RFM Distributions ─────────────────────────────────────────────────────

def plot_rfm_distributions(rfm: pd.DataFrame, save_path: str = None) -> None:
    """Histograms for Recency, Frequency, and Monetary."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    features = ["Recency", "Frequency", "Monetary"]
    colors   = ["#4C72B0", "#55A868", "#C44E52"]

    for ax, feat, color in zip(axes, features, colors):
        ax.hist(rfm[feat], bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.suptitle("RFM Feature Distributions (before scaling)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()