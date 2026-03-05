import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


# ── 1. Cleaning ──────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    print(f"  Dropped {before - len(df):,} rows with missing CustomerID")

    before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"  Dropped {before - len(df):,} cancellation rows")

    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"  Dropped {before - len(df):,} rows with Quantity<=0 or UnitPrice<=0")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"]  = df["CustomerID"].astype(int)
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

    print(f"  Clean dataset: {len(df):,} rows, {df['CustomerID'].nunique():,} unique customers")
    return df


# ── 2. RFM Feature Engineering ───────────────────────────────────────────────

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo",  "nunique"),
            Monetary=("TotalPrice",  "sum"),
        )
        .reset_index()
    )
    print(f"\nRFM table built: {len(rfm):,} customers")
    print(rfm.describe().round(2))
    return rfm


# ── 3. Log-Transform  (NEW) ───────────────────────────────────────────────────

def log_transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to Frequency and Monetary to compress extreme outliers.
    Recency is left as-is (already bounded, less skewed).
    Returns a new dataframe — original rfm is unchanged.
    """
    rfm_log = rfm.copy()
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
    rfm_log["Monetary"]  = np.log1p(rfm_log["Monetary"])
    print("  Log1p applied to Frequency and Monetary.")
    return rfm_log


# ── 4. Scaling ────────────────────────────────────────────────────────────────

def scale_rfm(rfm: pd.DataFrame):
    """
    Scale RFM features with RobustScaler.
    Returns (scaled_df, fitted_scaler).
    """
    scaler   = RobustScaler()
    features = ["Recency", "Frequency", "Monetary"]
    rfm_scaled = rfm.copy()
    rfm_scaled[features] = scaler.fit_transform(rfm[features])
    return rfm_scaled, scaler