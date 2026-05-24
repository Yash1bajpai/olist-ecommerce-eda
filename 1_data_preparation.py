"""
1_data_preparation.py
=====================
Stage 1: Load, merge, clean, and export the Olist master dataset.

Input  : Raw CSVs from ./archive/
Output : master_data.csv (project root)
"""

import os
import logging
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_PATH = os.path.join(BASE_DIR, "master_data.csv")

FILES = {
    "orders":    "olist_orders_dataset.csv",
    "items":     "olist_order_items_dataset.csv",
    "products":  "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}

# Columns to parse as datetime — explicitly listed to avoid silent inference errors
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> pd.DataFrame:
    """Load a single CSV from the archive directory."""
    path = os.path.join(ARCHIVE_DIR, filename)
    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig strips BOM if present
    log.info("Loaded %-45s → %d rows, %d cols", filename, *df.shape)
    return df


def load_all(file_map: dict) -> dict:
    return {key: load_csv(fname) for key, fname in file_map.items()}


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_master(dfs: dict) -> pd.DataFrame:
    """
    Merge strategy:
      orders ──(order_id)──► items       [1:many  — one order, multiple line items]
             ──(customer_id)► customers  [many:1  — many orders per unique customer]
      items  ──(product_id)► products    [many:1  — many items share a product]

    Left joins throughout so orders remain the authoritative spine.
    Items with missing product metadata are retained (nulls flagged later).
    """
    master = (
        dfs["orders"]
        .merge(dfs["items"],     on="order_id",    how="left")
        .merge(dfs["customers"], on="customer_id", how="left")
        .merge(dfs["products"],  on="product_id",  how="left")
    )
    log.info("Master shape after merge: %d rows × %d cols", *master.shape)
    return master


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce date strings to datetime64[ns].
    errors='coerce' turns unparseable values into NaT rather than raising —
    important because delivery columns are legitimately null for non-delivered orders.
    """
    present = [c for c in DATE_COLUMNS if c in df.columns]
    df[present] = df[present].apply(pd.to_datetime, errors="coerce")
    log.info("Converted %d date columns to datetime64", len(present))
    return df


def handle_structural_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural nulls = nulls that exist by design, not data quality issues.

    Rule: delivery timestamp columns are null for any order that hasn't reached
    that stage yet. Flagging them preserves the signal without dropping rows.
    Imputing or dropping would silently bias delivery-time analysis.
    """
    delivery_cols = [
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ]
    present = [c for c in delivery_cols if c in df.columns]
    for col in present:
        flag_col = f"is_{col}_missing"
        df[flag_col] = df[col].isna().astype("int8")

    null_summary = df[present].isna().sum()
    log.info("Structural null flags added:\n%s", null_summary.to_string())
    return df


def drop_true_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop fully identical rows only. Partial duplicates (e.g. same order_id
    with different item rows) are valid and must not be touched.
    """
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d exact duplicate rows", dropped)
    return df


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    int_cols   = df.select_dtypes("int64").columns
    float_cols = df.select_dtypes("float64").columns

    df[int_cols]   = df[int_cols].apply(pd.to_numeric, downcast="integer")
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

    log.info(
        "Downcasted %d int and %d float columns",
        len(int_cols), len(float_cols),
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_dates(df)
    df = handle_structural_nulls(df)
    df = drop_true_duplicates(df)
    df = cast_dtypes(df)
    return df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    log.info("Exported → %s  (%.1f MB, %d rows)", path, size_mb, len(df))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    log.info("=== Stage 1: Data Preparation ===")

    dfs    = load_all(FILES)
    master = build_master(dfs)
    master = clean(master)
    export(master, OUTPUT_PATH)

    log.info("=== Pipeline complete ===")
    return master


if __name__ == "__main__":
    run_pipeline()
