"""
2_eda_visualizations.py
=======================
Stage 2: Core EDA visualizations on the Olist master dataset.

Input  : master_data.csv, product_category_name_translation.csv
Output : plots/01_monthly_revenue_trend.png
         plots/02_top10_categories_revenue.png
         plots/03_avg_delivery_days_by_state.png
"""

import os
import logging
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE_DIR, "master_data.csv")
TRANS_PATH  = os.path.join(BASE_DIR, "archive", "product_category_name_translation.csv")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE_PRIMARY  = "#2C6E91"
PALETTE_ACCENT   = "#E07B39"
PALETTE_BARS     = sns.color_palette("Blues_r", 10)
PALETTE_STATES   = sns.color_palette("coolwarm_r", 27)
FONT_TITLE       = {"fontsize": 15, "fontweight": "bold", "pad": 14}
FONT_AXIS        = {"fontsize": 11}
SPINE_COLOR      = "#CCCCCC"

def apply_base_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#F8F9FA")
    ax.figure.patch.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.grid(axis="y", color=SPINE_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.tick_params(colors="#444444", labelsize=10)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2)


def save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ]
    master = pd.read_csv(MASTER_PATH, parse_dates=date_cols, low_memory=False)
    trans  = pd.read_csv(TRANS_PATH, encoding="utf-8-sig")
    log.info("Loaded master: %d rows | translation: %d rows", len(master), len(trans))
    return master, trans


# ---------------------------------------------------------------------------
# Plot 1 — Monthly Revenue Trend
# ---------------------------------------------------------------------------

def plot_monthly_revenue(master: pd.DataFrame) -> None:
    log.info("Rendering: Monthly Revenue Trend")

    # Only revenue-generating rows; canceled/unavailable orders carry price=0 or spurious values
    df = master[master["order_status"].isin(["delivered", "shipped", "invoiced"])].copy()

    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    monthly = (
        df.groupby("year_month")["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "revenue"})
        .sort_values("year_month")
    )
    monthly["year_month_str"] = monthly["year_month"].astype(str)

    # Sep-2016 is a soft-launch month with < 5 orders — a known dataset artifact, not a real trend point
    monthly = monthly[monthly["year_month_str"] >= "2017-01"]

    # Aug-2018 is the last (incomplete) month in the dataset; flagged so it's not misread as a dip
    last_month = monthly["year_month_str"].max()

    fig, ax = plt.subplots(figsize=(14, 5))
    apply_base_style(ax)

    ax.fill_between(
        monthly["year_month_str"], monthly["revenue"],
        alpha=0.12, color=PALETTE_PRIMARY,
    )
    ax.plot(
        monthly["year_month_str"], monthly["revenue"],
        color=PALETTE_PRIMARY, linewidth=2.2, marker="o", markersize=5, zorder=3,
    )

    # Annotate the incomplete trailing month so a reader doesn't interpret it as a real dip
    last_row = monthly[monthly["year_month_str"] == last_month].iloc[0]
    ax.annotate(
        "Incomplete month",
        xy=(last_month, last_row["revenue"]),
        xytext=(-60, 24),
        textcoords="offset points",
        fontsize=9, color="#999999",
        arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1),
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1_000:.0f}K"))
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["year_month_str"], rotation=45, ha="right")
    ax.set_title("Monthly Revenue Trend (Jan 2017 – Aug 2018)", **FONT_TITLE)
    ax.set_xlabel("Month", **FONT_AXIS)
    ax.set_ylabel("Total Revenue (BRL)", **FONT_AXIS)

    save(fig, "01_monthly_revenue_trend.png")


# ---------------------------------------------------------------------------
# Plot 2 — Top 10 Product Categories by Revenue
# ---------------------------------------------------------------------------

def plot_top_categories(master: pd.DataFrame, trans: pd.DataFrame) -> None:
    log.info("Rendering: Top 10 Categories by Revenue")

    df = master[master["order_status"] == "delivered"].copy()
    df = df.merge(trans, on="product_category_name", how="left")

    # ~600 rows have no category mapping (product_id exists but category is null in source)
    # Labeled explicitly so they don't silently vanish from the chart
    df["category_en"] = df["product_category_name_english"].fillna("Uncategorized")

    top10 = (
        df.groupby("category_en")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"price": "revenue"})
        .sort_values("revenue")          # ascending for horizontal readability
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    apply_base_style(ax)
    ax.grid(axis="x", color=SPINE_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.grid(axis="y", visible=False)

    bars = ax.barh(
        top10["category_en"], top10["revenue"],
        color=PALETTE_BARS, edgecolor="none", height=0.65,
    )

    for bar, val in zip(bars, top10["revenue"]):
        ax.text(
            bar.get_width() + top10["revenue"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"R$ {val/1_000:.0f}K",
            va="center", fontsize=9, color="#444444",
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1_000:.0f}K"))
    ax.set_title("Top 10 Product Categories by Revenue (Delivered Orders)", **FONT_TITLE)
    ax.set_xlabel("Total Revenue (BRL)", **FONT_AXIS)
    ax.set_ylabel("")

    save(fig, "02_top10_categories_revenue.png")


# ---------------------------------------------------------------------------
# Plot 3 — Avg Delivery Days by Customer State
# ---------------------------------------------------------------------------

def plot_delivery_by_state(master: pd.DataFrame) -> None:
    log.info("Rendering: Avg Delivery Days by Customer State")

    # Restrict to delivered orders with both timestamps present; carrier/approval nulls are irrelevant here
    df = master[
        (master["order_status"] == "delivered") &
        (master["order_delivered_customer_date"].notna()) &
        (master["order_purchase_timestamp"].notna())
    ].copy()

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Negative or zero delivery days are data-entry errors (timestamp reversals); drop them
    df = df[df["delivery_days"] > 0]

    state_avg = (
        df.groupby("customer_state")["delivery_days"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"delivery_days": "avg_delivery_days"})
    )

    # SP/RJ/MG are the economic hubs — their delivery times anchor the national baseline;
    # flagged with a darker bar so they read as reference points, not outliers
    hub_states = {"SP", "RJ", "MG"}
    state_avg["bar_color"] = state_avg["customer_state"].apply(
        lambda s: PALETTE_ACCENT if s in hub_states else PALETTE_PRIMARY
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    apply_base_style(ax)

    sns.barplot(
        data=state_avg,
        x="customer_state", y="avg_delivery_days",
        hue="customer_state",
        palette=dict(zip(state_avg["customer_state"], state_avg["bar_color"])),
        legend=False,
        edgecolor="none",
        ax=ax,
    )

    national_avg = state_avg["avg_delivery_days"].mean()
    ax.axhline(national_avg, color=PALETTE_ACCENT, linewidth=1.4, linestyle="--", zorder=5)
    ax.text(
        len(state_avg) - 0.4, national_avg + 0.4,
        f"National avg: {national_avg:.1f}d",
        ha="right", fontsize=9, color=PALETTE_ACCENT,
    )

    ax.set_title("Average Delivery Time by Customer State", **FONT_TITLE)
    ax.set_xlabel("Customer State (BR)", **FONT_AXIS)
    ax.set_ylabel("Avg Delivery Days", **FONT_AXIS)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    save(fig, "03_avg_delivery_days_by_state.png")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    log.info("=== Stage 2: EDA Visualizations ===")
    master, trans = load_data()
    plot_monthly_revenue(master)
    plot_top_categories(master, trans)
    plot_delivery_by_state(master)
    log.info("=== All plots saved to %s ===", PLOTS_DIR)


if __name__ == "__main__":
    run_pipeline()
