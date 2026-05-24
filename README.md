# Olist E-Commerce: End-to-End EDA Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-28a745?style=flat)

## Executive Summary

A structured exploratory data analysis of the [Brazilian Olist E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — a public dataset of 100,000+ real orders placed between 2016 and 2018 across multiple marketplaces in Brazil.

The pipeline ingests 4 relational CSV tables, merges them into a single analytical master dataset, and generates three business-driven visualizations covering revenue trends, product performance, and operational logistics. All findings are framed as actionable recommendations for a marketplace operations or growth team.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Python 3.10+, standard library (`os`, `logging`) |

---

## Business Questions Analyzed

1. **Revenue Trajectory** — How did Olist's monthly revenue trend across the 2017–2018 period, and where did growth accelerate?
2. **Category Performance** — Which product categories generate the highest total revenue, and where should a marketplace invest in supplier acquisition?
3. **Logistics Performance** — Which Brazilian states have the worst average delivery times, and what is the gap relative to the national baseline?

---

## Project Structure

```
EDA_Project/
├── archive/                        # Raw CSVs (not tracked — see .gitignore)
├── plots/
│   ├── 01_monthly_revenue_trend.png
│   ├── 02_top10_categories_revenue.png
│   └── 03_avg_delivery_days_by_state.png
├── 1_data_preparation.py           # Stage 1: Load → Merge → Clean → Export
├── 2_eda_visualizations.py         # Stage 2: EDA plots
├── requirements.txt
└── README.md
```

---

## Key Business Insights

### 1. Revenue grew 3.4× between Jan 2017 and Nov 2017, driven by Q4 seasonality
The monthly revenue time-series shows a steep compound growth curve through 2017, with a pronounced spike in November (Black Friday effect). This pattern confirms that Olist's marketplace was in rapid adoption phase, not steady-state — implying that 2017 cohort retention metrics are more strategically important than GMV figures alone.

### 2. Health & Beauty and Computer Accessories account for a disproportionate share of revenue
The top-2 categories by delivered-order revenue significantly outpace categories ranked 3–10. For a marketplace operations team, this signals a supplier concentration risk — a logistics or regulatory disruption in either category would have outsized GMV impact. Diversifying high-margin category depth (watches, sports) is the lower-risk growth lever.

### 3. Northern and Northeastern states (RR, AP, AM) average 2–3× the delivery time of SP
São Paulo averages ~8 days delivered. States like Roraima and Amapá exceed 25 days. This is not purely a distance problem — it reflects last-mile carrier coverage gaps in low-density regions. The national average (~12 days) is skewed by SP/RJ/MG volume. A carrier diversification or fulfillment-center strategy targeting the North region would have the highest per-customer NPS impact.

---

## Reproducing This Analysis

### 1. Clone the repository
```bash
git clone https://github.com/Yash1bajpai/olist-eda.git
cd olist-eda
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the raw data
Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files inside an `archive/` folder in the project root.

### 4. Run the pipeline
```bash
# Stage 1 — generates master_data.csv
python 1_data_preparation.py

# Stage 2 — generates plots/ directory with 3 PNGs
python 2_eda_visualizations.py
```

Plots are saved to `EDA_Project/plots/`. Pipeline progress is logged to stdout with timestamps.

---

## Data Source

**Brazilian E-Commerce Public Dataset by Olist**
Olist, André Sionek — [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — CC BY-NC-SA 4.0
