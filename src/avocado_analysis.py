"""
Avocado Price Analysis Using Python
PSC (CE0525) | Indus University | IU2441230585 | 5 CSE-F

Runs the complete pipeline:
  load -> clean -> EDA -> 6 main visualizations + extra plots -> insights report

Usage (from the project root):
    python src/avocado_analysis.py
    python src/avocado_analysis.py --data path/to/avocado.csv
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({"figure.figsize": (11, 5), "axes.titlesize": 14, "axes.titleweight": "bold"})

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images"
REPORT_DIR = ROOT / "reports"
TYPE_COLORS = ["#8bc34a", "#33691e"]
AGGREGATE_REGIONS = ["TotalUS", "California", "West", "Plains", "SouthCentral",
                     "Southeast", "Northeast", "Midsouth", "GreatLakes"]


# ----------------------------------------------------------------- 1. LOAD
def load_data(path):
    df = pd.read_csv(path)
    print(f"[load] {path}  shape={df.shape}")
    return df


# ---------------------------------------------------------------- 2. CLEAN
def clean_data(df):
    log = {"rows_raw": len(df)}

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={
        "AveragePrice": "average_price", "Total Volume": "total_volume",
        "4046": "plu_4046", "4225": "plu_4225", "4770": "plu_4770",
        "Total Bags": "total_bags", "Small Bags": "small_bags",
        "Large Bags": "large_bags", "XLarge Bags": "xlarge_bags", "Date": "date",
    })

    # data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["type"] = df["type"].astype("category")
    df["region"] = df["region"].astype("category")
    df["year"] = df["year"].astype(int)

    # missing values
    log["missing_before"] = int(df.isnull().sum().sum())
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.dropna(subset=["date", "type", "region"])
    log["missing_after"] = int(df.isnull().sum().sum())

    # duplicates
    log["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    df["month"] = df["date"].dt.month
    log["rows_clean"] = len(df)

    # outliers (IQR): reported, not removed
    out = []
    for col in ["average_price", "total_volume", "total_bags"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        out.append((col, n, round(100 * n / len(df), 2), round(lo, 2), round(hi, 2)))
    log["outliers"] = out

    print(f"[clean] raw={log['rows_raw']} clean={log['rows_clean']} "
          f"duplicates={log['duplicates']} missing={log['missing_before']}")
    return df, log


def city_only(df):
    d = df[~df["region"].isin(AGGREGATE_REGIONS)].copy()
    d["region"] = d["region"].cat.remove_unused_categories()
    return d


# ---------------------------------------------------------------- 3. PLOTS
def save(name):
    IMG_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] images/{name}")


def plot_all(df, cities):
    # 1. Average price over time
    s = df.groupby("date")["average_price"].mean()
    plt.figure()
    plt.plot(s.index, s.values, color="#2e7d32", lw=1.8, label="Weekly average")
    plt.plot(s.rolling(8).mean(), color="#e65100", lw=2.2, label="8-week moving average")
    plt.title("Average Avocado Price Over Time")
    plt.xlabel("Date"); plt.ylabel("Average price (USD)"); plt.legend()
    save("01_price_over_time.png")

    # 2. Organic vs conventional
    t = df.groupby("type")["average_price"].mean().sort_values()
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=t.index.astype(str), y=t.values, palette=TYPE_COLORS)
    for p in ax.patches:
        ax.annotate(f"${p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontweight="bold")
    plt.title("Average Price: Conventional vs Organic")
    plt.xlabel("Avocado type"); plt.ylabel("Average price (USD)")
    save("02_organic_vs_conventional.png")

    # 3. Price by year
    y = df.groupby(["year", "type"])["average_price"].mean().reset_index()
    plt.figure()
    sns.barplot(data=y, x="year", y="average_price", hue="type", palette=TYPE_COLORS)
    plt.title("Average Avocado Price by Year")
    plt.xlabel("Year"); plt.ylabel("Average price (USD)"); plt.legend(title="Type")
    save("03_price_by_year.png")

    # 4. Price by region (top / bottom 10 cities)
    r = cities.groupby("region")["average_price"].mean().sort_values(ascending=False)
    top, bottom = r.head(10), r.tail(10)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    sns.barplot(x=top.values, y=top.index.astype(str), ax=axes[0], palette="Reds_r")
    axes[0].set_title("Top 10 most expensive regions")
    sns.barplot(x=bottom.values, y=bottom.index.astype(str), ax=axes[1], palette="Greens")
    axes[1].set_title("Top 10 cheapest regions")
    for a in axes:
        a.set_xlabel("Average price (USD)"); a.set_ylabel("")
    save("04_price_by_region.png")

    # 5. Total volume by year (national, no double counting)
    us = df[df["region"] == "TotalUS"]
    if us.empty:
        us = cities
    v = us.groupby("year")["total_volume"].sum() / 1e9
    fig, ax = plt.subplots()
    ax.bar(v.index.astype(str), v.values, color="#a5d6a7", label="Total volume")
    ax.plot(v.index.astype(str), v.values, color="#1b5e20", marker="o", lw=2.5, label="Trend")
    for xi, yi in zip(v.index.astype(str), v.values):
        ax.annotate(f"{yi:.2f}B", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title("Total Avocado Volume Sold in the US by Year")
    ax.set_xlabel("Year"); ax.set_ylabel("Volume (billions)"); ax.legend()
    save("05_volume_by_year.png")

    # 6. Price vs volume
    smp = cities.sample(n=min(5000, len(cities)), random_state=42)
    plt.figure()
    sns.scatterplot(data=smp, x="total_volume", y="average_price", hue="type",
                    alpha=0.45, palette=TYPE_COLORS, s=25)
    plt.xscale("log")
    plt.title("Average Price vs Total Volume (log scale on x-axis)")
    plt.xlabel("Total volume (log scale)"); plt.ylabel("Average price (USD)")
    save("06_price_vs_volume.png")

    # Extra: outlier box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["average_price", "total_volume", "total_bags"]):
        sns.boxplot(y=df[col], ax=ax, color="#4c9a2a"); ax.set_title(f"Box plot: {col}")
    save("07_outlier_boxplots.png")

    # Extra: distribution
    plt.figure()
    sns.histplot(data=df, x="average_price", hue="type", kde=True, bins=40,
                 palette=TYPE_COLORS, alpha=0.5)
    plt.title("Distribution of Average Price by Type"); plt.xlabel("Average price (USD)")
    save("08_price_distribution.png")

    # Extra: price spread by year/type
    plt.figure()
    sns.boxplot(data=df, x="year", y="average_price", hue="type", palette=TYPE_COLORS)
    plt.title("Price Spread by Year and Type")
    save("09_price_spread_boxplot.png")

    # Extra: seasonality
    m = df.groupby("month")["average_price"].mean()
    plt.figure()
    sns.lineplot(x=m.index, y=m.values, marker="o", color="#2e7d32")
    plt.xticks(range(1, 13)); plt.title("Average Price by Month (Seasonality)")
    plt.xlabel("Month"); plt.ylabel("Average price (USD)")
    save("10_monthly_seasonality.png")

    # Extra: heatmap
    cols = ["average_price", "total_volume", "plu_4046", "plu_4225", "plu_4770",
            "total_bags", "small_bags", "large_bags", "xlarge_bags"]
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="YlGn", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save("11_correlation_heatmap.png")


# ------------------------------------------------------------- 4. INSIGHTS
def write_report(df, cities, log):
    t = df.groupby("type")["average_price"].mean()
    y = df.groupby("year")["average_price"].mean()
    r = cities.groupby("region")["average_price"].mean().sort_values()
    m = df.groupby("month")["average_price"].mean()
    us = df[df["region"] == "TotalUS"]
    vol = (us if not us.empty else cities).groupby("year")["total_volume"].sum()
    full = vol.drop(index=vol.index.max()) if len(vol) > 1 else vol
    corr = np.corrcoef(np.log10(cities["total_volume"]), cities["average_price"])[0, 1]
    strength = "weak" if abs(corr) < 0.3 else "moderate" if abs(corr) < 0.6 else "strong"
    direction = "negative (higher volume tends to go with lower prices)" if corr < 0 \
        else "positive (higher volume tends to go with higher prices)"
    pct = 100 * (t["organic"] / t["conventional"] - 1)

    lines = [
        "# Insights Report: Avocado Price Analysis", "",
        "*PSC (CE0525), Indus University, IU2441230585, 5 CSE-F*", "",
        "## 1. Data Cleaning Summary", "",
        f"- Raw records: **{log['rows_raw']:,}**; after cleaning: **{log['rows_clean']:,}**",
        f"- Duplicate rows removed: **{log['duplicates']}**",
        f"- Missing values found: **{log['missing_before']}** (remaining: {log['missing_after']})",
        "- Data types fixed: `date` to datetime; `type`, `region` to category; `year` to int",
        "- Outliers (IQR method) were detected and kept because they reflect real market events:", "",
        "| Column | Outliers | % of rows | Lower bound | Upper bound |", "|---|---|---|---|---|",
    ]
    lines += [f"| {c} | {n} | {p}% | {lo} | {hi} |" for c, n, p, lo, hi in log["outliers"]]
    lines += [
        "", "## 2. Key Findings", "",
        f"1. **Dataset:** {len(df):,} weekly records, {df['region'].nunique()} regions, "
        f"{df['date'].min().date()} to {df['date'].max().date()}.",
        f"2. **Overall average price:** ${df['average_price'].mean():.2f} per avocado.",
        f"3. **Organic vs conventional:** organic (${t['organic']:.2f}) is about {pct:.0f}% more expensive "
        f"than conventional (${t['conventional']:.2f}).",
        f"4. **Years:** highest average price in {y.idxmax()} (${y.max():.2f}), lowest in {y.idxmin()} (${y.min():.2f}).",
        f"5. **Regions:** most expensive is {r.index[-1]} (${r.iloc[-1]:.2f}), cheapest is {r.index[0]} (${r.iloc[0]:.2f}).",
        f"6. **Seasonality:** prices peak in month {m.idxmax()} (${m.max():.2f}) and dip in month {m.idxmin()} (${m.min():.2f}).",
        f"7. **Price vs volume:** correlation with log(volume) is {corr:.2f}, a {strength}, {direction} relationship.",
        f"8. **Volume trend:** full-year US volume went from {full.iloc[0]/1e9:.2f}B ({full.index[0]}) "
        f"to {full.iloc[-1]/1e9:.2f}B ({full.index[-1]}). The last year in the data is incomplete, so it is excluded here.",
        "", "## 3. Visualizations", "",
        "| # | Graph | File | Purpose |", "|---|---|---|---|",
        "| 1 | Line chart | `images/01_price_over_time.png` | Price trends and fluctuations |",
        "| 2 | Bar chart | `images/02_organic_vs_conventional.png` | Organic vs conventional price |",
        "| 3 | Bar chart | `images/03_price_by_year.png` | Higher/lower price years |",
        "| 4 | Bar chart | `images/04_price_by_region.png` | High/low-price regions |",
        "| 5 | Line/bar chart | `images/05_volume_by_year.png` | Volume changes over years |",
        "| 6 | Scatter plot | `images/06_price_vs_volume.png` | Price and volume relationship |",
        "| 7-11 | Extra plots | `images/07_...` to `images/11_...` | Outliers, distribution, seasonality, correlation |",
        "", "## 4. Conclusion", "",
        "The analysis shows how avocado prices differ by type, year, region and season, and how sales volume relates "
        "to price. Cleaning and visual exploration with Python (Pandas, NumPy, Matplotlib, Seaborn) made these patterns "
        "easy to identify.", "",
    ]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "insights_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[report] reports/insights_report.md")


# -------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "avocado.csv"))
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found: {args.data}\n"
                         "Download avocado.csv from https://www.kaggle.com/datasets/neuromusic/avocado-prices "
                         "and place it in the data/ folder.")

    df = load_data(args.data)
    df, log = clean_data(df)
    cities = city_only(df)
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "avocado_cleaned.csv", index=False)
    plot_all(df, cities)
    write_report(df, cities, log)
    print("\nDone. Check images/ and reports/insights_report.md")


if __name__ == "__main__":
    main()
