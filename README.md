# Avocado Price Analysis Using Python

**Course:** PSC (CE0525), Indus University  
**Enrollment No.:** IU2441230585 | **Class:** 5 CSE-F  
**GitHub:** https://github.com/ShresthBharucha/Avocado-Price-Analysis

## 1. Project Definition
This project analyzes weekly avocado prices and sales data using Python. The dataset is cleaned (missing values,
duplicates, data types, outliers), explored with exploratory data analysis (EDA), visualized with graphs, and
summarized in a short insights report.

## 2. Dataset
- **Name:** Avocado Prices
- **Source:** [Kaggle](https://www.kaggle.com/datasets/neuromusic/avocado-prices)
- **Attributes:** Date, Average Price, Total Volume, PLU 4046/4225/4770, Total Bags, Small/Large/XLarge Bags, Type, Year, Region

The dataset is used to analyze price changes over time, organic vs conventional prices, regional price differences,
sales volume changes, and the relationship between price and volume.

## 3. Visualizations
| # | Graph | Purpose |
|---|---|---|
| 1 | Average Price Over Time (Line) | Trends and fluctuations in price |
| 2 | Organic vs Conventional Price (Bar) | Compare the two types |
| 3 | Average Price by Year (Bar) | Higher/lower price years |
| 4 | Average Price by Region (Bar) | High/low-price regions |
| 5 | Total Volume by Year (Line/Bar) | Volume changes over years |
| 6 | Price vs Total Volume (Scatter) | Relationship between price and volume |
| 7-11 | Box plots, distribution, seasonality, correlation heatmap | Outliers and extra analysis |

All graphs are saved in `images/` after running the project.

## 4. Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook / Google Colab, GitHub

## 5. Project Structure
```
Avocado-Price-Analysis/
├── data/
│   └── avocado.csv              <- download from Kaggle and put here
├── notebooks/
│   └── Avocado_Price_Analysis.ipynb
├── src/
│   └── avocado_analysis.py      <- full pipeline as a script
├── images/                      <- graphs (created when you run the project)
├── reports/
│   └── insights_report.md       <- insights report (created when you run the project)
├── requirements.txt
└── README.md
```

## 6. How to Run
1. Download `avocado.csv` from the Kaggle link above and put it in the `data/` folder.
2. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run **either** option:
   - **Script:** `python src/avocado_analysis.py` (saves cleaned data, all graphs and the insights report)
   - **Notebook:** `jupyter notebook notebooks/Avocado_Price_Analysis.ipynb` (or open it in Google Colab and upload the CSV)

## 7. Key Steps
1. **Data cleaning:** drop the index column, rename columns, fix data types, handle missing values, remove duplicates, detect outliers (IQR method).
2. **EDA:** summaries by type and year; city-level regions are separated from aggregated regions (TotalUS, West, Northeast, etc.) so regional comparisons are fair.
3. **Visualization:** six main graphs plus extra plots.
4. **Insights:** findings are generated from the data into `reports/insights_report.md`.

## 8. Conclusion
This project demonstrates real-world data cleaning, exploratory data analysis, visualization and interpretation using
Python, giving insights into avocado prices and sales patterns across years, regions and avocado types.
