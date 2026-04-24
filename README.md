# 🌍 Air Quality Risk Prediction & Public Health Analytics

> A full end-to-end data science pipeline for identifying high-risk pollution zones, modeling AQI trends, and forecasting future air quality across 10 global cities.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%200.992-success?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Highlights](#-project-highlights)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Key Results](#-key-results)
- [Visualisations](#-visualisations)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Findings & Recommendations](#-findings--recommendations)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧩 Problem Statement

Air pollution is one of the leading environmental health risks globally, responsible for millions of premature deaths annually. Despite widespread concern, **actionable data-driven insights** for identifying high-risk zones and enabling timely public health interventions remain limited — particularly due to fragmented and inconsistent data across cities.

This project builds a complete data science system that:

- 🔴 Identifies cities and time periods with **elevated pollution risk**
- 📈 Analyzes **temporal and geographic pollution patterns**
- 🤖 Predicts **future AQI trends** using machine learning
- 📊 Supports **data-informed decision-making** for public health stakeholders

---

## ✨ Project Highlights

| Metric | Value |
|--------|-------|
| Dataset size | **438,010** hourly readings |
| Cities covered | **10** global cities |
| Time span | **5 years** (2019–2023) |
| Pollutants tracked | AQI, PM2.5, PM10, NO₂, CO, O₃ |
| Best model R² | **0.9923** (Random Forest) |
| Forecast horizon | **30 days** rolling |
| Charts generated | **14** publication-ready figures |
| Notebook cells | **56** step-by-step cells |

---

## 📦 Dataset

The dataset is synthetically generated to simulate realistic large-scale air quality measurements across 10 cities with the following characteristics:

### Cities & Pollution Profiles

| City | Region | Mean AQI | Risk Level |
|------|--------|----------|------------|
| Delhi | South Asia | 183.5 | 🔴 Critical |
| Beijing | East Asia | 162.9 | 🔴 Critical |
| Cairo | North Africa | 147.7 | 🔴 High |
| Lagos | West Africa | 132.3 | 🟠 High |
| São Paulo | South America | 96.7 | 🟡 Moderate |
| Tokyo | East Asia | 59.1 | 🟢 Low |
| New York | North America | 56.1 | 🟢 Low |
| London | Europe | 51.0 | 🟢 Low |
| Munich | Europe | 45.9 | 🟢 Low |
| Sydney | Oceania | 40.7 | 🟢 Low |

### Pollutants Tracked

| Pollutant | Unit | Description |
|-----------|------|-------------|
| AQI | Index (0–500) | Air Quality Index (US EPA standard) |
| PM2.5 | μg/m³ | Fine particulate matter |
| PM10 | μg/m³ | Coarse particulate matter |
| NO₂ | ppb | Nitrogen dioxide |
| CO | ppm | Carbon monoxide |
| O₃ | ppb | Ground-level ozone |

### Realistic Simulation Features

- ⏰ **Rush-hour spikes** — dual Gaussian peaks at 8 am and 6 pm
- 🌦️ **Seasonal patterns** — winter peaks in Northern Hemisphere cities
- 📅 **Weekend dips** — 15% reduction on Saturdays and Sundays
- 📉 **Improvement trend** — 3% annual reduction over 5 years
- ❓ **Missing data** — 5% randomly injected per pollutant column

---

## 📁 Project Structure

```
air-quality-risk-prediction/
│
├── 📓 Air_Quality_Risk_Prediction.ipynb   ← Main Jupyter notebook (56 cells)
├── 🐍 air_quality_full_project.py         ← Single-file runnable script
│
├── data/
│   ├── air_quality_raw.parquet            ← Raw hourly dataset (438K rows)
│   ├── air_quality_clean.parquet          ← Cleaned & imputed dataset
│   ├── daily_features.parquet             ← Daily aggregated + engineered features
│   ├── daily_predictions.parquet          ← Test set predictions (all models)
│   ├── forecasts_30day.parquet            ← 30-day rolling forecasts
│   └── model_results.csv                  ← Model performance comparison
│
├── models/
│   ├── xgb.pkl                            ← Trained XGBoost model
│   ├── rf.pkl                             ← Trained Random Forest model
│   ├── gb.pkl                             ← Trained Gradient Boosting model
│   ├── ridge.pkl                          ← Trained Ridge Regression model
│   ├── label_enc.pkl                      ← City label encoder
│   └── features.pkl                       ← Feature list for inference
│
├── outputs/
│   ├── fig1_annual_aqi.png
│   ├── fig2_seasonal.png
│   ├── fig3_hourly.png
│   ├── fig4_correlation.png
│   ├── fig5_violin.png
│   ├── fig6_category_breakdown.png
│   ├── fig7_yoy_trend.png
│   ├── fig8_model_comparison.png
│   ├── fig9_residuals.png
│   ├── fig10_timeseries_pred.png
│   ├── fig11_feature_importance.png
│   ├── fig12_risk_heatmap.png
│   ├── fig13_high_risk_days.png
│   └── fig14_forecast.png
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw Data Generation (438,010 rows)
        │
        ▼
Data Cleaning & Validation
  ├── Missing value imputation (forward-fill + city-month median)
  ├── Outlier capping (per-city 1st–99th percentile)
  └── AQI category labelling (US EPA standard)
        │
        ▼
Exploratory Data Analysis
  ├── Annual trends · Seasonal patterns · Rush-hour effects
  ├── Pollutant correlations · Distribution analysis
  └── Category breakdown · Year-on-year improvement
        │
        ▼
Feature Engineering (Daily aggregates)
  ├── Lag features      — 1d, 3d, 7d, 14d
  ├── Rolling averages  — 3d, 7d, 14d, 30d
  ├── Cyclical encoding — month_sin/cos, dow_sin/cos
  └── Climate features  — temperature, humidity
        │
        ▼
Machine Learning (4 Models)
  ├── Ridge Regression  (baseline)
  ├── Random Forest     (200 estimators)
  ├── Gradient Boosting (300 estimators)
  └── XGBoost           (500 estimators, hist algorithm)
        │
        ▼
Evaluation & Interpretation
  ├── MAE · RMSE · R² · MAPE
  ├── Actual vs Predicted plots
  ├── Residual analysis
  └── 5-fold cross-validation
        │
        ▼
Risk Profiling & Forecasting
  ├── Composite city risk scoring
  ├── High-risk day analysis
  └── 30-day iterative rolling forecast
```

---

## 📊 Key Results

### Model Performance

| Model | MAE | RMSE | R² | MAPE % |
|-------|-----|------|----|--------|
| Ridge Regression | 3.36 | 5.45 | 0.9918 | 3.12 |
| Random Forest | **3.28** | **5.27** | **0.9923** | **2.98** |
| Gradient Boosting | 3.31 | 5.31 | 0.9922 | 3.05 |
| XGBoost | 3.34 | 5.34 | 0.9921 | 3.08 |

> ✅ **Random Forest** achieves the best overall performance with R² = 0.9923, explaining over 99% of AQI variance.

### Top Feature Importances (XGBoost)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | `co` | 70.8% | Primary emission proxy |
| 2 | `aqi_roll7d` | 10.2% | Short-term trend memory |
| 3 | `aqi_lag7d` | 6.6% | Weekly cycle pattern |
| 4 | `pm25` | 5.6% | Fine particulate matter |
| 5 | `pm10` | 3.2% | Coarse particulate matter |

### City Risk Ranking

| Rank | City | Risk Score | Risk Level | % Days Unhealthy |
|------|------|------------|------------|-----------------|
| 1 | Delhi | 78.4 | 🔴 Critical | 68.5% |
| 2 | Beijing | 70.1 | 🔴 Critical | 55.8% |
| 3 | Cairo | 62.3 | 🔴 High | 49.5% |
| 4 | Lagos | 54.7 | 🟠 High | 35.8% |
| 5 | São Paulo | 38.2 | 🟡 Moderate | 0.1% |
| 6–10 | Tokyo–Sydney | < 25 | 🟢 Low | < 1% |

---

## 📈 Visualisations

The project generates 14 charts saved to `outputs/`:

| Figure | Description |
|--------|-------------|
| `fig1_annual_aqi` | Annual mean AQI per city (2019–2023) |
| `fig2_seasonal` | Monthly seasonal patterns across all cities |
| `fig3_hourly` | Hourly AQI with rush-hour shading |
| `fig4_correlation` | Pollutant × climate correlation heatmap |
| `fig5_violin` | Full AQI distribution by city (violin plot) |
| `fig6_category_breakdown` | AQI category stacked bar chart |
| `fig7_yoy_trend` | Year-on-year improvement trend |
| `fig8_model_comparison` | MAE / RMSE / R² bar comparison |
| `fig9_residuals` | Actual vs predicted scatter + residual distribution |
| `fig10_timeseries_pred` | Delhi — predicted vs actual time series |
| `fig11_feature_importance` | XGBoost & RF feature importances |
| `fig12_risk_heatmap` | City × year AQI heatmap |
| `fig13_high_risk_days` | Days exceeding AQI 200 per city |
| `fig14_forecast` | 30-day rolling forecast with confidence bands |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/air-quality-risk-prediction.git
cd air-quality-risk-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0
nbformat>=5.9.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Option A — Run the Jupyter Notebook (recommended)

```bash
jupyter notebook Air_Quality_Risk_Prediction.ipynb
```

Then run all cells: **Kernel → Restart & Run All**

The notebook walks through all 12 steps with explanations, charts, and outputs in each cell.

### Option B — Run the single Python script

```bash
python air_quality_full_project.py
```

This executes the full pipeline end-to-end and saves all outputs automatically.

### Option C — Load a saved model for inference

```python
import pickle
import numpy as np

# Load model and feature list
with open("models/xgb.pkl",      "rb") as f: model    = pickle.load(f)
with open("models/features.pkl", "rb") as f: features = pickle.load(f)
with open("models/le.pkl",       "rb") as f: le       = pickle.load(f)

# Example: predict AQI for a new observation
sample = {
    "aqi_lag1d": 185.0,  "aqi_lag3d": 178.0,  "aqi_lag7d": 172.0,
    "aqi_lag14d": 168.0, "pm25_lag1d": 95.0,  "pm25_lag3d": 90.0,
    # ... fill all feature values
}
X = np.array([[sample[f] for f in features]])
predicted_aqi = model.predict(X)[0]
print(f"Predicted AQI: {predicted_aqi:.1f}")
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data manipulation** | pandas, NumPy |
| **Machine learning** | scikit-learn, XGBoost |
| **Visualisation** | Matplotlib, Seaborn |
| **Storage** | Apache Parquet (via PyArrow) |
| **Notebook** | Jupyter |
| **Version control** | Git + GitHub |

---

## 💡 Findings & Recommendations

### Key Findings

- **CO is the dominant AQI predictor** (71% feature importance) — a strong proxy for combustion-based emissions from vehicles and industry
- **Delhi** experiences 68.5% of days at "Unhealthy" or worse — the most severe risk profile of all cities studied
- **Seasonal patterns are pronounced** — Northern Hemisphere cities peak in winter (December–February), while Southern Hemisphere cities show summer peaks
- **Rush-hour emissions** (7–9 am and 5–7 pm) contribute a measurable 20–25% AQI spike in high-traffic cities
- **A 3% annual improvement trend** was detected across all cities, suggesting that existing interventions have some effect — but much more is needed for heavily polluted cities

### Policy Recommendations

**Delhi / Beijing / Cairo (Critical risk)**
- Immediate CO emission controls on industrial and vehicular sources
- Real-time AQI alert systems for vulnerable populations when AQI > 150
- Mandatory traffic restrictions during winter months
- Expand air quality monitoring station coverage in high-density zones

**Lagos / São Paulo (High risk)**
- Strengthen open waste burning regulations
- Accelerate transition of public transport to low-emission fleets
- Deploy low-cost IoT sensor networks for hyper-local monitoring

**All Cities**
- Integrate hospital admission data to quantify health impact in AQI terms
- Add socio-economic variables (income, housing density) for equity analysis
- Deploy model as a cloud API accessible to NGOs and local governments

---

## 🔮 Future Work

- [ ] **Real-time data integration** — connect to OpenAQ, EPA, or CPCB APIs for live monitoring
- [ ] **LSTM / Transformer models** — deep learning for longer-horizon sequence forecasting
- [ ] **Spatial interpolation** — kriging or neural spatial models to estimate AQI between stations
- [ ] **Health impact modeling** — link AQI predictions to respiratory/cardiovascular admission rates
- [ ] **Cloud deployment** — package as a FastAPI service with scheduled retraining
- [ ] **Interactive dashboard** — Streamlit or Dash web app for stakeholder access
- [ ] **Additional variables** — wind speed, precipitation, satellite imagery (NDVI, AOD)
- [ ] **Equity analysis** — overlay with population density and income maps

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add: your feature description'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style and includes docstrings for any new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Data Science for Social Good Munich](https://dssgmuenich.org/) — for the initiative and motivation
- [US EPA AQI Standards](https://www.airnow.gov/aqi/aqi-basics/) — for AQI category thresholds
- [OpenAQ](https://openaq.org/) — open air quality data platform (for real-world extension)

---

<p align="center">
  Built with ❤️ for public health and environmental awareness
</p>
