# 🏥 Calibrate Health — Data Analyst Portfolio

> **A end-to-end analytics work sample built for Calibrate Health's Junior Data Analyst role.**  
> Synthetic member cohort · Clinical KPIs · Survival Analysis · Bidirectional LSTM · Anomaly Detection · Executive Dashboard

---

## Overview

This notebook demonstrates the core analytical skills required for data work in a metabolic health company:
SQL-based KPI reporting, clinical outcome modeling, machine learning for churn prediction,
data quality monitoring, and cross-functional dashboard design.

All data is **synthetic** and generated to reflect realistic distributions informed by
published GLP-1 clinical trial benchmarks (STEP-1 / SURMOUNT-1).

---

## Business Context

[Calibrate](https://www.joincalibrate.com) is redefining obesity care as a matter of biology,
not willpower. Its program combines GLP-1 medication management, 1:1 coaching,
and app-based tracking across four metabolic health pillars:

| Pillar | Description |
|---|---|
| 🍎 Food | Daily nutritional logging and meal quality tracking |
| 😴 Sleep | Sleep duration and consistency monitoring |
| 🏃 Exercise | Physical activity logging and goal tracking |
| 🧠 Emotional Wellbeing | Mood and stress check-ins |

---

## What This Notebook Contains

### Section 1 — Synthetic Member Data Generation
- 5,000-member cohort with realistic demographics, comorbidities (T2D, hypertension),
  GLP-1 medication type (Semaglutide, Tirzepatide, None), coaching utilization,
  and four-pillar engagement rates
- Outcome simulation calibrated to published trial data

### Section 2 — SQL KPI Queries (SQLite)
- Weight loss achievement rates (≥10%, ≥15%) by channel × medication
- Engagement quartile performance
- Churn rates by segment
- Documented query definitions for transparency and reproducibility

### Section 3 — Clinical Outcomes: Statistical Modeling
- **Linear Mixed-Effects Model (REML)** — isolates the independent effect of
  medication type, coaching sessions, and engagement on % body weight lost,
  controlling for baseline BMI, age, T2D, and hypertension;
  random intercept by channel (DTC vs Enterprise)
- **Kaplan-Meier survival curves** — retention probability stratified by engagement quartile
- **Cox Proportional Hazards model** — hazard ratios for churn drivers

### Section 4 — Churn Prediction: Bidirectional LSTM
- 12-week rolling time-series of four-pillar engagement per member
- **Bidirectional LSTM** with Batch Normalization, Dropout, EarlyStopping,
  and ReduceLROnPlateau
- Identifies members at churn risk ~4 weeks before disengagement
- Evaluated with ROC-AUC (~0.87), classification report

### Section 5 — Data Quality: Isolation Forest
- Automated anomaly detection on member records
- Flags impossible BMI values, unrealistic weight loss, and erroneous coaching counts
- Prevents dirty data from propagating into KPI reports

### Section 6 — Executive KPI Dashboard (Plotly)
- Six-panel cross-functional scorecard
- Covers Clinical, Coaching, Operations, and Finance metrics
- Built with Plotly for interactive in-notebook rendering

### Section 7 — Summary & Recommendations
- Key findings table
- Actionable recommendations tied directly to Calibrate's program and values

---

## Key Results

| Metric | Result |
|---|---|
| Mean weight loss at 12m | 14.2% body weight (completer analysis) |
| ≥10% WL — Tirzepatide | 81% of completers |
| ≥10% WL — Semaglutide | 66% of completers |
| ≥10% WL — No GLP-1 | 22% of completers |
| Top churn predictor | Engagement score (HR = 0.18, p < 0.001, Cox PH) |
| BiLSTM ROC-AUC | ~0.87 |
| Coaching effect (LME) | +1.2% WL per session/month (β = 0.012, p < 0.001) |
| Anomaly detection rate | 2.1% of records flagged (Isolation Forest) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.10+` | Core language |
| `Pandas` / `NumPy` | Data manipulation |
| `SQLite` / `pd.read_sql` | SQL KPI queries |
| `Statsmodels` | Linear Mixed-Effects Model (REML) |
| `Lifelines` | Kaplan-Meier, Cox Proportional Hazards |
| `Scikit-learn` | Isolation Forest, preprocessing, evaluation |
| `TensorFlow / Keras` | Bidirectional LSTM churn model |
| `Plotly` | Interactive dashboard and charts |

---

## How to Run

### Option A — Google Colab (recommended, zero setup)
Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Option B — VS Code / Local
```bash
git clone https://github.com/YOUR_USERNAME/calibrate-data-analyst-portfolio.git
cd calibrate-data-analyst-portfolio
pip install pandas numpy scipy statsmodels lifelines scikit-learn tensorflow plotly
jupyter notebook Calibrate_Data_Analyst_Portfolio.ipynb
```

---

## Repository Structure

```
calibrate-data-analyst-portfolio/
│
├── Calibrate_Data_Analyst_Portfolio.ipynb   # Main notebook
├── Calibrate_Data_Analyst_Portfolio.md      # Rendered markdown with outputs
└── README.md                                # This file
```

---

## Disclaimer

All data used in this project is **entirely synthetic**.
No real patient data was used. This notebook is a portfolio demonstration only
and does not represent actual Calibrate member outcomes.

---

## Author

**[Your Name]**  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username) · your.email@gmail.com

---

*Built with curiosity, care for metabolic health, and a lot of coffee. ☕*
