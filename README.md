Gabrielle Kelsey
This repository includes 6 jupyter notebooks, most of which can be run independently (excluding 01_ETL_Processing.ipynb) after getting the data and installing the needed dependencies. A more detailed description on how to run this is located below.

---

# WESM Price Prediction

An end-to-end machine learning project for predicting electricity spot market prices in the Philippines. The target variable is the **Generator Weighted Average Price (GWAP)** for the Luzon region at 5-minute Real-Time Dispatch (RTD) intervals, sourced from the Wholesale Electricity Spot Market (WESM) operated by IEMOP.

---

## Overview

The project covers the full ML pipeline — from raw market data ingestion to model evaluation — and benchmarks five model families across both regression and classification formulations of the same underlying problem.

**Key findings:**
- The best regression model (Polynomial ElasticNet) achieves **RMSE ≈ ₱930/MWh, R² ≈ 0.675** on the held-out test set.
- The best classification model (Multiclass Logistic Regression) achieves **Macro F1 = 0.764, ROC-AUC = 0.947** for detecting Normal / Negative / Positive Spike price states.
- Across all architectures, the dominant predictor is `GWAP_Lag_1` (price 5 minutes ago), confirming GWAP is strongly autoregressive. LSTMs with residual stacking approach but do not surpass classical linear models on this dataset.
- Rare positive spikes (GWAP > ₱10,000/MWh) remain the hardest prediction target — 8–11× higher RMSE than the normal price range — revealing that the bottleneck is feature availability, not model complexity.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | `pandas`, `NumPy` |
| Machine learning | `scikit-learn` |
| Deep learning (NN) | `PyTorch` |
| Deep learning (LSTM) | `TensorFlow / Keras` |
| Visualization | `matplotlib` |
| Notebooks | `Jupyter` |

---

## Repository Structure

```
wesm-price-prediction/
│
├── 01_ETL_Preprocessing.ipynb         # Load, clean, and merge raw GWAP, RTD, and Outage CSVs
├── 02_EDA.ipynb                        # Exploratory data analysis and feature correlation
│
├── 03a_VanillaLinearRegression.ipynb  # Model 1: Linear Regression (baseline)
├── 03b_PolynomialRidge.ipynb          # Model 2: Polynomial Ridge & ElasticNet Regression
├── 04_MulticlassLogisticRegression.ipynb  # Model 3: Multiclass Logistic Regression (classifier)
├── 05_NeuralNetwork.ipynb             # Model 4: Custom feedforward Neural Network (PyTorch)
├── 06_LSTM.ipynb                      # Model 5: LSTM with hyperparameter tuning (TensorFlow)
│
├── data_laoder.py                     # Custom mini-batch DataLoader (NumPy, built from scratch)
├── neural_network.py                  # Configurable PyTorch nn.Module with dynamic layer construction
│
├── raw_data/                          # (not tracked) Raw CSVs from IEMOP
│   ├── GWAP/
│   ├── RTD_Regional/
│   └── Outages/
└── *.csv                              # (not tracked) Processed final dataset
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_ETL_Preprocessing` | Ingests GWAP, RTD Regional Summaries, and Outage Schedules. Filters to Luzon (`CLUZ`), engineers `energy_shortage_mw`, and outputs a single merged dataset. |
| `02_EDA` | Analyzes price distributions, temporal patterns, feature correlations, and documents negative GWAP events (a valid market phenomenon per WESM PDM). |
| `03a_VanillaLinearRegression` | Trains and evaluates a vanilla OLS baseline. Includes segmented error analysis by price regime. |
| `03b_PolynomialRidge` | Expands to degree-2 polynomial features and regularizes with Ridge and ElasticNet. Best classical regression model. |
| `04_MulticlassLogisticRegression` | Classifies each interval as Normal (0–₱5,000), Negative (≤ ₱0), or Positive Spike (> ₱5,000). Best overall classifier. |
| `05_NeuralNetwork` | Trains a configurable feedforward network using the custom `NeuralNetwork` and `DataLoader` classes. |
| `06_LSTM` | Trains an LSTM with hyperparameter search over lookback window and hidden units. Also tests residual stacking, price-change decomposition, and an LSTM classifier. |

---

## Features

The 16 input features used across models:

| Feature | Description |
|---|---|
| `energy_demand_mw` | MW energy requirement (demand) |
| `energy_supply_mw` | MW energy generation (supply) |
| `reserve_demand_mw` | MW reserve requirement |
| `reserve_supply_mw` | MW reserve availability |
| `outage_count` | Number of scheduled generator outages |
| `GWAP_Lag_1` | GWAP 5 minutes ago (strongest predictor) |
| `GWAP_Lag_12` | GWAP 1 hour ago |
| `GWAP_Lag_288` | GWAP 24 hours ago |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour-of-day |
| `dow_sin`, `dow_cos` | Cyclical encoding of day-of-week |
| `month_sin`, `month_cos` | Cyclical encoding of month |
| `is_weekend` | Binary flag for weekends |
| `is_peak_hour` | Binary flag for peak demand hours |

> `energy_shortage_mw` (demand − supply) is excluded from model inputs due to perfect multicollinearity with the demand and supply features.

---

## Model Results (Test Set)

### Regression

| Model | RMSE (₱/MWh) | MAE (₱/MWh) | R² |
|---|---|---|---|
| Linear Regression | 934.47 | 459.48 | 0.671 |
| Polynomial Ridge | 972.19 | 552.92 | 0.644 |
| **Polynomial ElasticNet** | **930.09** | **471.71** | **0.675** |
| LSTM – Baseline | ~1,186 | — | ~0.473 |
| LSTM – Residual Stacking | ~924 | — | ~0.677 |
| LSTM – Price Change | ~1,025 | — | ~0.606 |

### Classification (Normal / Negative / Positive Spike)

| Model | Macro F1 | ROC-AUC |
|---|---|---|
| **Multiclass Logistic Regression** | **0.764** | **0.947** |
| LSTM Classifier | 0.578 | 0.882 |

---

## Getting Started

### 1. Get the Data

The project uses 5 months of WESM data (Oct 2025 – Feb 2026). The IEMOP website only hosts the latest 3 months, so use the archived link:

**[Download Full Dataset (Google Drive)](https://drive.google.com/drive/folders/1bBBfN9HHeusgjcAjRgM8dC-t4AMN0ijE?usp=sharing)**

- To run notebooks **02 onwards**, download `final_dataset_csv` and place the extracted CSV in the root folder.
- To run **01_ETL_Preprocessing** from scratch, download the `raw_data` folder and extract it as:

```
raw_data/
├── GWAP/
├── RTD_Regional/
└── Outages/
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib torch tensorflow jupyter
```

### 3. Run the Notebooks

Open notebooks in order (01 → 06) in Jupyter, or skip to any model notebook using the pre-processed CSV:

```bash
jupyter notebook
```

---

## Data Sources

| Source | Description | Link |
|---|---|---|
| GWAP Final | Generator Weighted Average Price by interval | [IEMOP](https://www.iemop.ph/market-data/generator-weighted-average-price-final/) |
| RTD Regional Summaries | Energy and reserve demand/supply by region | [IEMOP](https://www.iemop.ph/market-data/rtd-regional-summaries/) |
| Outage Schedules | Generator outages used in RTD dispatch | [IEMOP](https://www.iemop.ph/market-data/outage-schedules-used-in-rtd/) |

### Official Resources

- **IEMOP Market Data Dashboard:** https://www.iemop.ph/market-data/
- **WESM Price Determination Methodology (PDM):** [Download PDF](https://www.wesm.ph/downloads/download/TWFya2V0IFJlcG9ydHM=/NTYw)
- **WESM Compliance Bulletins:** https://www.wesm.ph/market-reports/compliance-bulletins/
