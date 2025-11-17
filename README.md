# Deep Learning + Alternative Data for Volatility Forecasting

This project builds a full volatility forecasting pipeline using both 
**classical econometric models** and **modern deep learning architectures**, 
enhanced with **alternative data** sources such as sentiment and news volume.

It is designed with:
- Clean data engineering  
- Strong baseline models (RV, HAR, GARCH)  
- Deep learning architectures (LSTM, Transformer)  
- Ablation testing  
- Clear, reproducible results  

---

## Project Overview

Financial volatility is not constant - it clusters, reacts to shocks, and 
is influenced by external information (news, sentiment, macro uncertainty).  
This project forecasts **next-day realized volatility (RV)** using:

### Classical models  
- **RV (Realized Volatility)**
- **HAR-RV (Heterogeneous Autoregressive Model)**
- **GARCH(1,1)**

### Deep learning models  
- **LSTM (Long Short-Term Memory Networks)**
- **Transformers with self-attention**

### Alternative data  
- Sentiment (FinBERT / social media)
- News volume
- Google Trends
- Market attention signals

---

## Repository Structure

```
volatility-forecasting/
│
├── data/
│ ├── raw/ # Unmodified downloaded data
│ └── processed/ # Cleaned + engineered features
│
├── notebooks/
│ ├── 01_data_download_and_cleaning.ipynb
│ ├── 02_compute_RV.ipynb
│ ├── 03_garch_modeling.ipynb
│ ├── 04_har_baseline.ipynb 
│ ├── 05_feature_engineering.ipynb
│ ├── 06_lstm_model.ipynb
│ ├── 07_lstm_with_alt_data.ipynb
│ ├── 08_transformer_model.ipynb
│ └── 09_ablation_tests.ipynb
│
├── src/
│ ├── data/ # Data loading + cleaning functions
│ ├── models/ # ML/DL model code
│ └── utils/ # Metrics, plotting, helpers
│
├── results/
│ ├── figs/ # Plots for analysis
│ └── tables/ # RMSE, QLIKE, performance tables
│
├── models/ # Saved model weights
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Methodology

### Realized Volatility (RV)
Compute daily realized volatility using intraday/high-frequency data or 
proxy from close-to-close log-returns.

### HAR-RV Model
Captures multi-horizon structure in volatility:
- Daily RV  
- Weekly RV  
- Monthly RV  

Represents long-memory behavior in volatility.

### GARCH(1,1)
Models volatility as a function of past shocks and past volatility:

\[
h_{t+1} = \omega + \alpha \varepsilon_t^2 + \beta h_t
\]

### LSTM
Uses a sequence of features (returns, RV, GARCH outputs, alt-data) to 
capture nonlinear temporal dependencies.

### Transformer
Uses self-attention to learn which past days matter most, capturing:
- Regime shifts  
- Sudden jumps  
- Long-range dependencies

---

## Alternative Data Sources

- **Sentiment** (FinBERT, social media polarity)
- **News volume** (# of articles per day)
- **Google Trends** search intensity
- **Volume & liquidity metrics**

These help detect **pre-shock information flow**, improving forecasts.

---

## Experimental Setup

- **Train/Test**: Rolling-window or walk-forward validation  
- **Metrics**:
  - RMSE  
  - MAE  
  - QLIKE (industry-standard for volatility forecasting)  
- **Baselines**: HAR, GARCH  
- **Deep Learning**: LSTM, Transformer  
- **Ablations**:
  - ML without alt-data  
  - ML + alt-data  
  - Transformer with vs without attention  

---

## Results (to be filled later)

- Error tables  
- Predicted vs actual RV plots  
- Attention heatmaps  
- Feature importance maps  

---

## Notebooks

### 01 – Data Download & Cleaning (VIX)
- **Notebook:** `notebooks/01_data_download_cleaning.ipynb`
- **Goal:** Download and clean historical daily VIX data as the base dataset for volatility modeling.
- **Data source:** Yahoo Finance (`^VIX`) via `yfinance`.
- **Key steps:**
  - Download daily OHLCV data starting from 2013-01-01.
  - Standardize column names and ensure a proper `Date` index.
  - Handle missing values and drop non-trading days if needed.
- **Output:** Cleaned daily VIX dataset saved to `data/raw/VIX_daily.csv`.

### 02 – Realized Volatility (RV) Estimation
- **Notebook:** `notebooks/02_realized_volatility.ipynb`
- **Goal:** Compute multiple realized volatility estimators from cleaned daily VIX data to form the target variable for classical and deep learning volatility models.
- **Data source:** Cleaned daily VIX dataset from Notebook 01 (`data/raw/VIX_daily.csv`).
- **Key steps:**
  - Compute log returns, squared returns, and absolute returns.
  - Generate rolling realized volatility using 5-day and 21-day windows.
  - Implement range-based estimators (Parkinson volatility) for improved accuracy over returns-only measures.
  - Visualize different RV estimators and compare their behavior over time.
- **Outputs:**
  - Realized volatility features saved to `data/processed/realized_vol.csv`.
  - Time-series plots of RV (returns-based vs range-based).

### 03 – GARCH(1,1) Volatility Modeling
- **Notebook:** `notebooks/03_garch_modeling.ipynb`
- **Goal:** Build a classical GARCH(1,1) model to capture volatility clustering in the VIX and compare it against realized volatility benchmarks.
- **Why GARCH?**  
  Volatility in financial markets is persistent - high-vol days tend to be followed by high-vol days.  
  GARCH(1,1) models this using past volatility and past squared shocks:
  - captures clustering  
  - captures mean-reversion  
  - widely used in risk models and volatility forecasting  
- **Key steps:**
  - Compute VIX log-returns.
  - Fit a **GARCH(1,1)** model with Student-t distribution.
  - Extract in-sample conditional volatility.
  - Perform **rolling 1-step-ahead forecasts** on the test set (walk-forward approach).
  - Compare GARCH volatility forecasts with 21-day realized volatility (RV_21).
  - Evaluate performance using MSE and MAE.
  - Conduct residual diagnostics (standardized residuals, histogram).
- **Model equation:**

\[
\sigma_{t+1}^{2} = \omega + \alpha \varepsilon_t^2 + \beta \sigma_t^2
\]

where  
- \( \varepsilon_t^2 \) = past shock  
- \( \sigma_t^2 \) = past volatility  
- Student-t innovations are used to model fat tails in VIX returns
- **Outputs:**
  - GARCH forecast file: `data/processed/garch_forecasts.csv`
  - Forecast comparison plots and diagnostics saved under `results/figs/`
- **Why it matters:**  
  GARCH is the industry standard for volatility modeling in:
  - derivatives pricing  
  - portfolio risk modeling  
  - volatility forecasting research  
  It serves as a strong **baseline** before moving into HAR, LSTM, and Transformer models.

### 04 – HAR-RV (Heterogeneous Autoregressive) Baseline
- **Notebook:** `notebooks/04_har_baseline.ipynb`
- **Goal:** Build a strong classical benchmark model for volatility forecasting using multiple time horizons of realized volatility.
- **Why HAR?**  
  HAR-RV is widely used in both academia and industry because financial volatility exhibits *long memory*.  
  Instead of modeling only yesterday’s volatility (like GARCH), HAR uses:
  - **Daily RV (1-day)**
  - **Weekly RV (5-day average)**
  - **Monthly RV (22-day average)**  
  capturing short-, medium-, and long-term components of volatility.
- **Key steps:**
  - Construct RV lag features:
    - `RV_d = RV(t-1)`
    - `RV_w = mean(RV(t-1 … t-5))`
    - `RV_m = mean(RV(t-1 … t-22))`
  - Fit the HAR model using OLS on **log(RV)**.
  - Generate in-sample fit and out-of-sample 1-step-ahead forecasts.
  - Compare predicted vs actual RV.
  - Compute MSE and MAE for evaluation.
- **Model equation:**

\[
\log(RV_t)
= \beta_0 
+ \beta_d \log(RV_{t-1})
+ \beta_w \log\left(\frac{1}{5}\sum_{i=1}^{5}RV_{t-i}\right)
+ \beta_m \log\left(\frac{1}{22}\sum_{i=1}^{22}RV_{t-i}\right)
+ \epsilon_t
\]

- **Outputs:**
  - HAR forecast file: `data/processed/har_forecasts.csv`
  - Plots: HAR prediction vs actual RV (saved under `results/figs`)
- **Why it matters:**  
  HAR-RV is a **high-quality baseline** used in high-frequency volatility research.  
  Deep learning models (LSTM, Transformers) should beat HAR to be considered meaningful improvements.

### 05 – Feature Engineering
- **Notebook:** `notebooks/05_feature_engineering.ipynb`
- **Why Feature Engineering?**  
  Deep learning models require richer input signals than classical models.  
  Rather than predicting volatility from a single RV series, we create a diverse set of features capturing:
  - return patterns  
  - volatility persistence  
  - high–low range information  
  - market regimes  
  - macro-like behaviors encoded inside VIX  
  - predictions from GARCH/HAR (meta-features)
- **Goal:** Build a full feature matrix for machine learning models (LSTM, Transformer, classical ML baselines).
- **Key steps:**
  - Merge clean OHLCV + realized volatility data.
  - Construct log-returns and lagged return features.
  - Add lagged realized-volatility features (1, 2, 5, 10, 21 days).
  - Rolling statistics on returns and RV:
    - mean, std, min, max over 5, 10, 21, 42 days
  - Range-based volatility estimators:
    - Parkinson  
    - Garman–Klass  
  - Regime indicators:
    - high-volatility regime flag (top 20%)  
    - volatility spike indicator  
    - return spike indicator  
  - VIX-specific macro proxies:
    - VIX level  
    - VIX % change  
    - volatility-of-volatility  
  - Add **GARCH** and **HAR** predictions as explanatory features (if available).
- **Outputs:**
  - Final ML-ready feature matrix:
    ```
    data/processed/features.csv
    ```
  - Contains all features for LSTM, Transformer, and ablation-study notebooks.
- **Why it matters:**  
  Strong feature engineering allows deep learning models to capture:
  - long-memory volatility structure  
  - nonlinear price dynamics  
  - volatility regimes  
  - early warning signals encoded in VIX movements  
  These features dramatically improve model performance over raw RV alone.

### 06 – LSTM Baseline
Deep learning volatility forecasting.

### 07 – LSTM w/ Alt Data
Sentiment, volume, optional macro signals.

### 08 – Transformer Model
Modern sequence modeling architecture.

### 09 – Ablation Studies
Feature importance, model component analysis.

---

## How to Run

1. Install dependencies:
```
   pip install -r requirements.txt
```
2. Launch Jupyter Lab:

   jupyter lab

3. Run notebooks in order:

   01_data_download_and_cleaning.ipynb  
   02_compute_RV.ipynb  
   03_garch_modeling.ipynb  
   04_har_baseline.ipynb  
   05_feature_engineering.ipynb  
   06_lstm_model.ipynb  
   07_lstm_with_alt_data.ipynb  
   08_transformer_model.ipynb  
   09_ablation_tests.ipynb   

---

## Author

Anh Hong Truong  
Quantitative Finance & Data Science  
GitHub: https://github.com/truonghonganh165