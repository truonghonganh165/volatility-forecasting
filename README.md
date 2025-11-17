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
│ ├── 03_HAR_baseline.ipynb
│ ├── 04_GARCH_baseline.ipynb
│ ├── 05_feature_engineering.ipynb
│ ├── 06_LSTM_model.ipynb
│ ├── 07_LSTM_with_alt_data.ipynb
│ ├── 08_Transformer_model.ipynb
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
   03_HAR_baseline.ipynb  
   04_GARCH_baseline.ipynb  
   05_feature_engineering.ipynb  
   06_LSTM_model.ipynb  
   07_LSTM_with_alt_data.ipynb  
   08_Transformer_model.ipynb  
   09_ablation_tests.ipynb  

---

## Author

Anh Hong Truong  
Quantitative Finance & Data Science  
GitHub: https://github.com/truonghonganh165