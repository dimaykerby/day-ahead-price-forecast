# Day-Ahead Electricity Price Forecasting

A production-style ML pipeline that benchmarks **LightGBM**, **Gaussian Process regression**, and **LSTM** on hourly day-ahead electricity prices across six European markets, served via an interactive **Streamlit** application.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Key Techniques](#key-techniques)
  - [Dynamic Feature Scaling](#dynamic-feature-scaling)
  - [Feature Engineering](#feature-engineering)
  - [Models](#models)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Train & Benchmark](#train--benchmark)
  - [Streamlit App](#streamlit-app)
- [Results](#results)
- [Extending the Project](#extending-the-project)

---

## Problem Statement

Electricity markets clear prices one day in advance through an auction mechanism. Accurate day-ahead price forecasting (DAFP) is critical for:

- **Trading desks** — optimise bid/offer strategies
- **Energy retailers** — hedge procurement costs
- **Renewable operators** — value storage dispatch and curtailment decisions

The 2022 European energy crisis (gas supply disruption → prices spiking from ~50 to >800 €/MWh) makes this dataset a particularly challenging non-stationary benchmark that exposes weaknesses in models relying on static feature normalisation.

---

## Dataset

**ENTSO-E Day-Ahead Prices 2022** — sourced from [Kaggle / Henri Upton](https://www.kaggle.com/datasets/henriupton/electricity-dayahead-prices-entsoe).

| Column | Description |
|---|---|
| `date` | Date in `YYYY/MM/DD` format |
| `hour` | Hourly interval (e.g. `14:00 - 15:00`) |
| `france` | French day-ahead price (€/MWh) |
| `italy` | Italian day-ahead price (€/MWh) |
| `belgium` | Belgian day-ahead price (€/MWh) |
| `spain` | Spanish day-ahead price (€/MWh) |
| `uk` | UK day-ahead price (£/MWh) |
| `germany` | **Default target** — German day-ahead price (€/MWh) |

- **8 760 hourly observations** covering full-year 2022
- Price range: −19 to 871 €/MWh (Germany); **highly non-stationary**
- DST duplicate timestamp deduplicated; spring-forward gap forward-filled

---

## Key Techniques

### Dynamic Feature Scaling

Standard global normalisation (fit once on training set) fails on non-stationary price data: statistics computed over 2021 low-price regimes are meaningless when applied to Q3 2022 crisis prices.

The solution — **rolling z-score normalisation** — computes mean and standard deviation over a trailing window, with a mandatory 1-step shift to prevent future leakage:

```
z_t = (price_t − μ_{t−1, window}) / (σ_{t−1, window} + ε)
```

Two window sizes are used: **24 hours** (intraday regime) and **168 hours** (weekly regime). The resulting features directly inform the LightGBM model about the current price level and its variability — the rolling features rank as the top-5 most important by gain.

### Feature Engineering

All features are computed without external APIs:

| Feature Group | Features |
|---|---|
| **Calendar (cyclical)** | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend` |
| **Lag features** | `lag_1h`, `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, `lag_24h`, `lag_48h`, `lag_168h` |
| **Dynamic scaling** | `roll_zscore_24h`, `roll_zscore_168h`, `roll_mean_24h/168h`, `roll_std_24h/168h` |
| **Solar proxy** | Deterministic irradiance proxy from hour-of-day × month seasonality |
| **Demand proxy** | Double-peak weekday load profile (morning + evening), weekend-scaled |
| **Cross-market spreads** | Germany–France/Italy/Belgium/Spain/UK spreads (lagged 1h) |
| **Cross-market lags** | 24h-lagged prices for neighbouring countries |

Cyclical encoding (sin/cos) prevents the model from treating hour 23 and hour 0 as maximally distant.

### Models

| Model | Implementation | Notes |
|---|---|---|
| **LightGBM** | `lightgbm.LGBMRegressor` | Full 33-feature tabular input; early stopping on 10% validation hold-out; Huber-equivalent via asymmetric loss |
| **Gaussian Process** | `sklearn.GaussianProcessRegressor` | RBF + White kernel; trained on last 600 points (O(n³) cost); returns calibrated **uncertainty intervals** |
| **LSTM** | PyTorch 2-layer LSTM | 24-step input sequence → 1-step forecast; Huber loss; gradient clipping; ReduceLROnPlateau scheduler |

The **GP model** provides the only probabilistic forecast with 95% confidence intervals — valuable for risk management use cases. The **LSTM** captures temporal dependencies implicitly through the sequence rather than requiring hand-crafted lags.

---

## Project Structure

```
day_ahead_forecast/
├── data/
│   └── electricity_dah_prices.csv   # ENTSO-E 2022 hourly prices
├── src/
│   ├── __init__.py
│   ├── features.py                  # feature engineering + dynamic scaling
│   └── models.py                    # LightGBM, GP, LSTM wrappers
├── models/                          # serialised artefacts (git-ignored)
├── app.py                           # Streamlit app (3 tabs)
├── train.py                         # CLI benchmark runner
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd day_ahead_forecast

# 2. Create and activate the pyenv environment
pyenv virtualenv 3.12.12 day_ahead_forecast
pyenv local day_ahead_forecast

# 3. Install dependencies
pip install -r requirements.txt
```

> PyTorch CPU-only installation (~200 MB). For GPU support replace the torch line in requirements.txt with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

### Train & Benchmark

```bash
# Train all three models on Germany (default) — ~5 min on CPU
python train.py

# Choose a different target country
python train.py --target france

# Train only LightGBM and GP (skip LSTM)
python train.py --models lgbm gp

# Full options
python train.py --data data/electricity_dah_prices.csv \
                --target germany \
                --models-dir models \
                --models lgbm gp lstm
```

**Example output:**

```
════════════════════════════════════════
  Benchmark results — target: germany
  Train: 6873 rows  |  Test: 1719 rows
────────────────────────────────────────
  Model       MAE (€/MWh)  RMSE (€/MWh)
────────────────────────────────────────
  lgbm              11.52         22.41
  gp                28.14         45.89
  lstm              18.73         32.05
════════════════════════════════════════
```

Serialised models are saved to `models/` and a `benchmark_{target}.json` summary is written for the app.

### Streamlit App

```bash
# The app must be launched from the project root
streamlit run app.py
```

Open `http://localhost:8501`. The app has three tabs:

| Tab | Contents |
|---|---|
| **📊 Data Explorer** | Interactive price series, correlation matrix, hourly load profile, rolling volatility |
| **🏆 Model Benchmark** | One-click training, metric comparison bar chart, LightGBM feature importance |
| **🔮 Forecast** | Actual vs predicted overlay, 95% CI band (GP), scatter plot with OLS trendline, windowed MAE/RMSE |

> Run at least one `python train.py` before opening the Forecast tab.

### Monte Carlo BESS Arbitrage

You can extend the price forecast into a battery-investment style analysis:

1. Build residual-driven price scenarios around the point forecast
2. Dispatch a battery in each scenario with SoC and efficiency constraints
3. Annualize arbitrage revenues and compute P10/P50/P90
4. Run sensitivity cases and a simple NPV/IRR snapshot

Run:

```bash
# Ensure the project pyenv is active
pyenv activate day_ahead_forecast

# Train at least one model for the chosen target (example: Italy + LightGBM)
python train.py --target italy --models lgbm

# Monte Carlo arbitrage run (1000 scenarios, 4-day horizon)
python monte_carlo_arbitrage.py \
  --target italy \
  --model lgbm \
  --n-sims 1000 \
  --horizon-hours 96 \
  --error-model bootstrap
```

Example result from a validated run (`italy`, `lgbm`, `n-sims=1000`, `horizon=96h`):

- Annual revenue P10: ~543,832 EUR
- Annual revenue P50: ~734,623 EUR
- Annual revenue P90: ~814,051 EUR

Outputs are written under `outputs/`:

- `annual_revenue_distribution.csv` — one annualized revenue per scenario
- `annual_revenue_histogram.html` — histogram with P10/P50/P90 markers
- `sensitivity_table.csv` — base and stress cases (efficiency, spreads, availability)
- `finance_p50.csv` — simple project/equity NPV and IRR at P50 revenue
- `investment_memo.md` — short investment-style narrative summary

#### Error Model Options

- `bootstrap` (default): resample historical forecast residuals
- `garch`: fit GARCH(1,1) volatility on residuals and sample shocks with forecasted sigma

#### Dispatch Simplification

The default dispatch is intentionally simple and transparent:

- Daily greedy ranking of cheapest charge hours and highest discharge hours
- SoC band, power limit, and round-trip efficiency are respected
- Revenue is gross arbitrage spread before market fees and imbalance costs

This is a practical first benchmark before moving to a full LP/MILP optimizer.

---

## Results

Evaluated on the chronological 20% test set (last ~72 days of 2022, including the high-volatility Q4 period):

| Model | MAE (€/MWh) | RMSE (€/MWh) | Uncertainty |
|---|---|---|---|
| LightGBM | **~11.5** | **~22** | ✗ |
| LSTM | ~18–20 | ~32–36 | ✗ |
| Gaussian Process | ~28–32 | ~46–50 | ✓ (calibrated 95% CI) |

LightGBM dominates on point-forecast accuracy thanks to its ability to exploit the rich lag and dynamic-scaling feature set. The GP trades accuracy for calibrated uncertainty — useful when downstream decisions require risk quantification. The LSTM sits between the two, learning temporal patterns directly from sequences without relying on manual lag selection.

---

## Extending the Project

| Idea | How |
|---|---|
| **Real weather data** | Integrate [Open-Meteo API](https://open-meteo.com/) (free, no key) for temperature, wind speed, cloud cover per country |
| **Renewable generation** | Add ENTSO-E Transparency Platform generation mix (wind/solar actual) via the `entsoe-py` library |
| **Multi-step forecast** | Change the LSTM head to output 24 values for a full day-ahead horizon |
| **Quantile regression** | Replace LightGBM MSE objective with `quantile` to get prediction intervals without GP overhead |
| **XGBoost / CatBoost** | Drop-in replacements for the LightGBM wrapper |
| **Conformal prediction** | Wrap any model with `MAPIE` for distribution-free coverage guarantees |
