# Electricity Load Forecasting

Forecast hourly electricity demand using historical load, weather variables, and calendar signals. The project is organized as a reproducible machine learning workflow with simple baselines, an LSTM model, and a final test split kept separate from validation.

## Objective

Build a forecasting pipeline that predicts the next hour of national electricity demand (`nat_demand`) and compares the LSTM against interpretable baseline models. The goal is to show practical time-series modeling discipline: chronological splits, leakage-aware scaling, baseline comparison, model evaluation, and clear reporting.

## Dataset

- **Source:** [Electricity Load Forecasting on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting/data)
- **Main file:** `data/raw/continuous_dataset.csv`
- **Frequency:** hourly observations
- **Date range:** 2015-01-03 01:00 to 2020-06-27 00:00
- **Rows:** 48,048
- **Target:** `nat_demand`
- **Inputs:** weather measurements from three locations, holiday flags, school flags, and engineered calendar features.

The raw data is kept outside version control. Download the Kaggle dataset and place `continuous_dataset.csv` in `data/raw/`.

## Problem Framing

This is a one-step-ahead regression problem:

- Input: the previous 168 hours of features
- Output: electricity demand for the next hour
- Split strategy: chronological `train / validation / test`
- Scaling: fitted only on the training split
- Test usage: final evaluation only

## Features

The model uses:

- Historical demand: `nat_demand`
- Weather variables: temperature, humidity, liquid precipitation, and wind speed across the available locations
- Calendar signals: hour, day of week, month, weekend flag
- Operational flags: holiday and school indicators

Calendar features are encoded cyclically with sine/cosine transforms.

## Models

Implemented models:

- `naive_previous_day`: demand from the same hour one day earlier
- `seasonal_naive_previous_week`: demand from the same hour one week earlier
- `linear_regression`: tabular model with lag and rolling demand features
- `random_forest`: nonlinear tabular baseline
- `xgboost`: optional tabular baseline when `xgboost` is installed
- `lstm`: sequence model trained on the training split and selected by validation loss

## Current Baseline Metrics

Baseline metrics are computed on the untouched final test split.

| model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| random_forest | 23.96 | 34.54 | 2.02% |
| linear_regression | 29.68 | 38.77 | 2.51% |
| seasonal_naive_previous_week | 71.14 | 99.16 | 5.94% |
| naive_previous_day | 73.33 | 110.60 | 5.99% |

After training the LSTM, run the evaluation script again to add the neural model to `reports/metrics.csv` and generate the forecast plot in `reports/figures/`.

## Project Structure

```text
Electricity-Load-Forecasting/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|   |-- 01_exploratory_data_analysis.ipynb
|   |-- 02_data_preparation_and_windowing.ipynb
|   `-- 03_kaggle_lstm_training.ipynb
|-- src/
|   |-- data.py
|   |-- features.py
|   |-- baselines.py
|   |-- models.py
|   |-- train.py
|   `-- evaluate.py
|-- app/
|   `-- streamlit_app.py
|-- models/
`-- reports/
    `-- figures/
```

## How to Run

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it here:

```text
data/raw/continuous_dataset.csv
```

Train the LSTM:

```bash
python -m src.train --epochs 30
```

Evaluate baselines and the trained LSTM on the final test split:

```bash
python -m src.evaluate
```

Open the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Outputs:

- `models/lstm_forecast.pt`: trained model checkpoint
- `models/lstm_forecast.json`: model metadata
- `reports/metrics.csv`: model comparison table
- `reports/figures/lstm_test_forecast.png`: actual vs forecast plot

## Notebooks

The notebooks are kept for exploration and explanation:

- `01_exploratory_data_analysis.ipynb`: dataset inspection and demand patterns
- `02_data_preparation_and_windowing.ipynb`: initial LSTM windowing workflow
- `03_kaggle_lstm_training.ipynb`: Kaggle-oriented LSTM training notebook

For reproducible project runs, prefer the scripts in `src/`.

## Denmark and Energinet Extension

This version uses a Kaggle electricity demand dataset. For Denmark-focused energy roles, the natural next version is to reuse the same pipeline with Danish electricity consumption data from Energinet. That would make the problem framing more directly relevant to Nordic grid operations, day-ahead planning, renewable integration, and Danish market context.

## Next Steps

- Add direct 24-hour multi-output forecasting.
- Compare recursive 24-hour LSTM forecasts against direct multi-output forecasts.
- Add validation-based hyperparameter search for the tabular baselines and LSTM.
- Expand reporting with residual analysis by hour, weekday, weekend, and holiday.
- Adapt the workflow to Danish electricity data from Energinet.
