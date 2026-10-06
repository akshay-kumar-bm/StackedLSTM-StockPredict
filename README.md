# StackedLSTM-StockPredict

A stacked (bidirectional) LSTM notebook that forecasts the closing price of a single NSE-listed stock (the TATAGLOBAL dataset) and rolls the forecast 100 days ahead.

## What it does
Given a daily OHLC history, the notebook predicts the next day's `Close` price from the previous 100 days and then feeds its own predictions back in to produce a 100-day forecast.

## Dataset
- `data.csv` (also loaded in the notebook from a public GitHub CSV, `NSE-TATAGLOBAL.csv`): 2,035 daily rows with `Date, Open, High, Low, Last, Close, Total Trade Quantity, Turnover (Lacs)`; the first row is 2018-09-28.
- Only the `Close` column is modelled.

## Approach (as implemented in `Stock_Price_Prediction.ipynb`)
1. Load CSV with pandas, select `Close`.
2. Scale to [0, 1] with `MinMaxScaler`.
3. Sequential 75/25 split (1,526 train rows / remainder test, no shuffling).
4. Build supervised windows with `time_step = 100` -> X_train (1425, 100), X_test (408, 100).
5. Model (Keras / TensorFlow):
   - `LSTM(50, return_sequences=True)`
   - `Bidirectional(LSTM(50, return_sequences=True))`
   - `Bidirectional(LSTM(50))`
   - `Dense(1)`
   - Loss: mean absolute error, optimizer: Adam, 20 epochs, batch size 64, validation on the test windows.
6. Inverse-transform predictions, plot train/test predictions against the actual series.
7. Recursive forecast of the next 100 days and plot.

## Results shown in the notebook
- Mean absolute error on the test set (inverse-scaled): about 3.01 price units.
- A `sqrt(MAE)` figure of about 1.74 is also printed (note: this is the square root of the MAE, not an RMSE).
- Plots of train/test fit and the 100-day forecast are embedded as notebook outputs. No baseline comparison or walk-forward validation is included.

## Tech stack
Python, pandas, NumPy, scikit-learn (MinMaxScaler, metrics), TensorFlow/Keras, Matplotlib. Developed in a Colab-style environment.

## Project structure
```
Stock_Price_Prediction.ipynb   # the full pipeline
data.csv                       # dataset copy
Stock_price_prediction.pdf     # PDF export of the notebook/report
Recording 2024-04-10 114957.mp4  # screen recording (about 20 MB)
```
There is no requirements file.

## Run
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib jupyter
jupyter notebook Stock_Price_Prediction.ipynb
```

## Limitations / future work
- Single feature (Close) and single stock; no technical indicators or volume.
- Recursive 100-day forecasting accumulates error and is illustrative, not a trading signal.
- Data is sorted newest-first in the CSV and the notebook does not re-sort by date before windowing, so the temporal ordering used for the split should be checked.
- Add a naive baseline, walk-forward evaluation and a requirements file.
