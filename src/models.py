"""Forecasting models. Every model produces one-step-ahead (next hour) forecasts over the
test period, always using the true history up to the hour being predicted.

    seasonal_naive  -> baseline: same hour yesterday
    sarima_forecast -> SARIMA with a 24-hour season
    lstm_forecast   -> LSTM on a sliding window of past hours
    hybrid_forecast -> SARIMA for the linear/seasonal part + LSTM on SARIMA's residuals
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

SEASON = 24      # hours in a day
WINDOW = 48      # hours of history the LSTM sees


def seasonal_naive(train: pd.Series, test: pd.Series) -> np.ndarray:
    full = pd.concat([train, test])
    return full.shift(SEASON).loc[test.index].to_numpy()


# --------------------------------------------------------------------------- SARIMA

def _fit_sarima(train: pd.Series):
    model = SARIMAX(
        train, order=(1, 0, 1), seasonal_order=(1, 1, 1, SEASON),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=50)


def sarima_forecast(train: pd.Series, test: pd.Series):
    """Returns (test predictions, in-sample train predictions).

    Parameters are estimated on train only, then the fitted model is rolled over the
    test period so each prediction uses the real observations before it.
    """
    fitted = _fit_sarima(train)
    train_pred = fitted.predict(start=train.index[0], end=train.index[-1]).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extended = fitted.append(test, refit=False)
    test_pred = extended.predict(start=test.index[0], end=test.index[-1]).to_numpy()
    return np.clip(test_pred, 0, None), train_pred


# ----------------------------------------------------------------------------- LSTM

class _LSTM(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _windows(values: np.ndarray, start: int):
    """Build (X, y) where y[i] = values[start + i] and X[i] = the WINDOW values before it."""
    xs = np.stack([values[i - WINDOW:i] for i in range(start, len(values))])
    ys = values[start:]
    return (
        torch.tensor(xs, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(ys, dtype=torch.float32),
    )


def _lstm_on_array(values: np.ndarray, n_train: int, epochs: int, seed: int) -> np.ndarray:
    """Train on values[:n_train], predict values[n_train:]. Scaling uses train stats only."""
    torch.manual_seed(seed)
    mean, std = values[:n_train].mean(), values[:n_train].std() + 1e-8
    scaled = (values - mean) / std

    x_train, y_train = _windows(scaled[:n_train], WINDOW)
    x_test, _ = _windows(scaled, n_train)

    model = _LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(x_train[idx]), y_train[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        return model(x_test).numpy() * std + mean


def lstm_forecast(train: pd.Series, test: pd.Series, epochs: int = 40, seed: int = 0):
    values = np.concatenate([train.to_numpy(), test.to_numpy()])
    return np.clip(_lstm_on_array(values, len(train), epochs, seed), 0, None)


# --------------------------------------------------------------------------- Hybrid

def hybrid_forecast(
    train: pd.Series, test: pd.Series,
    sarima_test: np.ndarray, sarima_train: np.ndarray,
    epochs: int = 40, seed: int = 0,
):
    """SARIMA prediction + LSTM prediction of what SARIMA got wrong.

    The first SEASON+1 in-sample residuals are dropped: SARIMA has no seasonal history
    there, so those errors are start-up artefacts, not signal.
    """
    burn = SEASON + 1
    resid_train = (train.to_numpy() - sarima_train)[burn:]
    resid_test = test.to_numpy() - sarima_test
    residuals = np.concatenate([resid_train, resid_test])
    resid_pred = _lstm_on_array(residuals, len(resid_train), epochs, seed)
    return np.clip(sarima_test + resid_pred, 0, None)
