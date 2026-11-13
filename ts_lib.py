"""Shared time-series helpers for 05 (DO forecast), 06 (oxygen state) and 08 (cross-site).

Purpose
    The three sequence scripts must treat gaps, windows, splits and scaling in
    exactly the same way, so that logic lives here once:
      * prepare_series(): hourly grid, short gaps (<= MAX_GAP_H) interpolated,
        longer gaps marked as "bad" hours.
      * valid_ends(): which windows may be used. A sample is skipped if its
        input window or its target touches a bad hour, or if the target DO was
        not actually measured (models are never scored on interpolated values).
      * chrono_split(): chronological split with no overlap between sets.
      * fit_scaler(): mean/sd from TRAINING rows only.
      * build_lstm() / train(): the Keras model and its early-stopped fit.

Outputs
    None; imported by the numbered scripts.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # hide TensorFlow start-up chatter

import numpy as np
import pandas as pd

import config as cfg

CHANNELS = ["do_mg_l", "temperature_c", "ph", "salinity_ppt", "hour_sin", "hour_cos"]
DO = 0                                                  # column index of DO in CHANNELS


def prepare_series(df: pd.DataFrame) -> dict:
    """Regular hourly grid + channels + masks for one dataset."""
    df = df.set_index(cfg.TIME_COL).sort_index()
    grid = pd.date_range(df.index[0], df.index[-1], freq="h")
    raw = df[cfg.FEATURES].reindex(grid)
    filled = cfg.interpolate_short_gaps(raw, cfg.MAX_GAP_H)
    values = pd.DataFrame({
        "do_mg_l": filled["do_mg_l"], "temperature_c": filled["temperature_c"],
        "ph": filled["ph"], "salinity_ppt": filled["salinity_ppt"],
        "hour_sin": np.sin(2 * np.pi * grid.hour / 24),
        "hour_cos": np.cos(2 * np.pi * grid.hour / 24)}, index=grid)[CHANNELS]
    return {"time": grid,
            "values": values.to_numpy(np.float32),
            "bad": filled.isna().any(axis=1).to_numpy(),          # inside a gap > MAX_GAP_H
            "do_measured": raw["do_mg_l"].notna().to_numpy()}     # a real reading, not interpolated


def valid_ends(series: dict, window: int, horizon: int) -> np.ndarray:
    """Indices t (hour of the last input reading) of every usable sample.

    Inputs are hours t-window+1 .. t, the target is hour t+horizon. Everything
    from the first input to the target must be free of long gaps.
    """
    n = len(series["bad"])
    bad_cum = np.concatenate([[0], np.cumsum(series["bad"])])
    t = np.arange(window - 1, n - horizon)
    clean = (bad_cum[t + horizon + 1] - bad_cum[t - window + 1]) == 0
    return t[clean & series["do_measured"][t + horizon]]


def chrono_split(t: np.ndarray, horizon: int, n: int, fracs=(0.70, 0.15)) -> tuple:
    """Chronological train/val/test. A sample belongs to a set only if BOTH its
    last input reading and its target fall inside that set's time span, so no
    target hour of one set is ever a training target or 'last reading' of another."""
    i1, i2 = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    train = t[t + horizon < i1]
    val = t[(t >= i1) & (t + horizon < i2)]
    test = t[t >= i2]
    return train, val, test, i1


def fit_scaler(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and sd from rows before train_end only (training period)."""
    rows = values[:train_end]
    mean, sd = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def windows(scaled: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """Stack the input windows ending at each t: shape (samples, window, channels)."""
    view = np.lib.stride_tricks.sliding_window_view(scaled, window, axis=0)   # (n-w+1, ch, w)
    return np.ascontiguousarray(view[t - window + 1].transpose(0, 2, 1))


def seed_everything(seed: int) -> None:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)                 # python, numpy and TF seeds


def build_lstm(window: int, units: int, n_out: int = 1, activation=None):
    """One LSTM layer -> dropout -> small dense layer -> output."""
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input((window, len(CHANNELS))),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n_out, activation=activation)])


def train(model, X_tr, y_tr, X_va, y_va, epochs: int, loss="mse", patience: int = 8,
          class_weight=None, batch_size: int = 128):
    """Fit with early stopping on the validation set; returns epochs actually run."""
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                            restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                     batch_size=batch_size, callbacks=[stop], verbose=0,
                     class_weight=class_weight, shuffle=True)   # shuffles batches of TRAINING windows only
    return len(hist.history["loss"]), float(min(hist.history["val_loss"]))


def sqrt_class_weights(y: np.ndarray, n_classes: int = 3) -> dict:
    """Square-root inverse-frequency class weights, mean 1 over the classes present."""
    counts = np.bincount(y, minlength=n_classes)
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = np.sqrt(counts.sum() / (present.sum() * counts[present]))
    w[present] /= w[present].mean()
    return {i: float(w[i]) for i in range(n_classes) if present[i]}


def regression_scores(y_true, y_pred) -> dict:
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    ss_res, ss_tot = float((err**2).sum()), float(((y_true - np.mean(y_true))**2).sum())
    return {"rmse": round(float(np.sqrt((err**2).mean())), 4),
            "mae": round(float(np.abs(err).mean()), 4),
            "r2": round(1.0 - ss_res / ss_tot, 4)}
