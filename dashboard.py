"""dashboard.py - Streamlit dashboard for the Aquabot recommendation engine.

Purpose
    The farmer-facing view of the pipeline. Enter (or pick a preset of) the
    four readings in the sidebar and the page shows: the pond status as a
    colour card, the dissolved-oxygen reading and the LSTM forecast with its
    honest test score, four gauges against the P. vannamei bands, the last
    24 h of the dataset, the prioritised action list from recommend.py and
    the recorded metrics.

    Honesty in the UI: the forecast metric carries the model's held-out test
    R² as its help text, and a warning caption appears whenever that R² is
    below 0.5 or the model did not beat persistence on the test set. The
    forecast is made from the latest gap-free week in the dataset with its
    last hour replaced by the sidebar readings.

Run
    python -m streamlit run dashboard.py
    (set AQUABOT_DATASET to synthetic or real first; see README)

Test
    python tools/test_dashboard.py   (streamlit.testing.v1.AppTest)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import recommend as rec

SLIDERS = {   # key: (label, min, max, step)
    "ph": ("pH", 5.0, 10.5, 0.1),
    "temperature_c": ("Temperature (°C)", 20.0, 40.0, 0.1),
    "do_mg_l": ("Dissolved oxygen (mg/L)", 0.0, 15.0, 0.1),
    "salinity_ppt": ("Salinity (ppt)", 0.0, 45.0, 0.5),
}
URGENCY_ICON = {"immediate": "🔴", "today": "🟠", "this week": "🟡", "routine": "🟢"}


@st.cache_data
def load_history() -> pd.DataFrame:
    return cfg.load_dataset() if cfg.DATA_CSV.exists() else pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    return json.loads(cfg.METRICS_JSON.read_text(encoding="utf-8")) if cfg.METRICS_JSON.exists() else {}


@st.cache_resource
def load_forecaster():
    """LSTM + its scaler, or None if 05 has not been run (or TensorFlow is missing)."""
    model_path, meta_path = cfg.MODEL_DIR / "lstm_do_forecast.keras", cfg.MODEL_DIR / "lstm_do_forecast_meta.json"
    if not (model_path.exists() and meta_path.exists()):
        return None
    try:
        import ts_lib as ts                                # sets the TensorFlow log level
        import tensorflow as tf
        return tf.keras.models.load_model(model_path), json.loads(meta_path.read_text(encoding="utf-8")), ts
    except Exception:                                      # dashboard must still open without TF
        return None


def forecast_do(reading: dict) -> tuple[float, dict] | None:
    """DO forecast: latest gap-free window, last hour replaced by the sidebar reading."""
    loaded, hist = load_forecaster(), load_history()
    if loaded is None or hist.empty:
        return None
    model, meta, ts = loaded
    series = ts.prepare_series(hist)
    ends = ts.valid_ends(series, meta["window_h"], 0)      # windows with no long gap
    if len(ends) == 0:
        return None
    t = int(ends[-1])
    window = series["values"][t - meta["window_h"] + 1:t + 1].copy()
    window[-1, :4] = [reading["do_mg_l"], reading["temperature_c"], reading["ph"], reading["salinity_ppt"]]
    scaled = (window - np.array(meta["mean"])) / np.array(meta["sd"])
    change = float(model.predict(scaled[None].astype(np.float32), verbose=0)[0, 0]) * meta["delta_sd"]
    return max(reading["do_mg_l"] + change, 0.0), meta


def gauge(key: str, value: float) -> go.Figure:
    label, lo, hi, _ = SLIDERS[key]
    (g_lo, g_hi), (m_lo, m_hi) = cfg.BANDS[key]["good"], cfg.BANDS[key]["moderate"]
    g_hi, m_hi = min(g_hi, hi), min(m_hi, hi)
    c = cfg.STATUS_COLOURS
    steps = [{"range": [lo, m_lo], "color": c["Critical"]}, {"range": [m_lo, g_lo], "color": c["Moderate"]},
             {"range": [g_lo, g_hi], "color": c["Good"]}, {"range": [g_hi, m_hi], "color": c["Moderate"]},
             {"range": [m_hi, hi], "color": c["Critical"]}]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": label, "font": {"size": 14}},
        gauge={"axis": {"range": [lo, hi]}, "bar": {"color": "#222222", "thickness": 0.25},
               "steps": [s for s in steps if s["range"][1] > s["range"][0]]}))
    fig.update_layout(height=210, margin={"l": 20, "r": 20, "t": 45, "b": 10})
    return fig


def history_chart(hist: pd.DataFrame, key: str) -> go.Figure:
    last = hist.tail(24)
    fig = go.Figure(go.Scatter(x=last[cfg.TIME_COL], y=last[key], mode="lines+markers",
                               line={"color": cfg.SERIES_COLOURS[0], "width": 2}, name=cfg.PRETTY[key],
                               connectgaps=False))
    for band, status in (("good", "Moderate"), ("moderate", "Critical")):
        for edge in cfg.BANDS[key][band]:
            if np.isfinite(edge):
                fig.add_hline(y=edge, line_dash="dash", line_color=cfg.STATUS_COLOURS[status],
                              annotation_text=f"{status} beyond {edge:g}", annotation_font_size=10)
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      yaxis_title=cfg.PRETTY[key], showlegend=False)
    return fig


def main() -> None:
    st.set_page_config(page_title="Aquabot pond monitor", page_icon="🦐", layout="wide")
    st.title("Aquabot: pond water quality and recommendations")
    st.caption(f"Penaeus vannamei grow-out · models trained on the **{cfg.DATASET}** dataset "
               f"({cfg.DATASET_NOTE})")

    # ---- sidebar: presets and sliders ------------------------------------------
    st.sidebar.header("Current readings")
    preset = st.sidebar.selectbox("Preset", list(rec.PRESETS), key="preset")
    if st.session_state.get("applied_preset") != preset:          # preset changed: move the sliders
        for key, value in zip(cfg.FEATURES, rec.PRESETS[preset]):
            st.session_state[key] = float(value)
        st.session_state["applied_preset"] = preset
    reading = {key: st.sidebar.slider(label, lo, hi, step=step, key=key)
               for key, (label, lo, hi, step) in SLIDERS.items()}
    st.sidebar.caption("Bands: pH 7.5-8.5, 28-32 °C, DO ≥ 5 mg/L, salinity 10-25 ppt are Good.")

    result = rec.recommend(reading["ph"], reading["temperature_c"], reading["do_mg_l"], reading["salinity_ppt"])
    hist = load_history()

    # ---- status card ---------------------------------------------------------------
    colour = cfg.STATUS_COLOURS[result["status"]]
    detail = f"Rules: {result['rule_status']}"
    if result["model_status"] is not None:
        detail += (f" · {result['model_name']}: {result['model_status']} "
                   f"({result['model_confidence']:.0%} confident)")
    st.markdown(
        f"<div style='background:{colour};color:white;padding:18px 24px;border-radius:10px'>"
        f"<div style='font-size:15px;opacity:0.9'>Pond status</div>"
        f"<div style='font-size:38px;font-weight:700;line-height:1.2'>{result['status']}</div>"
        f"<div style='font-size:14px'>{detail}</div></div>", unsafe_allow_html=True)
    if result["agreement"] is False:
        st.warning("Rules and model disagree, so the more severe status is shown. Verify the probe readings.")

    # ---- headline metrics ---------------------------------------------------------
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("DO now", f"{reading['do_mg_l']:.1f} mg/L")
    forecast = forecast_do(reading)
    if forecast is None:
        c2.metric("LSTM forecast", "not available", help="Run 05_lstm_do_forecast.py to train the forecaster.")
    else:
        value, meta = forecast
        c2.metric(f"LSTM forecast, +{meta['horizon_h']} h", f"{value:.1f} mg/L",
                  delta=f"{value - reading['do_mg_l']:+.1f} mg/L",
                  help=f"Held-out test R² = {meta['test_r2']:.3f} (RMSE {meta['test_rmse']:.3f} mg/L). "
                       f"Persistence on the same hours: R² = {meta['persistence_r2']:.3f} "
                       f"(RMSE {meta['persistence_rmse']:.3f} mg/L).")
        weak = meta["test_r2"] < 0.5
        no_skill = meta["test_rmse"] >= meta["persistence_rmse"]
        if weak or no_skill:
            why = ([f"its test R² is only {meta['test_r2']:.2f}"] if weak else []) + (
                ["it did not beat simply repeating the last reading"] if no_skill else [])
            c2.caption(f"⚠️ Low-confidence forecast: {' and '.join(why)}. Do not act on it alone.")
    c3.metric("Immediate actions", result["n_immediate"])

    # ---- gauges ----------------------------------------------------------------------
    st.subheader("Readings against the vannamei bands")
    for col, key in zip(st.columns(4), cfg.FEATURES):
        col.plotly_chart(gauge(key, reading[key]), width="stretch", key=f"gauge_{key}")

    # ---- actions and history --------------------------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.subheader("What to do")
        for act in result["actions"]:
            st.markdown(f"{URGENCY_ICON[act['urgency']]} **{act['urgency'].upper()} · {act['parameter']}**  \n"
                        f"{act['action']}  \n*Why: {act['reason']}*")
        st.caption("Doses are typical extension-guidance values; confirm with a local soil and water test.")
    with right:
        st.subheader("Last 24 hours in the dataset")
        if hist.empty:
            st.info("No dataset found. Run 01_generate_dataset.py or 00_prepare_dataset.py.")
        else:
            key = st.radio("Parameter", cfg.FEATURES, index=2, horizontal=True,
                           format_func=lambda k: cfg.PRETTY[k], key="history_param")
            st.plotly_chart(history_chart(hist, key), width="stretch", key="history_chart")

    with st.expander("Model metrics (from reports/<dataset>/metrics.json)"):
        metrics = load_metrics()
        if not metrics:
            st.info("No metrics yet. Run the numbered scripts or run_all.py.")
        else:
            rows = []
            for block, name in (("03_random_forest", "Random Forest"), ("04_lightgbm", "LightGBM")):
                if block in metrics:
                    rows.append({"model": name, **metrics[block]["test"],
                                 "rule reference accuracy": metrics[block]["rule_reference"]["accuracy"]})
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.json(metrics, expanded=False)


main()
