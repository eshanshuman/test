"""01 - Generate one synthetic Penaeus vannamei grow-out crop (hourly).

Purpose
    No open prawn-pond dataset has pH, temperature, DO and salinity together,
    so this script simulates one Feb-Sep crop. It is a process model, not
    random numbers: DO comes from an hourly oxygen mass balance, so the
    structure a real pond shows (afternoon DO peak, dawn minimum, decline as
    biomass and feed grow, bloom crashes) is there for the models to find.
    A separate sensor layer then adds what a real probe adds: noise, pH drift
    between calibrations, missing values and radio outages.

Paper link
    Stands in for the pond sensor datasets used in the reviewed IoT
    aquaculture papers (Chen et al. 2022; Hemal et al. 2024; Huang & Khabusi
    2025). Results on it show what the pipeline can do when the physics is
    known; they are NOT field evidence. Real-data results come from 00/NERRS.

Outputs
    data/prawn_pond_synthetic.csv  (timestamp, ph, temperature_c, do_mg_l,
    salinity_ppt, quality_label); block "01_generate_dataset" in
    reports/synthetic/metrics.json.

Knobs: --noise-scale multiplies all sensor noise; --label-noise flips that
fraction of labels to another class (ablation only, default 0).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config as cfg
from water_chem import do_saturation

N_HOURS = 5000
START = "2026-02-15 00:00"
NOISE_SCALE = 1.0          # multiplies every sensor sigma below
LABEL_NOISE = 0.0          # fraction of labels flipped (ablation)
SIGMA = {"ph": 0.02, "temperature_c": 0.08, "do_mg_l": 0.06, "salinity_ppt": 0.15}
MISSING_FRAC = 0.02        # each value independently lost
N_OUTAGES = 5              # radio outages, every channel lost for 3-14 h
CALIBRATION_DAYS = 30      # pH probe is re-calibrated monthly
MONSOON_START_DAY = 106    # 1 June, counted from 15 February


def ar1(rng, n, phi, sigma):
    """AR(1) noise: today's weather resembles yesterday's."""
    out = np.zeros(n)
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def simulate_true_state(rng) -> pd.DataFrame:
    """Noise-free pond state, hour by hour."""
    n = N_HOURS
    hour = np.arange(n) % 24
    day = np.arange(n) / 24.0
    crop = np.arange(n) / (n - 1)                       # 0 at stocking, 1 at harvest

    # --- weather: rain events (more in the monsoon) and heat waves ---------
    rain = np.zeros(n)
    i = 0
    while i < n:
        p = 0.012 if day[i] >= MONSOON_START_DAY else 0.002
        if rng.random() < p:
            length = int(rng.integers(2, 9))
            rain[i:i + length] = rng.uniform(0.5, 2.0)  # intensity
            i += length + 6
        else:
            i += 1
    heat = np.zeros(n)
    for start_day in rng.uniform(45, 120, 3):
        length = int(rng.integers(72, 145))
        s = int(start_day * 24)
        heat[s:s + length] += rng.uniform(2.0, 3.0) * np.sin(np.linspace(0, np.pi, length))

    # --- temperature: seasonal + diurnal (peak 15:00) + AR(1) + events -----
    temp = (29.3 + 1.8 * np.cos(2 * np.pi * (day - 90) / 365)
            + 1.6 * np.cos(2 * np.pi * (hour - 15) / 24)
            + ar1(rng, n, 0.97, 0.6) + heat - 0.8 * np.clip(rain, 0, 1))

    # --- light: daylight curve x daily cloud cover (dark when raining) -----
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    cloud = np.repeat(rng.uniform(0.55, 1.0, n // 24 + 1), 24)[:n]
    light = daylight * np.where(rain > 0, 0.35, cloud)

    # --- hourly loop: salinity, algal bloom, DO mass balance, pH -----------
    sal, bloom, do, ph = (np.zeros(n) for _ in range(4))
    crash = np.zeros(n)                                  # decay demand, mg/L/h
    s, b, o, p_lag, rain_acid, crashing, demand = 17.0, 0.3, 6.5, 8.0, 0.0, 0, 0.0
    ph_wander = ar1(rng, n, 0.98, 0.06)                  # slow alkalinity changes
    for t in range(n):
        # salinity: evaporation concentrates, exchange pulls to source, rain dilutes
        source = 20.0 if day[t] < MONSOON_START_DAY else 11.0
        s += 0.006 * (1 + 0.3 * heat[t]) + 0.003 * (source - s) - 0.06 * rain[t] * s / 15
        # bloom: logistic growth to a carrying capacity that rises with feed input
        cap = 0.7 + 0.8 * crop[t]
        b += 0.004 * b * (1 - b / cap) * (0.5 + cloud[t]) + rng.normal(0, 0.003)
        if crashing == 0 and rng.random() < 0.01 * max(0.0, b - 0.85) ** 2:
            crashing = 36                                # die-off lasts 36 h
        if crashing:
            b *= 0.967
            demand += 0.016                              # dead algae being decomposed
            crashing -= 1
        demand *= 0.98
        b = max(b, 0.05)
        # dissolved oxygen (mg/L per hour), four sub-steps for stability
        sat = float(do_saturation(temp[t], s))
        night = hour[t] >= 20 or hour[t] < 6
        k_air = 0.04 + ((0.10 + 0.08 * crop[t]) if night else 0.0)   # paddlewheels at night
        photo = 2.0 * light[t] * b
        resp = (0.20 + 0.44 * crop[t] + 0.12 * b) * 1.07 ** (temp[t] - 30)
        for _ in range(4):
            degas = 0.10 * max(0.0, o - sat)             # supersaturated water bubbles off
            o += 0.25 * (k_air * (sat - o) - degas + photo - resp - demand)
            o = max(o, 0.1)
        # pH: algae remove CO2 in daylight (pH up), respiration adds it; rain is acidic
        rain_acid = 0.9 * rain_acid + 0.07 * rain[t]
        target = 8.0 + 0.10 * (o - 5.5) + 0.15 * (b - 0.8) - rain_acid + ph_wander[t]
        p_lag += (target - p_lag) / 3.0                  # responds over ~3 h
        sal[t], bloom[t], do[t], ph[t], crash[t] = s, b, o, p_lag, demand

    ts = pd.date_range(START, periods=n, freq="h")
    return pd.DataFrame({cfg.TIME_COL: ts, "ph": ph, "temperature_c": temp,
                         "do_mg_l": do, "salinity_ppt": sal,
                         "bloom_index": bloom, "crash_demand": crash})


def add_sensor_layer(true: pd.DataFrame, rng, noise_scale: float) -> pd.DataFrame:
    """What the probes report: noise, pH drift, random gaps and outages."""
    obs = true[[cfg.TIME_COL] + cfg.FEATURES].copy()
    n = len(obs)
    for col in cfg.FEATURES:
        obs[col] += rng.normal(0, SIGMA[col] * noise_scale, n)
    # pH drift grows linearly after each monthly calibration, then resets to zero
    hours_since_cal = np.arange(n) % (CALIBRATION_DAYS * 24)
    rates = rng.normal(0, 0.08, n // (CALIBRATION_DAYS * 24) + 1)   # pH units per month
    drift = rates[np.arange(n) // (CALIBRATION_DAYS * 24)] * hours_since_cal / (CALIBRATION_DAYS * 24)
    obs["ph"] += drift * noise_scale
    obs["do_mg_l"] = obs["do_mg_l"].clip(lower=0)     # a probe never reads below zero
    for col in cfg.FEATURES:
        obs.loc[rng.random(n) < MISSING_FRAC, col] = np.nan
    for start in rng.integers(0, n - 20, N_OUTAGES):
        obs.loc[start:start + int(rng.integers(3, 15)) - 1, cfg.FEATURES] = np.nan
    return obs.round({"ph": 3, "temperature_c": 2, "do_mg_l": 3, "salinity_ppt": 2})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--noise-scale", type=float, default=NOISE_SCALE)
    ap.add_argument("--label-noise", type=float, default=LABEL_NOISE)
    args = ap.parse_args()
    rng = np.random.default_rng(cfg.SEED)

    cfg.section("01 Simulating the pond (noise-free state)")
    true = simulate_true_state(rng)
    labels = cfg.label_frame(true)                       # labels from the TRUE state
    obs = add_sensor_layer(true, rng, args.noise_scale)
    if args.label_noise > 0:
        flip = rng.random(len(labels)) < args.label_noise
        labels[flip] = [rng.choice([c for c in cfg.CLASSES if c != lab]) for lab in labels[flip]]
        print(f"  LABEL_NOISE: flipped {int(flip.sum())} labels")
    obs[cfg.LABEL_COL] = labels
    print(f"  {len(obs)} hourly rows, {obs[cfg.TIME_COL].iloc[0]} to {obs[cfg.TIME_COL].iloc[-1]}")
    print(true[cfg.FEATURES].describe().loc[["min", "mean", "max"]].round(2).to_string())

    cfg.section("Class balance")
    counts = labels.value_counts().reindex(cfg.CLASSES, fill_value=0)
    for name, c in counts.items():
        print(f"  {name:<9}{c:>6}  {100 * c / len(labels):5.1f} %")
    worst = pd.DataFrame({c: cfg.severity(true[c], c) for c in cfg.FEATURES}).eq(2).sum()
    print("  hours each parameter is Critical on its own:")
    print("   " + ", ".join(f"{cfg.PRETTY[c]} {int(v)}" for c, v in worst.items()))

    cfg.section("Daily DO cycle (noise-free state)")
    by_hour = true.groupby(true[cfg.TIME_COL].dt.hour)["do_mg_l"].mean()
    days = true.groupby(true[cfg.TIME_COL].dt.date)["do_mg_l"]
    h_min = true.loc[days.idxmin(), cfg.TIME_COL].dt.hour.mode()[0]
    h_max = true.loc[days.idxmax(), cfg.TIME_COL].dt.hour.mode()[0]
    print(f"  most common hour of the daily DO minimum : {h_min:02d}:00")
    print(f"  most common hour of the daily DO maximum : {h_max:02d}:00")
    print(f"  mean DO by hour: lowest at {by_hour.idxmin():02d}:00 ({by_hour.min():.2f} mg/L), "
          f"highest at {by_hour.idxmax():02d}:00 ({by_hour.max():.2f} mg/L)")
    thirds = true["do_mg_l"].groupby(np.arange(len(true)) * 3 // len(true)).mean()
    print("  mean DO, first / middle / last third of the crop: "
          + " / ".join(f"{v:.2f}" for v in thirds) + " mg/L")
    n_crash = int((true["crash_demand"].diff() > 0).astype(int).diff().eq(1).sum())
    print(f"  bloom crashes simulated: {n_crash}")

    cfg.section("Sensor layer")
    miss = obs[cfg.FEATURES].isna().mean() * 100
    print("  missing values (%): " + ", ".join(f"{cfg.PRETTY[c]} {v:.1f}" for c, v in miss.items()))
    noisy_labels = cfg.label_frame(obs)
    seen = noisy_labels.notna()
    agree = float((noisy_labels[seen] == labels[seen]).mean())
    print(f"  rows with all four readings: {int(seen.sum())}")
    print(f"  rule applied to the NOISY readings matches the true label on {100 * agree:.2f} % of them")

    obs.to_csv(cfg.SYNTHETIC_CSV, index=False)
    print(f"\n  wrote {cfg.SYNTHETIC_CSV.relative_to(cfg.ROOT).as_posix()}")
    block = {
        "rows": len(obs), "noise_scale": args.noise_scale, "label_noise": args.label_noise,
        "class_counts": counts.to_dict(), "do_min_hour": int(h_min), "do_max_hour": int(h_max),
        "mean_do_by_crop_third": [round(float(v), 3) for v in thirds],
        "bloom_crashes": n_crash, "complete_rows": int(seen.sum()),
        "rule_on_noisy_readings_accuracy": round(agree, 4)}
    path = cfg.ROOT / "reports" / "synthetic" / "metrics.json"   # always synthetic here
    path.parent.mkdir(parents=True, exist_ok=True)
    allm = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    allm["01_generate_dataset"] = block
    path.write_text(json.dumps(allm, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
