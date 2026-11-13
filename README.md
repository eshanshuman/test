# Aquabot: pond water-quality monitoring and prawn-farming recommendations

Software pipeline for the UROP project "AI-Enabled Smart Pond Water Quality
Monitoring and Prawn Farming Recommendation Aquabot System" (SRM IST,
Department of Data Science and Business Systems). Species: *Penaeus vannamei*
in Indian grow-out ponds. Software only: no hardware, sensors or firmware.

The pipeline takes hourly pH, temperature, dissolved oxygen (DO) and salinity
and produces: a pond status (Good / Moderate / Critical), a DO forecast, an
oxygen-state warning, and a prioritised list of management actions, with a
Streamlit dashboard on top. Every model is compared with simple baselines and
is only said to "work" where it beats them.

**Every number in this file comes from one clean run of `python run_all.py`**
(22 of 22 steps succeeded, 34.4 minutes on a 4-core CPU without a GPU),
formatted by `tools/summarize_results.py` into `reports/results_summary.md`.
The real-pond results (sections 3 and 4.8) were added afterwards and come from
`python run_all.py --only ponds` (2 of 2 steps succeeded, 10.3 minutes).

## 1. Setup (Windows, PowerShell)

```powershell
py -0p                                  # check that Python 3.11 is installed
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Choosing the dataset. Two datasets are supported and their outputs are kept
apart in `figures/<dataset>/`, `models/<dataset>/` and `reports/<dataset>/`.

```powershell
$env:AQUABOT_DATASET = "synthetic"      # the simulated pond
$env:AQUABOT_DATASET = "real"           # data/pond_dataset.csv
Remove-Item Env:AQUABOT_DATASET         # unset: real if data/pond_dataset.csv exists, else synthetic
```

(In the old Command Prompt: `set AQUABOT_DATASET=synthetic`.)

Run everything, then look at the results:

```powershell
python tools/fetch_nerrs.py                                              # once: download the real data
python 00_prepare_dataset.py data/raw/nerrs_apadb_2013_jun_sep.csv
python 00_prepare_dataset.py data/raw/nerrs_apacp_2013_jun_sep.csv --output data/pond_dataset_site2.csv
python run_all.py                                                        # whole pipeline, all datasets
python tools/summarize_results.py --out reports/results_summary.md
python -m streamlit run dashboard.py
```

For the real fish ponds, download the Kaggle dataset (doi:10.34740/KAGGLE/DSV/3748790),
unzip its 11 CSV files into `data/raw/udanor/` (git-ignored) and `run_all.py`
will prepare and analyse them; `python run_all.py --only ponds` runs just that
part. The prepared hourly files in `data/ponds/` are in the repository, so
`09_real_ponds.py` also runs without the raw files.

The code was developed and run on Linux with Python 3.11; it uses `pathlib`
throughout and no shell scripts, but it has not yet been executed on Windows.

## 2. What each script does

| File | What it does |
|---|---|
| `config.py` | Everything shared: seed 42, paths, dataset switch, the vannamei thresholds and labelling rule, colours, plot style, helpers (`section`, `record_metric`, gap interpolation, the shared train/test split). |
| `water_chem.py` | DO saturation from temperature and salinity (Benson & Krause, as in APHA 4500-O), PSS-78 conductivity-to-salinity, unit conversions. `python water_chem.py` runs its checks. |
| `ts_lib.py` | Shared time-series logic for 05, 06 and 08: gap handling, windows, chronological split, training-only scaler, the LSTM. |
| `00_prepare_dataset.py` | Data adapter: turns any CSV/Excel logger export into the pipeline format (column guessing, unit conversion, QC bounds, hourly grid, labels). |
| `01_generate_dataset.py` | Simulates one Feb-Sep crop hour by hour with a DO mass balance, then adds a sensor layer (noise, pH drift, gaps). |
| `02_eda.py` | Correlations, distributions by class, class balance, 30-day DO trace, mutual information (fig01-05). |
| `03_random_forest.py` | Status classifier with a 5-fold grid search; confusion matrix, ROC, permutation importance; scores the labelling rule itself as the reference (fig06-08). |
| `04_lightgbm.py` | Same split, early stopping, exact tree-SHAP contributions (fig09-10). |
| `05_lstm_do_forecast.py` | LSTM DO forecast from a 168-h window against persistence and same-hour-yesterday; horizon sweep (fig11-13). |
| `06_do_state_classifier.py` | LSTM naming the oxygen state (Anoxia / Hypoxia / Normoxia) H hours ahead, 10 repeated runs (fig14). |
| `07_lr_vs_rf_scaling.py` | Linear Regression vs Random Forest for estimating DO from the other probes, as training size grows (fig15). |
| `08_cross_site.py` | Real data only: model trained at one site, tested at another (fig16). |
| `09_real_ponds.py` | Real catfish ponds: per-pond and leave-one-pond-out DO forecasts and oxygen-state warning against both baselines (fig18). |
| `recommend.py` | Hybrid rules + model recommendation engine; 4 worked scenarios; surrogate decision tree (fig17). |
| `dashboard.py` | Streamlit + Plotly dashboard. |
| `run_all.py` | Runs the whole pipeline and times every step. |
| `tools/fetch_nerrs.py` | Downloads and cleans the real data. |
| `tools/prepare_ponds.py` | Checks, cleans and hourly-averages the real fish-pond files; decides which ponds are usable. |
| `tools/summarize_results.py` | Prints every headline number as Markdown tables. |
| `tools/test_prepare_roundtrip.py` | Round-trip test of the data adapter. |
| `tools/test_dashboard.py` | Headless dashboard test (`streamlit.testing.v1.AppTest`). |

Domain rules (in `config.py`). The pond label is the worst single parameter
(Liebig's law of the minimum).

| Parameter | Good | Moderate | otherwise |
|---|---|---|---|
| pH | 7.5-8.5 | 7.0-9.0 | Critical |
| Temperature (°C) | 28-32 | 26-34 | Critical |
| DO (mg/L) | ≥ 5 | ≥ 3 | Critical |
| Salinity (ppt) | 10-25 | 5-35 | Critical |

Oxygen states: Anoxia < 2, Hypoxia 2-4, Normoxia ≥ 4 mg/L.

## 3. The datasets

### Synthetic pond (`data/prawn_pond_synthetic.csv`)

5,000 hourly rows from 2026-02-15. A process model, not random numbers: DO is
an hourly mass balance (re-aeration toward saturation, night paddlewheels,
photosynthesis, respiration that grows through the crop, bloom crashes); pH
follows photosynthesis; rain dilutes salinity and lowers pH. A sensor layer
then adds noise, pH drift between monthly calibrations, 2 % missing values and
radio outages. **Labels come from the noise-free state**, so the threshold
rule applied to the noisy readings matches the true label on 97.54 % of
complete rows; that gap is pure sensor noise.

Caveats: the coefficients were tuned by hand until the pond looked like a
managed pond (DO lowest at 07:00, highest at 15:00; mean DO by crop third
6.34 / 5.63 / 5.56 mg/L); they are plausible, not calibrated to field data.
The class balance (41.5 % Good, 52.6 % Moderate, 6.0 % Critical) is therefore
a design outcome. Salinity and pH never reach Critical in this crop, and the
strong pH-DO link is built in by construction.

### Real data: estuarine monitoring data (`data/pond_dataset.csv`)

No open prawn-pond dataset has all four parameters, so the real dataset is
NOAA NERRS System-Wide Monitoring Program data from Apalachicola Bay, Florida,
as shipped with the SWMPr R package: Dry Bar (main dataset) and Cat Point
(cross-site test), June-September 2013, 15-minute readings averaged to 2,928
hourly rows per site. A value is kept only if its NERRS QC flag starts with
`<0>`. Provenance is in `data/raw/README.md`.

Caveats, which apply to every real-data result below:

- **It is an estuary, not a pond**: tidal, with no aerators, feeding or
  stocking. It tests the pipeline on real sensor behaviour; it does not
  describe a vannamei pond.
- The labels are the vannamei thresholds applied to estuary water (15.0 %
  Good, 66.2 % Moderate, 18.9 % Critical at Dry Bar). They say what a prawn
  would experience there, nothing about the bay's ecological state.
- Timestamps were shifted from UTC to local standard time (UTC-5) so that
  "hour of day" is meaningful; mean DO is lowest at 06:00 and highest at 17:00.
- Low-oxygen hours are rare (Anoxia in 10 hours, Hypoxia in 162 of 2,928).

### Real catfish ponds, freshwater (`data/ponds/pond_XX.csv`)

The only data here from real STOCKED ponds: "Sensor Based Aquaponics Fish Pond
Datasets" (Udanor et al. 2022, Data in Brief 43:108400; Kaggle
doi:10.34740/KAGGLE/DSV/3748790; CC BY 4.0). African catfish (*Clarias
gariepinus*) in indoor tarpaulin ponds at the University of Nigeria Nsukka,
June-December 2021, a reading every few seconds to minutes. All 11 raw files
match the SHA-256 sums published with the leakage-aware re-analysis.
`tools/prepare_ponds.py` applies range limits (DO 0.1-20 mg/L, pH 4-11,
temperature 15-40 °C), removes runs of 30 or more identical DO or pH readings
(stuck sensor) and takes hourly medians of at least 3 valid readings.

| Pond | Raw rows | DO readings kept | Hours with DO | DO p5 / median / p95 (mg/L) | Hours < 4 mg/L | Hours < 2 mg/L |
|---|---|---|---|---|---|---|
| 1 | 83,126 | 78.2 % | 1,176 | 0.83 / 4.21 / 12.11 | 554 | 301 |
| 3 | 169,185 | 88.9 % | 1,426 | 3.09 / 5.08 / 14.33 | 141 | 15 |
| 4 | 89,797 | 90.1 % | 1,230 | 1.12 / 4.57 / 10.34 | 520 | 219 |
| 6 | 91,015 | 86.6 % | 1,367 | 0.48 / 5.25 / 15.62 | 559 | 371 |
| 7 | 279,612 | 68.5 % | 1,944 | 2.64 / 7.18 / 18.20 | 347 | 61 |
| 9 | 151,785 | 74.5 % | 1,335 | 2.45 / 7.50 / 15.66 | 283 | 38 |

Caveats, which apply to every result in section 4.8:

- **Freshwater catfish, not vannamei.** It shows how the DO methods behave in
  a stocked pond; it says nothing about prawn thresholds or brackish water.
- **No salinity sensor** (salinity is written as 0.0), so there are no status
  labels and no status classification on this dataset: DO tasks only.
- **The DO sensor is noisy.** The 95th percentile of hourly DO is 10-18 mg/L,
  implausibly high for a stocked indoor pond, and only about half to
  two-thirds of each pond's calendar hours have valid DO.
- **Five of the 11 files are excluded**, with the evidence printed by the
  script: pond 2 repeats 97.7 % of pond 1's (timestamp, DO) pairs, so it is
  not an independent pond; pond 8 has 644 hours of valid DO; ponds 10, 11 and
  12 have 6, 13 and 1.
- **The ponds are not stationary.** Conditions in a pond's late (test) period
  can differ sharply from its early (training) period; in pond 6 mean DO is
  7.41 mg/L over the training samples and 2.77 over the test samples.

## 4. Results

### 4.1 Status classification (03, 04)

| Dataset | Method | Accuracy | Macro F1 |
|---|---|---|---|
| Synthetic (919 test rows) | Random Forest | 0.9771 | 0.9673 |
| | LightGBM | 0.9761 | 0.9607 |
| | Labelling rule on the same readings | 0.9771 | 0.9758 |
| | Always the majority class | 0.5288 | 0.2306 |
| Real, estuarine monitoring data (579 test rows) | Random Forest | 1.0000 | 1.0000 |
| | LightGBM | 0.9931 | 0.9917 |
| | Labelling rule on the same readings | 1.0000 | 1.0000 |
| | Always the majority class | 0.6632 | 0.2658 |

Neither model beats the labelling rule on either dataset, which is the
expected and honest outcome: a classifier fed four instantaneous readings
cannot do better than the rule that defines the label. On real data the rule
is 1.0 by construction (the labels were made by it from these readings), so
the perfect Random Forest score is threshold re-learning, not a discovery. On
synthetic data the remaining 2.3 % error is sensor noise. The classifiers'
role in this project is the second opinion in the recommendation engine.

### 4.2 DO forecast at +24 h (05), main model, 48 units

| Dataset | Method | RMSE (mg/L) | MAE (mg/L) | R² |
|---|---|---|---|---|
| Synthetic (715 test hours) | LSTM | 1.391 | 1.028 | 0.641 |
| | Persistence | 1.385 | 0.992 | 0.645 |
| Real, estuarine monitoring data (406 test hours) | LSTM | 1.032 | 0.805 | -0.064 |
| | Persistence | 1.103 | 0.822 | -0.215 |

At +24 h "same hour yesterday" is the last reading, so it equals persistence.

**Day-ahead DO forecasting does not work on either dataset.** On synthetic
data the LSTM does not beat persistence (skill -0.009). On real data it beats
persistence (skill +0.124) but its R² is negative, i.e. worse than always
predicting the test-period mean. Predicting the level instead of the change
(`--target level`) gives R² 0.623 on synthetic and 0.157 on real data.

### 4.3 DO forecast by horizon (05), 32 units, 3 seeds, test RMSE in mg/L

| Horizon | Synthetic: LSTM | Persistence | Yesterday | Real: LSTM | Persistence | Yesterday |
|---|---|---|---|---|---|---|
| +1 h | 0.183 ± 0.011 | 0.511 | 1.598 | 0.546 ± 0.004 | 0.598 | 1.152 |
| +3 h | 0.503 ± 0.045 | 1.416 | 1.600 | 0.880 ± 0.026 | 1.085 | 1.154 |
| +6 h | 1.000 ± 0.054 | 2.403 | 1.602 | 1.005 ± 0.044 | 1.389 | 1.156 |
| +12 h | 1.421 ± 0.030 | 3.029 | 1.556 | 0.993 ± 0.007 | 1.562 | 1.104 |
| +24 h | 1.356 ± 0.015 | 1.385 | 1.385 | 1.023 ± 0.024 | 1.103 | 1.103 |

LSTM R² by horizon: synthetic 0.994, 0.953, 0.815, 0.628, 0.659; real 0.730,
0.290, 0.071, 0.039, -0.045. The LSTM is clearly useful up to +6 h on the
synthetic pond and only at +1 h on the estuary; beyond that its lead over the
better baseline is small and its absolute accuracy is poor.

### 4.4 Oxygen-state classifier (06), macro F1, 10 runs

| Dataset | Horizon, split | LSTM | Persistence | Same hour yesterday | LSTM wins vs persistence / yesterday | Beats both |
|---|---|---|---|---|---|---|
| Synthetic | +6 h, random days | 0.691 ± 0.090 | 0.543 ± 0.105 | 0.618 ± 0.079 | 10 / 6 of 10 | yes |
| | +1 h, random days | 0.704 ± 0.089 | 0.840 ± 0.062 | 0.618 ± 0.079 | 1 / 8 of 10 | no |
| | +6 h, chronological | 0.569 ± 0.021 | 0.628 | 0.602 | 0 / 1 of 10 | no |
| Real, estuarine monitoring data | +6 h, random days | 0.541 ± 0.110 | 0.464 ± 0.084 | 0.545 ± 0.112 | 10 / 5 of 10 | no |
| | +1 h, random days | 0.614 ± 0.130 | 0.670 ± 0.111 | 0.545 ± 0.111 | 3 / 7 of 10 | no |
| | +6 h, chronological | 0.494 ± 0.010 | 0.539 | 0.488 | 0 / 9 of 10 | no |

A mostly negative result. The classifier beats both baselines in one of six
configurations, and that advantage disappears under the stricter
chronological split. One hour ahead, "the state stays the same" is hard to
beat. The brief's quantum-LSTM figure of 92.4 % cannot be reproduced or
compared: that paper was not supplied.

### 4.5 Linear Regression vs Random Forest (07), test RMSE in mg/L

| Training rows | Synthetic: linear | Synthetic: forest | Real: linear | Real: forest |
|---|---|---|---|---|
| 25 | 0.953 ± 0.037 | 1.185 ± 0.033 | 0.970 ± 0.033 | 0.962 ± 0.031 |
| 50 | 0.875 ± 0.009 | 1.005 ± 0.029 | 0.905 ± 0.015 | 0.901 ± 0.018 |
| 100 | 0.856 ± 0.009 | 0.823 ± 0.018 | 0.874 ± 0.008 | 0.816 ± 0.011 |
| 200 | 0.847 ± 0.006 | 0.724 ± 0.014 | 0.858 ± 0.005 | 0.764 ± 0.012 |
| 400 | 0.835 ± 0.003 | 0.651 ± 0.007 | 0.851 ± 0.003 | 0.705 ± 0.008 |
| 800 | 0.831 ± 0.002 | 0.594 ± 0.006 | 0.849 ± 0.001 | 0.654 ± 0.005 |
| 1600 | 0.828 ± 0.001 | 0.540 ± 0.003 | 0.848 ± 0.001 | 0.604 ± 0.003 |
| Full pool (3200 / 2312) | 0.827 | 0.504 | 0.847 | 0.578 |

Mean ± 95 % CI over 25 random draws. The forest is clearly ahead (intervals
not overlapping) from 100 training rows on both datasets; below that, the
straight line is as good or better. At the full pool the forest reaches R²
0.926 (synthetic) and 0.729 (real) against 0.802 and 0.417 for the line.
Always predicting the mean gives RMSE 1.861 and 1.110.

### 4.6 Cross-site test at Cat Point (08), estuarine monitoring data

| Horizon | Trained at Dry Bar | Trained at Cat Point | Persistence | Transfer penalty | R² transfer / local |
|---|---|---|---|---|---|
| +1 h | 0.509 ± 0.005 | 0.508 ± 0.005 | 0.528 | +0.3 % | 0.588 / 0.591 |
| +3 h | 0.768 ± 0.016 | 0.724 ± 0.033 | 0.791 | +6.0 % | 0.068 / 0.169 |
| +6 h | 0.882 ± 0.009 | 0.806 ± 0.016 | 0.942 | +9.4 % | -0.223 / -0.023 |

A model that never saw Cat Point beats persistence there at all three
horizons, with a penalty that grows with the horizon. Only +1 h is a usable
forecast in absolute terms. Both sites are in the same bay over the same
months, so this is a mild transfer test.

### 4.7 Recommendation engine

| Scenario | pH, °C, DO, ppt | Rules | Model, synthetic | Model, real | Final status |
|---|---|---|---|---|---|
| Healthy afternoon | 8.1, 30.0, 7.5, 18 | Good | Good (100 %) | Good (100 %) | Good |
| Dawn hypoxia | 7.6, 28.5, 2.4, 18 | Critical | Critical (99 %) | Critical (100 %) | Critical |
| Acid rain event | 6.8, 27.5, 5.5, 8 | Critical | Moderate (99 %) | Critical (100 %) | Critical |
| Heatwave + bloom | 9.2, 34.6, 11.0, 24 | Critical | Critical (96 %) | Good (100 %) | Critical |

The two disagreements are the reason the engine is a hybrid: each model is
confidently wrong about conditions absent from its training data (no Critical
pH in the synthetic crop; no heatwave in the estuary). The engine then takes
the more severe status and adds "Verify the probe readings". The depth-3
surrogate tree agrees with the model on 97.0 % (synthetic) and 80.7 % (real)
of test readings.

### 4.8 Real catfish ponds, freshwater (09)

Six ponds, 24-h input window, 32 LSTM units, 3 seeds. "Local" is trained on the
pond's own early data; "transferred" is trained on the other five ponds and
never sees the test pond. Means over the six ponds:

| Task | Local LSTM | LSTM from other 5 ponds | Persistence | Same hour yesterday | Local beats better baseline | Transferred beats better baseline | Transferred beats local |
|---|---|---|---|---|---|---|---|
| DO forecast RMSE (mg/L), +1 h | 0.870 | 0.852 | 0.660 | 1.278 | 0 of 6 | 0 of 6 | 4 of 6 |
| DO forecast RMSE (mg/L), +3 h | 1.176 | 1.355 | 0.861 | 1.154 | 0 of 6 | 0 of 6 | 2 of 6 |
| DO forecast RMSE (mg/L), +6 h | 1.440 | 1.768 | 0.982 | 1.229 | 0 of 6 | 0 of 6 | 2 of 6 |
| Oxygen state macro F1, +6 h | 0.294 | 0.372 | 0.673 | 0.644 | 0 of 6 | 0 of 6 | 4 of 6 |

Per pond at +6 h (LSTMs: mean ± sd over seeds):

| Pond | Test hours | RMSE: local | RMSE: transferred | RMSE: persistence | RMSE: yesterday | F1: local | F1: transferred | F1: persistence | F1: yesterday |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 115 | 0.996 ± 0.034 | 2.045 ± 0.321 | 0.894 | 0.875 | 0.323 ± 0.037 | 0.156 ± 0.055 | 0.497 | 0.482 |
| 3 | 158 | 0.495 ± 0.124 | 1.280 ± 0.235 | 0.021 | 0.033 | 0.960 ± 0.057 | 0.684 ± 0.096 | 1.000 | 1.000 |
| 4 | 114 | 1.627 ± 0.179 | 2.439 ± 0.446 | 1.386 | 1.349 | 0.173 ± 0.069 | 0.482 ± 0.057 | 0.522 | 0.538 |
| 6 | 98 | 2.247 ± 0.944 | 2.003 ± 0.167 | 1.300 | 1.385 | 0.103 ± 0.000 | 0.271 ± 0.027 | 0.633 | 0.642 |
| 7 | 168 | 1.863 ± 0.341 | 1.246 ± 0.119 | 1.043 | 1.871 | 0.176 ± 0.133 | 0.295 ± 0.050 | 0.891 | 0.722 |
| 9 | 98 | 1.411 ± 0.159 | 1.593 ± 0.054 | 1.249 | 1.861 | 0.030 ± 0.028 | 0.343 ± 0.051 | 0.495 | 0.481 |

The +1 h and +3 h rows per pond are in `reports/results_summary.md`.

**A clear negative result.** No LSTM, local or transferred, beats the better
of the two baselines in any of the six ponds, at any horizon, on either task.
Persistence is the best method at +1 h in every pond. The models fail because
the ponds change between the training and test periods and a model trained on
the early period cannot follow; persistence adapts because it only looks at
the last reading. Pond 3's test period is almost flat (persistence RMSE 0.021
mg/L) and contains only Normoxia, so its scores say little. Test sets are
small (98 to 181 hours per pond). Nothing was tuned after seeing these
results. This agrees with the finding attributed to the leakage-aware
re-analysis of the same dataset, that simple baselines matched deep models.

**The doses in `recommend.py` are unverified.** They are typical figures
recalled from Indian extension guidance (CIBA / MPEDA) and have not been
checked against the source documents; they are collected in the `DOSES`
dictionary, and every action tells the farmer to confirm with a local soil
and water test.

## 5. What each dataset does and does not show

| | Synthetic pond | Real (estuarine monitoring data) | Real catfish ponds, freshwater |
|---|---|---|---|
| Shows | That the pipeline recovers known pond dynamics; how much error is pure sensor noise; behaviour under a pond-like daily DO cycle. | That the pipeline survives real sensor noise, drift, QC flags and gaps; honest forecast skill on measured DO; transfer between two sites. | How the DO methods behave in real stocked ponds with a cheap, noisy sensor: persistence beats every LSTM in all six ponds, so the short-horizon LSTM gains seen on the other two datasets do not carry over. |
| Does not show | Anything about a real pond: the dynamics and class balance were designed. Good results here are a check of the method, not field evidence. | Anything about a prawn pond: no aeration, feeding or stocking, and tides dominate. Status accuracy is 1.0 by construction. | Anything about vannamei or brackish water: freshwater catfish, no salinity, no status labels. Small test sets and one architecture, so it does not show that no model could do better. |

No dataset here validates the system for a working vannamei farm. That needs
logged vannamei pond data, which the adapter in section 7 is built to accept.
The one test on stocked ponds argues for keeping the simple baselines and the
rule-based warnings in front of any learned forecast.

## 6. Methodology choices a reviewer will ask about

- **Baselines everywhere.** Classifiers are compared with the labelling rule
  and the majority class; forecasts with persistence and same-hour-yesterday,
  on exactly the same test hours.
- **Macro F1, not accuracy,** as the headline classification metric, because
  the classes are unbalanced and accuracy rewards ignoring the rare one.
- **No resampling** (no SMOTE) at any point; class imbalance is handled with
  class weights inside the loss where it is handled at all.
- **No shuffling of time for forecasting.** 05 and 08 use a chronological
  70/15/15 split; a sample belongs to a set only if both its last input and
  its target fall inside that set's time span.
- **Scalers are fitted on training rows only**, including in the cross-site
  test, where Cat Point is scaled with Dry Bar's scaler.
- **Gaps.** Gaps of up to 6 h are interpolated for model inputs; any window
  touching a longer gap is skipped; a target must be a real measurement.
- **Predicting the change in DO**, then adding it to the last reading, so that
  "no change" is the network's starting point.
- **Leakage check.** Synthetic +24 h R² is 0.641, above the 0.5 trigger; the
  cause is the daily cycle, since persistence alone scores 0.645 on the same
  hours, and the split-order check passes.
- **Random-day vs chronological split in 06.** Both are reported because the
  random-day split is optimistic (a test day's neighbours are in training).
- **07 uses a random test set** because it estimates DO at the same moment
  rather than forecasting it; neighbouring hours can fall on both sides.
- **Real ponds (09) deviate in two ways**, both forced by long sensor gaps: a
  24-h input window instead of 168 h, and the 70/15/15 split taken over each
  pond's ordered usable samples rather than its calendar span (valid hours
  cluster early, so a calendar split leaves some ponds without test hours).
  The periods stay disjoint in time and the scaler uses training hours only.
- **Repeats.** 06 is repeated 10 times, sweeps use 3 seeds, 07 uses 25 draws;
  results are given as mean with spread.
- **Not tuned:** the LSTMs use one architecture each and LightGBM uses fixed
  settings with early stopping; only the Random Forest is grid-searched.

## 7. Plugging in a new dataset

Any CSV or Excel export with a timestamp and the four parameters can be used:

```powershell
python 00_prepare_dataset.py my_pond_log.xlsx --dayfirst --temp-unit F --do-unit pct --salinity-unit ec_us
$env:AQUABOT_DATASET = "real"
python run_all.py --only real
```

- Columns are guessed from their names and the mapping is printed; override
  with `--time --ph --temp --do --salinity`.
- Units: `--temp-unit C|F`; `--do-unit mgl|pct`; `--salinity-unit
  ppt|ec_us|ec_ms|ec25_us|ec25_ms` (conductivity is converted with PSS-78);
  `--assume-salinity 15` if there is no salinity column.
- Expert grades instead of thresholds: `--label-col Grade --label-map
  "A:Good,B:Moderate,C:Critical"` (this path has not been exercised on a
  graded file yet).
- The output replaces `data/pond_dataset.csv`; use `--output` to write
  elsewhere. `python tools/test_prepare_roundtrip.py` checks the adapter on a
  deliberately messy copy of the synthetic data.

## 8. References

The review papers named in the project brief are listed for context. Their
PDFs were not available while this code was written, so no result here is
claimed to reproduce or to be compared with theirs; likewise for the other
works the brief mentions (Rohini et al., the 2026 LightGBM paper, the Vietnam
LSTM, the quantum-LSTM, de França et al.).

- Chen et al. (2022). *Sensors*, 22, 6700.
- Hemal et al. (2024). *Sensors*, 24, 3682.
- Huang & Khabusi (2025). *Processes*, 13, 73.
- Udanor et al. (2022). *Data in Brief*, 43, 108400. Dataset: "Sensor Based
  Aquaponics Fish Pond Datasets", Kaggle, doi:10.34740/KAGGLE/DSV/3748790
  (CC BY 4.0).
- Leakage-aware re-analysis of the Udanor et al. dataset:
  github.com/MohamedElminur1/DL_Models_Fishweight. Its published SHA-256 sums
  were used to verify the raw files. Its finding that simple baselines matched
  deep models on this dataset is cited as described in the project brief; the
  re-analysis itself was not read while this code was written.
- NOAA National Estuarine Research Reserve System (NERRS). System-wide
  Monitoring Program. Data accessed from the NOAA NERRS Centralized Data
  Management Office website: http://www.nerrsdata.org
- Beck, M. W. (2016). SWMPr: An R package for retrieving, organizing, and
  analyzing environmental data for estuaries. *The R Journal*, 8(1), 219-232.
