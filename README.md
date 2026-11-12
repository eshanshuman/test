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
python run_all.py                                                        # whole pipeline, both datasets
python tools/summarize_results.py --out reports/results_summary.md
python -m streamlit run dashboard.py
```

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
| `recommend.py` | Hybrid rules + model recommendation engine; 4 worked scenarios; surrogate decision tree (fig17). |
| `dashboard.py` | Streamlit + Plotly dashboard. |
| `run_all.py` | Runs the whole pipeline and times every step. |
| `tools/fetch_nerrs.py` | Downloads and cleans the real data. |
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

## 3. The two datasets

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

**The doses in `recommend.py` are unverified.** They are typical figures
recalled from Indian extension guidance (CIBA / MPEDA) and have not been
checked against the source documents; they are collected in the `DOSES`
dictionary, and every action tells the farmer to confirm with a local soil
and water test.

## 5. What each dataset does and does not show

| | Synthetic pond | Real (estuarine monitoring data) |
|---|---|---|
| Shows | That the pipeline recovers known pond dynamics; how much error is pure sensor noise; behaviour under a pond-like daily DO cycle. | That the pipeline survives real sensor noise, drift, QC flags and gaps; honest forecast skill on measured DO; transfer between two sites. |
| Does not show | Anything about a real pond: the dynamics and class balance were designed. Good results here are a check of the method, not field evidence. | Anything about a prawn pond: no aeration, feeding or stocking, and tides dominate. Status accuracy is 1.0 by construction. |

Neither dataset validates the system for a working vannamei farm. That needs
logged pond data, which the adapter in section 7 is built to accept.

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
- NOAA National Estuarine Research Reserve System (NERRS). System-wide
  Monitoring Program. Data accessed from the NOAA NERRS Centralized Data
  Management Office website: http://www.nerrsdata.org
- Beck, M. W. (2016). SWMPr: An R package for retrieving, organizing, and
  analyzing environmental data for estuaries. *The R Journal*, 8(1), 219-232.
