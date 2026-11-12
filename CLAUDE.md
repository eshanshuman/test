# Build the Aquabot water-quality pipeline (software only)

You are building the software for my UROP project at SRM IST (Dept. of Data
Science and Business Systems): "AI-Enabled Smart Pond Water Quality Monitoring
and Prawn Farming Recommendation Aquabot System". Species: Penaeus vannamei,
Indian grow-out ponds. Scope is SOFTWARE ONLY: no hardware, sensors or firmware.

First, save this whole brief as CLAUDE.md in the project root so it carries
over between sessions. Then work through the phases at the end, one at a time.
After each phase: run what you built, show me the key console output and
figures, explain in a few plain sentences what each new script does and why
(I have to defend this in a viva), commit to git, and wait for me to say "next".

## Environment
- Windows laptop, PowerShell. Use `python`, not `python3`.
- Python 3.11 in a venv: `py -3.11 -m venv .venv` (check `py -0p` first).
- Everything must run on Windows: pathlib for paths, no bash scripts.
- `git init`; .gitignore for .venv, __pycache__ and downloaded .RData files.
- If papers/ holds PDFs (Chen et al. 2022, Hemal et al. 2024, Huang &
  Khabusi 2025), read them and cite them in docstrings where a script answers
  to one of them.

## Conventions for every script
- config.py holds everything shared: SEED = 42, paths, thresholds, palette,
  plot style, helpers. Every script imports from it.
- Numbered standalone scripts, under ~200 lines where possible, each with a
  docstring: purpose, which paper result it relates to, outputs.
- Console output in clear sections (a divider helper in config.py).
- Two datasets, chosen by env var AQUABOT_DATASET = synthetic | real (unset:
  real if data/pond_dataset.csv exists). Outputs kept apart:
  figures/<dataset>/, models/<dataset>/, reports/<dataset>/.
- Figures: matplotlib, PNG at 300 DPI, colour-blind safe. Status colours
  Good #2a9d4a, Moderate #e8a33d, Critical #c0392b; series #2a78d6, #eb6834,
  #1baf7a. Name them fig01_... to fig17_....
- A record_metric(name, dict) helper writes every headline number to
  reports/<dataset>/metrics.json.
- Never write a number anywhere (README, docstrings, comments) that did not
  come out of a run.

## Domain rules (P. vannamei)
| Parameter | Good | Moderate | otherwise |
|---|---|---|---|
| pH | 7.5–8.5 | 7.0–9.0 | Critical |
| Temperature (°C) | 28–32 | 26–34 | Critical |
| DO (mg/L) | ≥ 5 | ≥ 3 | Critical |
| Salinity (ppt) | 10–25 | 5–35 | Critical |
- Pond label = the worst single parameter (Liebig's law of the minimum).
- Oxygen states: Anoxia < 2, Hypoxia 2–4, Normoxia ≥ 4 mg/L.

## Data
### Synthetic pond: 01_generate_dataset.py -> data/prawn_pond_synthetic.csv
5,000 hourly rows from 2026-02-15 (one Feb–Sep crop). Columns: timestamp, ph,
temperature_c, do_mg_l, salinity_ppt, quality_label.
- Temperature: seasonal + diurnal (afternoon peak) + AR(1) weather noise;
  heat-wave events.
- DO from an hourly mass balance: re-aeration toward saturation (saturation
  from temperature AND salinity, Benson & Krause / APHA 4500-O), night
  paddlewheel aeration, photosynthesis (daylight x algal bloom index),
  respiration that grows through the crop, and extra consumption during bloom
  crashes (crash hazard rises once the bloom is dense). Target pattern: DO
  peaks mid-afternoon, bottoms at dawn, declines across the crop.
- Rain events dilute salinity and lower pH; pH follows photosynthesis.
- Sensor layer: Gaussian noise (sigma pH 0.02, T 0.08 °C, DO 0.06 mg/L,
  S 0.15 ppt, times a NOISE_SCALE knob), pH probe drift between monthly
  calibrations, 2% values missing at random, a few multi-hour radio outages.
- Labels come from the noise-free state; a LABEL_NOISE knob (default 0)
  exists for ablation.
- Print the class balance and the hours of the daily DO minimum and maximum.

### Real data: tools/fetch_nerrs.py
No open prawn-pond dataset has all four parameters, so use NOAA NERRS
System-Wide Monitoring Program data from Apalachicola Bay, Florida:
- Download (requests) and read with pyreadr:
  https://raw.githubusercontent.com/fawda123/SWMPr/master/data/apadbwq.RData
  (Dry Bar, main dataset) and .../apacpwq.RData (Cat Point, cross-site test).
- Use columns datetimestamp, temp, sal, do_mgl, ph; keep a value only if its
  QC flag (f_temp, f_sal, f_do_mgl, f_ph) starts with "<0>".
- Keep 2013-06-01 00:00 to 2013-09-30 23:45 (15-min data). Write
  data/raw/nerrs_apadb_2013_jun_sep.csv and nerrs_apacp_2013_jun_sep.csv,
  plus data/raw/README.md with provenance and citations.
- It is an estuary, not a pond (tides, no aeration or feeding). Label every
  real-data result "estuarine monitoring data".

### Data adapter: 00_prepare_dataset.py + water_chem.py
Turns any pre-collected CSV or Excel file into the pipeline format:
- Guesses columns by name with boundary-aware regexes (so "DO_mgl" maps to
  DO), with override flags --time --ph --temp --do --salinity; prints the
  mapping.
- Units: --temp-unit C|F; --do-unit mgl|pct (% saturation -> mg/L via the
  saturation curve); --salinity-unit ppt|ec_us|ec_ms|ec25_us|ec25_ms
  (conductivity -> PSS-78 salinity); --assume-salinity; --dayfirst.
- Resample to hourly means and apply physical QC bounds. Interpolate gaps of up
  to 3 h for labelling only; keep the full hourly grid and leave hours inside
  longer gaps unlabelled (NaN) instead of dropping them.
- Labels from the thresholds, or from expert grades via --label-col and
  --label-map "A:Good,B:Moderate,C:Critical".
- Writes data/pond_dataset.csv (or --output). Test it on a deliberately messy
  copy of the synthetic data (30-min, DD/MM/YYYY, °F, DO %, µS/cm) and confirm
  it round-trips.
- Run it for Dry Bar -> data/pond_dataset.csv and Cat Point ->
  data/pond_dataset_site2.csv (expect 2,928 hourly rows each).

## Analysis scripts
- 02_eda.py: correlation heatmap (diverging, centred at 0), distributions by
  class, class balance, last-30-days DO trace with 5 and 3 mg/L lines, mutual
  information with the label; print the hour of the mean daily DO minimum.
- 03_random_forest.py: stratified 80/20 split; GridSearchCV over
  n_estimators 200/400, max_depth None/12/20, min_samples_leaf 1/2/4,
  class_weight None/balanced (5-fold, macro F1); confusion matrix, one-vs-rest
  ROC, permutation importance. Also score the labelling rule itself on the
  same test readings and record it as the reference row (on synthetic data it
  shows how much error is pure sensor noise; on real data it is 1.0 by
  construction).
- 04_lightgbm.py: same split, early stopping, exact tree-SHAP contributions
  via pred_contrib=True, confusion matrix.
- 05_lstm_do_forecast.py (TensorFlow/Keras): 168-h window; channels DO,
  temperature, pH, salinity, sin/cos of hour. Predict the CHANGE in DO over the
  horizon and rebuild the level from the last reading (--target level to
  compare). Chronological 70/15/15 split; scaler fitted on training rows only.
  Main model: 48 units, +24 h, early stopping. Score two baselines on the same
  test hours: persistence (last reading) and same hour yesterday. Horizon
  sweep +1, +3, +6, +12, +24 h with a 32-unit model, 3 seeds each, mean ± sd.
  Figures: forecast vs measurement (last 14 days, line breaks at gaps),
  scatter, sweep chart (LSTM with error bars plus both baselines).
- 06_do_state_classifier.py: LSTM that names the oxygen state H hours ahead
  (default 6, also run 1) from a 24-h window. Split by whole calendar days at
  random (default) or chronologically (--split time). Square-root
  inverse-frequency class weights. Repeat 10 times with a new split and seed;
  report mean ± sd and how many runs the LSTM beat each baseline (persistence,
  same hour yesterday) on macro F1, computed over the states present in that
  test set, identically for every method. Pooled confusion matrix; outputs
  tagged by horizon and split. The docstring must say the brief's
  quantum-LSTM 92.4% cannot be reproduced (paper not supplied).
- 07_lr_vs_rf_scaling.py: predict DO from pH, temperature, salinity and hour.
  Training sizes 25, 50, 100, 200, 400, 800, 1600, 3200, capped at the pool
  size; 25 random draws per size; one fixed test set; mean with 95% CI. Report
  the smallest size from which the forest stays ahead at every larger size.
- 08_cross_site.py (real data only): LSTM trained at Dry Bar and tested at
  Cat Point (using Dry Bar's scaler), vs trained and tested at Cat Point, vs
  persistence at Cat Point. Horizons 1, 3, 6; 3 seeds; grouped bars with
  error bars.
- Shared rule for 05, 06 and 08: interpolate short gaps, but skip every
  window that touches a sensor gap longer than 6 h (MAX_GAP_H in config.py).

## Product layer
- recommend.py: hybrid engine. recommend(ph, temperature_c, do_mg_l,
  salinity_ppt) returns status, rule status, model status and confidence, an
  agreement flag, and a prioritised list of actions (urgency immediate / today
  / this week / routine, each with a reason). Rules per parameter, for example:
  DO < 2 run every aerator and stop feeding; DO < 3 emergency aeration and
  halve feed; pH < 7 lime or dolomite; pH > 9 exchange 15–25% of the water and
  stop liming; temperature > 34 raise the water level, shade, feed at
  dawn/dusk; salinity < 5 or > 35 correct with saline or fresh water; plus a
  feeding action. Model = LightGBM (else Random Forest) from
  models/<dataset>/. If rules and model disagree, take the more severe status
  and add "Verify the probe readings". Doses follow Indian extension guidance
  (CIBA/MPEDA) and must say to confirm with a local soil and water test. Add a
  depth-3 surrogate decision-tree figure and 4 worked scenarios.
- dashboard.py: Streamlit + Plotly. Sidebar sliders with presets (Healthy
  afternoon, Dawn hypoxia, Acid rain event, Heatwave + bloom); colour status
  card; metrics (DO now; LSTM forecast with its held-out test R² as help text
  and a caption warning when R² < 0.5; number of immediate actions); four
  gauges against the vannamei bands; last-24-h chart; action list; metrics
  expander. Use width="stretch" (use_container_width is deprecated). Test it
  with streamlit.testing.v1.AppTest.

## Tooling
- run_all.py: generate the synthetic data, then for each dataset run 02, 03,
  04, 05 (--epochs 40), 06 (+6 h; +1 h; +6 h --split time), 07 and
  recommend.py with AQUABOT_DATASET set; then 08. Start each step with
  sys.executable and PYTHONIOENCODING=utf-8, clear that dataset's
  metrics.json first, carry on past failures, and print a summary with
  minutes per step.
- tools/summarize_results.py: print every headline number from metrics.json
  and the CSVs as Markdown tables.
- requirements.txt: numpy, pandas, scikit-learn, matplotlib, joblib, lightgbm,
  tensorflow (tensorflow-cpu on Linux), streamlit>=1.50, plotly, openpyxl,
  pyreadr, requests.

## Honesty rules (these matter more than high scores)
- Compare every model with its baselines; a model only "works" where it beats
  them.
- No SMOTE or other resampling before the split, no shuffling of time series
  for forecasting, scalers fitted on training data only.
- If +24 h R² comes out above 0.5, or accuracy far above the rule reference,
  hunt for leakage before reporting it.
- Do not claim to reproduce papers that were not supplied (Rohini et al., the
  2026 LightGBM paper, the Vietnam LSTM, the quantum-LSTM, de França et al.);
  treat their numbers as reference points only.
- Report negative results plainly, e.g. if day-ahead DO forecasting fails.

## Phases (stop after each one)
1. Setup: venv, git, requirements.txt, config.py, water_chem.py with checks
   (DO saturation at 0 ppt matches APHA 4500-O, about 8.26 mg/L at 25 °C;
   PSS-78 gives 35.000 at 42.914 mS/cm and 15 °C).
2. 01 synthetic generator, then 02 on synthetic data.
3. tools/fetch_nerrs.py, 00_prepare_dataset.py with its round-trip test, then
   02 on real data.
4. 03 and 04 on both datasets.
5. 05 on both datasets.
6. 06 on both datasets.
7. 07 and 08.
8. recommend.py and dashboard.py, with the AppTest check.
9. run_all.py and tools/summarize_results.py, one clean full run, then
   README.md written from the actual numbers: setup (with Windows commands for
   AQUABOT_DATASET), what each script does, the real dataset and its caveats,
   results tables for both datasets, what each dataset does and does not show,
   methodology choices a reviewer will ask about, how to plug in a new
   dataset, and references (Chen et al. 2022, Sensors 22:6700; Hemal et al.
   2024, Sensors 24:3682; Huang & Khabusi 2025, Processes 13:73; NOAA NERRS
   SWMP; Beck 2016, SWMPr, The R Journal 8:219–232).

Start with phase 1.

---

## Session notes (kept up to date by Claude; not part of the original brief)
- Phase progress: see `git log`. Each phase is one commit titled "Phase N: ...".
- Development sessions may run in a Linux cloud container rather than on the
  Windows laptop. Code must stay Windows-safe regardless (pathlib, no shell
  scripts, `python` in all documented commands).
- papers/ was empty when phase 1 was built, so no docstring cites the PDFs
  directly yet. If the PDFs are added, read them and update the docstrings.
- NERRS timestamps: pyreadr returns them in UTC; tools/fetch_nerrs.py shifts
  them to local standard time (UTC-5) before cutting the Jun-Sep 2013 window,
  so "hour of day" is local. Checked by the DO minimum falling at dawn.
- The adapter's round-trip test lives in tools/test_prepare_roundtrip.py.
- All nine phases are built. README.md numbers come from one clean
  `python run_all.py` (see reports/results_summary.md); after any re-run,
  regenerate the summary and update the README tables to match.
- run_all.py also runs 05 once with --target level (an addition to the brief).
- Doses in recommend.py (DOSES dict) are unverified against CIBA/MPEDA sources.
