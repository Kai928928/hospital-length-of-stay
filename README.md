# Hospital Length-of-Stay Prediction & Predictive Bed Scheduling

A reproducible machine-learning portfolio project at the intersection of
**healthcare analytics** and **operations research**: predict inpatient
length of stay (LOS) from real electronic health records (EHR), then use the
predictions to drive a small bed-allocation simulation that shows how
predictive scheduling reduces patient waiting without adding capacity.

> Data: [UCI *Diabetes 130-US Hospitals for Years 1999–2008*](https://archive.ics.uci.edu/dataset/296/diabetes)
> — 101,766 inpatient encounters across 130 US hospitals.

---

## Why this project

Most Kaggle-style projects stop at a leaderboard metric. This one closes the
loop: **predict patient demand → schedule scarce hospital resources**. That is
exactly the pattern behind real hospital-operations problems such as
pre-operative check scheduling: first estimate how long a patient needs the
resource, then use an optimization heuristic to allocate it. The two skills —
ML on EHR data and resource scheduling — are presented as one coherent story.

## Highlights

- **Real-world EHR data**, 100k+ records, widely cited in academic research
- End-to-end pipeline: EDA → cleaning → feature engineering → baseline
  reproduction → model improvement → evaluation → scheduling simulation
- **Honest evaluation**: only *predicted* LOS is used for scheduling decisions
- Fully reproducible: two commands download the data and run everything
- Result: **mean patient waiting time cut by ~36%** with unchanged bed capacity

## Project structure

```
hospital-length-of-stay/
├── data/
│   └── download_data.py        # downloads & extracts the UCI dataset
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modeling_and_scheduling.ipynb
├── scripts/
│   ├── train.py                # end-to-end pipeline runner
│   └── make_notebooks.py       # regenerates the notebooks
├── src/
│   ├── config.py               # paths, target, feature lists
│   ├── data_utils.py           # loading, cleaning, clinical features
│   ├── features.py             # encoding, train/test split
│   ├── models.py               # baseline + improved models, evaluation
│   └── scheduling.py           # bed-allocation simulation (FIFO vs SPT)
├── results/                    # figures, metrics, saved model (generated)
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python data/download_data.py    # download & extract diabetic_data.csv
python scripts/train.py         # run the full pipeline (~30 s on a laptop)
```

Open the notebooks for the step-by-step walkthrough:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## Methodology

### 1. Data understanding (EDA)
- Target `time_in_hospital` is heavily right-skewed (median 3 days); a
  log-transform is used for visualisation.
- Clinical workload features (`num_lab_procedures`, `num_medications`,
  `number_diagnoses`) correlate most strongly with LOS.

### 2. Cleaning
- Dropped identifiers (`encounter_id`, `patient_nbr`) and near-empty columns
  (`weight` ~97% missing, `payer_code` ~40% missing).
- Converted `age` bins to numeric midpoints.
- Collapsed hundreds of raw ICD-9 codes into **19 clinical categories**.
- Flagged unusual discharges (death / hospice / transfer / AMA).

### 3. Feature engineering
- `med_count`: 23 diabetes-medication indicator columns aggregated into one.
- `visit_history_total`: prior outpatient + emergency + inpatient visits.
- Ordinal encodings for glucose / A1C lab results.
- One-hot encoding for remaining categorical features.

### 4. Models (baseline → improvement)
| model | MAE (days) | RMSE | R² | train time |
|---|---|---|---|---|
| LinearRegression (baseline) | 1.842 | 2.420 | 0.327 | 0.2 s |
| RandomForest | 1.765 | 2.330 | 0.376 | 4.6 s |
| **HistGradientBoosting** | **1.726** | **2.290** | **0.398** | **1.5 s** |

*20% test split, fixed seed 42, full 101k-row dataset. Metrics reproduced by
`scripts/train.py`.*

### 5. Predictive bed allocation (the scheduling half)
A daily-batched discrete-event simulation over a 30-day horizon with a fixed
number of beds:

- **FIFO** — beds assigned first-come-first-served.
- **SPT (prediction-driven)** — each day, waiting patients are served in order
  of *predicted* LOS (shortest first), the classic scheduling rule that
  minimises total waiting time.

Result (1,200 patients, 80 beds, same arrival stream):

| policy | mean wait (days) | total wait (days) | max wait (days) | bed occupancy |
|---|---|---|---|---|
| FIFO | 7.18 | 8,617 | 15.0 | 0.977 |
| SPT (prediction-driven) | **4.57** | **5,484** | 28.0 | 0.977 |

**−36% mean waiting time with the same bed capacity**, at the cost of a higher
maximum wait — the classic trade-off of shortest-processing-time scheduling.

## Reproducibility notes

- Fixed random seed (`src/config.py: RANDOM_SEED = 42`).
- Feature importance falls back to **permutation importance** for models
  without native importance (e.g. HistGradientBoosting in recent scikit-learn),
  which is model-agnostic and more trustworthy.
- Optional `xgboost` is picked up automatically if installed.

## References

- Strack B, et al. (2014). *Impact of HbA1c Measurement on Hospital
  Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.*
  BioMed Research International.
- UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years
  1999–2008: <https://archive.ics.uci.edu/dataset/296/diabetes>

## License

Code: MIT. Data: see UCI repository terms.
