# Hydraulic System Monitor

Predictive maintenance system for hydraulic components using machine learning.
Classifies the condition of 4 components (cooler, valve, pump, accumulator) from 17 sensor readings, served through a REST API.

---

## Live Demo

**API docs:** https://hydraulic-system-monitor-production.up.railway.app/docs

> **Note:** deployed on Railway's free tier, which sleeps the service after periods of inactivity. If the link returns a 404 or times out, it's asleep — try again in ~30s, or run it locally with the steps below.

**Example request:**
```bash
POST /predict
Content-Type: application/json

{
  "PS1": [152.0, 151.0, ...],
  "PS2": [104.0, 103.0, ...],
  "PS3": [...], "PS4": [...], "PS5": [...], "PS6": [...],
  "EPS1": [...],
  "FS1": [...], "FS2": [...],
  "TS1": [...], "TS2": [...], "TS3": [...], "TS4": [...],
  "VS1": [...], "CE": [...], "CP": [...], "SE": [...]
}
```

**Response:**
```json
{
  "predictions": {
    "cooler": {"condition": "full_efficiency", "confidence": 0.98},
    "valve": {"condition": "optimal", "confidence": 0.96},
    "pump": {"condition": "no_leakage", "confidence": 0.99},
    "accumulator": {"condition": "optimal", "confidence": 0.97}
  }
}
```

Each sensor array holds one full cycle of raw readings (length depends on the sensor's sampling rate — see [Features](#features) below). The API extracts the same statistical features used in training and runs them through the four trained models.

---

## Architecture

17 sensors → feature engineering (136 features) → 4 ML models → REST API

**Stack:** Python, scikit-learn, XGBoost, FastAPI, Docker, Railway

---

## Model Performance

| Component   | Model         | Test F1 | Test Accuracy |
|-------------|---------------|---------|----------------|
| Cooler      | RandomForest  | 1.000   | 1.000          |
| Valve       | RandomForest  | 0.979   | 0.979          |
| Pump        | XGBoost       | 0.993   | 0.993          |
| Accumulator | RandomForest  | 0.993   | 0.993          |

### Why the numbers are this high

These scores are not a sign of an unusually good model — they're a property of the dataset. The [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) dataset was collected on a test rig where each component's degradation was deliberately induced and directly instrumented (e.g. the pump's leakage state is set by the rig, not inferred). With 17 correlated sensors converted into 136 statistical features, several of these classes become linearly separable — a `full_efficiency` cooler cycle and a `near_failure` cooler cycle look nothing alike in pressure and temperature statistics. A 100% test score here reflects how distinguishable the classes are in this lab-controlled dataset, not that the model would generalize this cleanly to noisier, less-instrumented real-world sensor data.

**On data leakage:** the split is a single stratified `train_test_split` (80/20, stratified on the cooler target, `random_state=42`) applied once at the row (cycle) level, then reused via the same train/test indices for all four targets. This is **not** a leakage-safe split by experiment or time block — the dataset's cycles are not grouped by run/batch in a way this project's split accounts for, so cycles that are temporally or procedurally close to each other could end up split across train and test. This is a known limitation, not something the high scores compensate for — take the reported numbers as an upper bound on this dataset rather than a generalization guarantee.

---

## The Four Models

Each component is a separate multi-class classification problem, trained and evaluated independently.

- **Cooler** — 3 classes (full efficiency / reduced efficiency / near failure). RandomForest won on test F1.
- **Valve** — 4 classes (optimal / small lag / severe lag / near failure), the most imbalanced target. RandomForest won on test F1.
- **Pump** — 3 classes (no leakage / weak leakage / severe leakage). XGBoost won on test F1.
- **Accumulator** — 4 classes (optimal / slightly reduced / severely reduced / near failure). RandomForest won on test F1.

Model selection wasn't a per-component design decision made upfront — LogisticRegression, RandomForest, and XGBoost were trained and cross-validated (5-fold stratified) for all four components, and the model with the highest test F1 per component was kept. XGBoost only came out ahead for the pump; for the other three, RandomForest was equal or better. All pipelines include a `StandardScaler` step ahead of the model.

---

## Features

136 features = 17 raw sensor signals × 8 statistics computed per cycle.

**Sensors** (from the UCI rig, sampled at rates from 60 Hz to 6000 Hz depending on the sensor): 6 pressure sensors (PS1–PS6), motor power (EPS1), 2 volume flow sensors (FS1–FS2), 4 temperature sensors (TS1–TS4), vibration (VS1), cooling efficiency (CE), cooling power (CP), and efficiency factor (SE).

**Per-sensor aggregation:** each sensor's raw time series for a cycle is reduced to 8 descriptive statistics — mean, std, min, max, range, RMS, skewness, and kurtosis — regardless of the sensor's original sampling rate. This turns a variable-length raw signal into a fixed-length feature vector per cycle.

---

## Dataset

[Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) — UCI Machine Learning Repository

- 2205 cycles of 60 seconds, filtered to 1449 stable cycles
- 17 sensors at different sampling rates (60–6000 Hz)
- 4 classification targets

> Dataset not included in repo. Download from UCI and place files in `data/`.

Exploratory analysis (class distributions, sensor signal comparisons, correlations between features) is in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). The training pipeline — feature extraction, the train/test split, cross-validation, and model comparison — is in [`notebooks/training.ipynb`](notebooks/training.ipynb).

---

## Local Setup

```bash
git clone https://github.com/AlanVignolo/hydraulic-system-monitor
cd hydraulic-system-monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Docker

```bash
docker build -t hydraulic-monitor .
docker run -p 8000:8000 hydraulic-monitor
```
