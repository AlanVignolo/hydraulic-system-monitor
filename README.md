# Hydraulic System Monitor

Predictive maintenance system for hydraulic components using machine learning.
Classifies the condition of 4 components (cooler, valve, pump, accumulator) from 17 sensor readings.

**Live demo:** https://hydraulic-system-monitor-production.up.railway.app/docs

---

## Architecture

17 sensors → feature engineering (136 features) → 4 ML models → REST API

**Stack:** Python, scikit-learn, XGBoost, FastAPI, Docker, Railway

---

## Model Performance

| Component   | Model         | Test F1 | Test Accuracy |
|-------------|---------------|---------|---------------|
| Cooler      | RandomForest  | 1.000   | 1.000         |
| Valve       | RandomForest  | 0.979   | 0.979         |
| Pump        | XGBoost       | 0.993   | 0.993         |
| Accumulator | RandomForest  | 0.993   | 0.993         |

---

## Dataset

[Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) — UCI Machine Learning Repository

- 2205 cycles of 60 seconds, filtered to 1449 stable cycles
- 17 sensors at different sampling rates (60–6000 Hz)
- 4 classification targets

> Dataset not included in repo. Download from UCI and place files in `data/`.

---

## API Usage

**Health check:**
```bash
GET /health
```

**Predict component conditions:**
```bash
POST /predict
Content-Type: application/json

{
  "PS1": [152.0, 151.0, ...],
  "PS2": [104.0, 103.0, ...],
  ...
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

---

## Local Setup

```bash
git clone https://github.com/AlanVignolo/hydraulic-system-monitor
cd hydraulic-system-monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t hydraulic-monitor .
docker run -p 8000:8000 hydraulic-monitor
```