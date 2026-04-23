import joblib
import numpy as np
from scipy import stats
from app.core.config import MODELS_PATH, COMPONENTS

SENSOR_ORDER = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1", "CE", "CP", "SE"
]

def extract_features(sensor_array: list) -> dict:
    arr = np.array(sensor_array)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "range": np.max(arr) - np.min(arr),
        "rms": np.sqrt(np.mean(arr**2)),
        "skewness": stats.skew(arr),
        "kurtosis": stats.kurtosis(arr),
    }
    
def build_feacture_vector(sensor_data: dict) -> np.ndarray:
    features = {}
    for sensor in SENSOR_ORDER:
        sensor_features = extract_features(sensor_data[sensor])
        for feat_name, feat_value in sensor_features.items():
            features[f"{sensor}_{feat_name}"] = feat_value
    values = [0.0 if np.isnan(v) else v for v in features.values()]
    return np.array(values).reshape(1, -1)

def load_models() -> dict:
    models = {}
    for component in COMPONENTS:
        path = MODELS_PATH / f"{component}_model.pkl"
        models[component] = joblib.load(path)
    return models

