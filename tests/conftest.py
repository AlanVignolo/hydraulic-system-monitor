import numpy as np
import pytest

from app.ml.predictor import SENSOR_ORDER

SENSOR_LENGTHS = {
    "PS1": 6000, "PS2": 6000, "PS3": 6000, "PS4": 6000,
    "PS5": 6000, "PS6": 6000, "EPS1": 6000,
    "FS1": 600, "FS2": 600,
    "TS1": 60, "TS2": 60, "TS3": 60, "TS4": 60,
    "VS1": 60, "CE": 60, "CP": 60, "SE": 60,
}


@pytest.fixture
def sample_sensor_data():
    rng = np.random.default_rng(seed=42)
    return {
        sensor: rng.normal(loc=100.0, scale=5.0, size=SENSOR_LENGTHS[sensor]).tolist()
        for sensor in SENSOR_ORDER
    }
