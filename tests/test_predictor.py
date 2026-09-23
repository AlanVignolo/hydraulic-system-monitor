import numpy as np

from app.ml.predictor import SENSOR_ORDER, build_feacture_vector, extract_features, load_models


def test_extract_features_known_values():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = extract_features(arr)

    assert features["mean"] == 3.0
    assert features["min"] == 1.0
    assert features["max"] == 5.0
    assert features["range"] == 4.0
    assert set(features.keys()) == {
        "mean", "std", "min", "max", "range", "rms", "skewness", "kurtosis"
    }


def test_build_feacture_vector_shape(sample_sensor_data):
    X = build_feacture_vector(sample_sensor_data)

    assert X.shape == (1, len(SENSOR_ORDER) * 8)


def test_build_feacture_vector_handles_nan():
    flat_data = {sensor: [7.0] for sensor in SENSOR_ORDER}
    X = build_feacture_vector(flat_data)

    assert not np.isnan(X).any()


def test_load_models_returns_all_components():
    models = load_models()

    assert set(models.keys()) == {"cooler", "valve", "pump", "accumulator"}
    for model in models.values():
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
