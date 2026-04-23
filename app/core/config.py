from pathlib import Path

MODELS_PATH = Path(__file__).parent.parent.parent / "models"

COMPONENTS = ["cooler", "valve", "pump", "accumulator"]

LABELS = {
    "cooler": {0: "full_efficiency", 1: "near_failure", 2: "reduced_efficiency"},
    "valve": {0: "near_failure", 1: "optimal", 2: "severe_lag", 3: "small_lag"},
    "pump": {0: "no_leakage", 1: "weak_leakage", 2: "severe_leakage"},
    "accumulator": {0: "near_failure", 1: "optimal", 2: "severely_reduced", 3: "slightly_reduced"},
}