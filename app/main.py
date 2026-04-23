import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.ml.predictor import build_feacture_vector, load_models
from app.schemas.prediction import PredictionRequest, PredictionResponse, ComponentPrediction
from app.core.config import COMPONENTS, LABELS

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.update(load_models())
    print(f"Models loaded: {list(models.keys())}")
    yield
    models.clear()

app = FastAPI(
    title="Hydraulic System Monitor API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    sensor_data = request.model_dump()
    X = build_feacture_vector(sensor_data)
    
    predictions = {}
    for component in COMPONENTS:
        model = models[component]
        pred_class = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        predictions[component] = ComponentPrediction(
            condition=LABELS[component].get(pred_class, str(pred_class)), 
            confidence=confidence,
        )
    
    return PredictionResponse(predictions=predictions)

