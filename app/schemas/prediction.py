from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    PS1 : list[float]
    PS2 : list[float]
    PS3 : list[float]
    PS4 : list[float]
    PS5 : list[float]
    PS6 : list[float]
    EPS1 : list[float]
    FS1 : list[float]
    FS2 : list[float]
    TS1 : list[float]
    TS2 : list[float]
    TS3 : list[float]
    TS4 : list[float]
    VS1 : list[float]
    CE : list[float]
    CP : list[float]
    SE : list[float]

class ComponentPrediction(BaseModel):
    condition: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: Dict[str, ComponentPrediction]