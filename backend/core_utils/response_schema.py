from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: Dict[str, float]

class ExplanationRequest(PredictionResponse):
    pass

class ExplanationResponse(BaseModel):
    explanation: str
    disclaimer: str

class Doctor(BaseModel):
    city: str
    name: str
    type: str # e.g. Government, Multi-specialty
    specialization: str
    confidence_level: str
    reason: str

class DoctorsResponse(BaseModel):
    city: str
    doctors: List[Doctor]
