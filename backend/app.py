from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.services.predictor import predict_mri
from backend.services.llm_explainer import generate_medical_explanation
from backend.services.doctor_finder import get_doctors_by_city
from pydantic import BaseModel
from typing import Optional

# --- Schemas ---
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: dict

class ExplanationResponse(BaseModel):
    explanation: str
    disclaimer: str

class ExplanationRequest(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: dict

class Doctor(BaseModel):
    name: str
    specialization: str
    hospital: str
    contact: str
    city: str

class DoctorsResponse(BaseModel):
    city: str
    doctors: list[Doctor]

# --- App Initialization ---
app = FastAPI(title="NeuroScan AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "System Online", "service": "NeuroScan AI"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_route(file: UploadFile = File(...)):
    """Receives an MRI image and returns the classification."""
    return predict_mri(file)

@app.post("/explain", response_model=ExplanationResponse)
async def explain_route(data: ExplanationRequest):
    """Generates an AI explanation for the prediction."""
    return generate_medical_explanation(data.model_dump())

@app.get("/doctors", response_model=DoctorsResponse)
def doctors_route(city: str = Query(..., description="City to search for specialists")):
    """Finds doctors in the specified city."""
    return {
        "city": city,
        "doctors": get_doctors_by_city(city)
    }
