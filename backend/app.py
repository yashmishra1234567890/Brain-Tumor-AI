from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from services.predictor import predict_mri
from services.llm_explainer import generate_medical_explanation as explain_prediction
from services.doctor_finder import get_doctors_by_city
from core_utils.response_schema import (
    PredictionResponse, 
    ExplanationResponse, 
    DoctorsResponse,
    ExplanationRequest
)

app = FastAPI(title="Brain Tumor AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    return predict_mri(file)

@app.post("/explain", response_model=ExplanationResponse)
async def explain(data: ExplanationRequest):
    return explain_prediction(data.model_dump())

@app.get("/doctors", response_model=DoctorsResponse)
def doctors(city: str = Query(...)):
    return {
        "city": city,
        "doctors": get_doctors_by_city(city)
    }
