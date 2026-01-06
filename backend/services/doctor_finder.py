import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "doctor.json")

def get_doctors_by_city(city: str):
    if not os.path.exists(DATA_PATH):
        return []
        
    try:
        with open(DATA_PATH, "r") as f:
            doctors = json.load(f)
    except Exception:
        return []

    city = city.lower().strip()
    return [d for d in doctors if d.get("city", "").lower() == city]

