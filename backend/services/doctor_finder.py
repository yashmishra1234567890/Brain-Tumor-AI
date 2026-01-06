import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "doctors.json")

def get_doctors_by_city(city: str):
    with open(DATA_PATH, "r") as f:
        doctors = json.load(f)

    city = city.lower()
    return [d for d in doctors if d["city"].lower() == city]
