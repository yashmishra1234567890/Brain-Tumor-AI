import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from core_utils.image_utils import preprocess_image

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")

CLASS_LABELS = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

# Load Model
model = load_model(MODEL_PATH)

def predict_mri(file):
    img_array = preprocess_image(file.file)
    preds = model.predict(img_array)[0]

    result = {
        "prediction": CLASS_LABELS[int(np.argmax(preds))],
        "confidence": round(float(np.max(preds)) * 100, 2),
        "all_probabilities": {
            CLASS_LABELS[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(CLASS_LABELS))
        }
    }
    return result
