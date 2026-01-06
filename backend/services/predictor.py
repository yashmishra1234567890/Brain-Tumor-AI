import backend.core_utils.image_utils as img_utils
import os
import io
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Initialize Model
model = load_model(MODEL_PATH)

def predict_mri(image_input):
    """
    Process MRI image and return prediction results.
    Accepts: FastAPI UploadFile, bytes, or file-like object.
    """
    if hasattr(image_input, "file"):
        file_obj = image_input.file
    elif isinstance(image_input, bytes):
        file_obj = io.BytesIO(image_input)
    else:
        file_obj = image_input

    img_array = img_utils.preprocess_image(file_obj)
    predictions = model.predict(img_array)[0]
    
    max_idx = np.argmax(predictions)
    confidence = round(float(np.max(predictions)) * 100, 2)
    
    return {
        "prediction": CLASS_LABELS[int(max_idx)],
        "confidence": confidence,
        "all_probabilities": {
            label: round(float(prob) * 100, 2)
            for label, prob in zip(CLASS_LABELS, predictions)
        }
    }
