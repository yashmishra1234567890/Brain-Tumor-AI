# 🧠 NeuroScan AI — Brain Tumor Detection & Analysis

**NeuroScan AI** is an AI-powered medical assistance system that analyzes **brain MRI scans** to provide **probabilistic tumor classification**, **AI-generated medical explanations**, and **city-based hospital recommendations**.

The system combines a **custom-trained Convolutional Neural Network (CNN)** with **Generative AI (LLM)** to deliver a production-style, end-to-end AI workflow.

> ⚠️ **Disclaimer**  
> This project is for **educational and research purposes only**.  
> It is **NOT a medical device** and must **NOT** be used for diagnosis or treatment decisions.

---

## 🔗 Live Demo

| Component | Link |
|----------|------|
| 🧠 Live Application (Streamlit) | https://brain-tumor-ai-4u6rxcjzx8qqnkdkdhkaba.streamlit.app/ |

> The live demo executes backend logic directly inside Streamlit for simplicity, while the FastAPI layer is preserved for production deployment.

---

## ✨ Key Features

### 🔍 Brain Tumor Classification
- **Custom CNN trained from scratch** on MRI images  
- **No pre-trained weights used**
- Achieves **~84% validation accuracy**
- Supports 4 classes:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- Outputs **class probabilities**, not just labels

### 🤖 AI Medical Explanation
- Uses **Mistral-7B-Instruct via OpenRouter**
- Converts predictions into **human-readable medical insights**
- Strictly constrained to **avoid diagnosis or treatment advice**

### 👨‍⚕️ Hospital Recommendation
- City-based hospital suggestions
- Uses **curated sample datasets** (no scraping, no reviews)
- Ranked using heuristic signals:
  - Hospital type (Government / Academic / Multi-specialty)
  - Relevant specialization (Neurology / Neurosurgery)

### 📊 Interactive Dashboard
- Built with **Streamlit**
- MRI image preview
- Confidence & probability visualization
- AI explanation panel with medical disclaimers

---

## 🏗️ System Architecture
```
Streamlit UI
     │
     │ (Cloud Mode: Direct Function Calls)
     │ (Local Mode: REST API)
     ▼
  AI Core
     ├── CNN Model Inference (TensorFlow/Keras)
     ├── LLM Explanation (OpenRouter / Mistral)
     ├── Hospital Recommendation Logic
     └── Structured Response Schema
```

---

## 🛠️ Tech Stack

- **Machine Learning**: TensorFlow, Keras, NumPy
- **Deep Learning Model**: Custom CNN (trained from scratch)
- **Frontend**: Streamlit
- **Backend**: FastAPI (local / production)
- **GenAI / LLM**: OpenRouter (`mistralai/mistral-7b-instruct`)
- **Utilities**: Pillow, Requests, Python-Dotenv

---

## 📂 Project Structure

```
Brain_tumor/
├── backend/
│   ├── app.py                    # FastAPI backend
│   ├── run.py                    # Backend launcher
│   ├── model/
│   │   └── model.h5              # Trained CNN model
│   ├── services/
│   │   ├── predictor.py          # CNN inference logic
│   │   ├── llm_explainer.py      # LLM explanation logic
│   │   └── doctor_finder.py      # Hospital recommendation logic
│   ├── utils/                    # Image & response helpers
│   └── data/
│       └── doctors.json          # Curated hospital dataset
│
├── frontend/
│   ├── streamlit_app.py          # Streamlit UI
│   └── requirements.txt
│
├── notebooks/
│   └── brain_diagnosis.ipynb     # Model training & experiments
│
├── image_test_sample/            # Sample MRI images
└── README.md
```



---

## 📊 Dataset Used

- **Brain Tumor MRI Dataset (Kaggle)**
- 4 classes: Glioma, Meningioma, Pituitary, No Tumor
- ~7,000 MRI images
- Used to train the CNN **from scratch**

---

## 🧪 How to Use

1. Open the live Streamlit app  
2. Upload a brain MRI image (or use a sample image)  
3. Click **Analyze Image**
4. View:
   - Predicted tumor class
   - Confidence & probability distribution
   - AI-generated medical explanation
5. Enter a city to view **recommended hospitals**

---

## 🧠 Design Decisions & Ethics

### ❌ What this project does NOT do
- No scraping of commercial platforms (Practo, Justdial, Google Maps)
- No fake ratings or reviews
- No medical diagnosis or treatment advice
- No medical device claims

### ✅ What this project DOES
- Uses curated, representative hospital datasets
- Provides transparent probabilistic outputs
- Includes medical disclaimers at every stage
- Restricts LLMs to **explanation only**

-----------------------------------------------------------------------

