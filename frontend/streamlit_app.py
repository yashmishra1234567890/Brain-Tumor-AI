import streamlit as st
import requests
import time
import os
import sys

# --- Configuration ---
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Toggle this for deployment
# Options: "Cloud" (Streamlit Share), "Local" (FastAPI running separately)
DEPLOYMENT_MODE = "Cloud"

BACKEND_URL = "http://localhost:8001"

# Setup Backend Path for Cloud Mode
if DEPLOYMENT_MODE == "Cloud":
    # Get the absolute path to the project root (one level up from frontend)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    
    # Import backend services directly
    try:
        from backend.services.predictor import predict_mri
        from backend.services.llm_explainer import generate_medical_explanation
        from backend.services.doctor_finder import get_doctors_by_city
        
        # Load secrets into os.environ for backend modules to see
        # Priority: 1. Streamlit Secrets (Cloud), 2. dotenv (Local)
        try:
            # Try loading from .env file for local testing
            from dotenv import load_dotenv
            load_dotenv()
            
            if hasattr(st, "secrets"):
                for key, value in st.secrets.items():
                    os.environ[key] = str(value)
        except Exception:
            pass  # No secrets found, likely running locally without secrets.toml
                
    except ImportError as e:
        st.error(f"Failed to import backend modules: {e}")
        st.stop()

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    .status-badge {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        color: white;
    }
    .status-online {
        background-color: #28a745;
    }
    .status-offline {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def check_api_health():
    if DEPLOYMENT_MODE == "Cloud":
        # In cloud mode, if imports worked, we are "online"
        return True, "Integrated Mode"
    else:
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=2)
            if response.status_code == 200:
                return True, response.json().get("status", "Unknown")
        except requests.exceptions.ConnectionError:
            return False, "Connection refused"
        except Exception as e:
            return False, str(e)
        return False, "Unknown Error"

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 NeuroScan AI")
    st.markdown("---")
    
    st.subheader("System Status")
    is_online, status_msg = check_api_health()
    
    if is_online:
        st.markdown(f'<div class="status-badge status-online">🟢 API Online</div>', unsafe_allow_html=True)
        st.caption(f"Response: {status_msg}")
    else:
        st.markdown(f'<div class="status-badge status-offline">🔴 API Offline</div>', unsafe_allow_html=True)
        if DEPLOYMENT_MODE == "Local":
            st.caption("Please ensure the backend server is running on port 8001.")
        
    st.markdown("---")
    st.info(
        "**About**\n\n"
        "This advanced MRI diagnostic assistant uses Deep Learning to classify brain tumors into 4 categories:\n"
        "- Glioma\n- Meningioma\n- Pituitary\n- No Tumor"
    )

# --- Main Page ---
st.title("🏥 Brain Tumor MRI Analysis")
st.markdown("Upload a brain MRI scan to detect potential tumors and generate medical insights.")

# Layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Input Image")
    
    input_method = st.radio("Choose input method:", ["Upload Image", "Use Sample Image"])
    
    image_data = None
    file_name = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded MRI Scan", use_container_width=True)
            image_data = uploaded_file.getvalue()
            file_name = uploaded_file.name

    else:
        # relative path to image_test_sample from root
        sample_dir = "image_test_sample"
        # Handle case if running from frontend dir (locally mostly)
        if not os.path.exists(sample_dir) and os.path.exists(os.path.join("..", sample_dir)):
             sample_dir = os.path.join("..", sample_dir)
            
        if os.path.exists(sample_dir):
            sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            selected_sample = st.selectbox("Select a sample image:", sample_files)
            
            if selected_sample:
                file_path = os.path.join(sample_dir, selected_sample)
                st.image(file_path, caption=f"Sample: {selected_sample}", use_container_width=True)
                with open(file_path, "rb") as f:
                    image_data = f.read()
                file_name = selected_sample
        else:
            st.warning("Sample directory not found.")

with col2:
    st.subheader("2. Analysis Results")
    
    if image_data:
        if not is_online:
            st.error("Cannot proceed. Backend API is offline.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    result = None
                    
                    if DEPLOYMENT_MODE == "Cloud":
                        # Direct call
                        result = predict_mri(image_data)
                    else:
                        # HTTP Call
                        files = {"file": (file_name, image_data, "image/jpeg")}
                        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=10)
                        if response.status_code == 200:
                            result = response.json()
                        else:
                            st.error(f"Prediction failed: {response.status_code}")
                    
                    if result:
                        prediction = result['prediction']
                        confidence = result['confidence']
                        probs = result['all_probabilities']
                        
                        # Display Metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Prediction", prediction)
                        m2.metric("Confidence", f"{confidence}%")
                        
                        # Display Probability Chart
                        st.subheader("Class Probabilities")
                        st.bar_chart(probs)
                        
                        # Advanced Features Tabs
                        tab1, tab2 = st.tabs(["🤖 AI Explanation", "👨‍⚕️ Find Specialist"])
                        
                        with tab1:
                            if st.button("Generate Detailed Explanation"):
                                with st.spinner("Consulting AI Specialist..."):
                                    expl_res = {}
                                    if DEPLOYMENT_MODE == "Cloud":
                                        expl_res = generate_medical_explanation(result)
                                    else:
                                        expl_res = requests.post(
                                            f"{BACKEND_URL}/explain",
                                            json=result
                                        ).json()
                                        
                                    st.markdown("### Medical Insight")
                                    st.write(expl_res.get("explanation", "No explanation available."))
                                    st.warning(f"⚠️ **Disclaimer:** {expl_res.get('disclaimer')}")
                                    
                        with tab2:
                            city = st.text_input("Enter your city to find nearby neurologists:", placeholder="e.g., Indore, Mumbai")
                            if city:
                                doctors = []
                                if DEPLOYMENT_MODE == "Cloud":
                                     # The finder returns a list directly
                                     doctors = get_doctors_by_city(city)
                                else:
                                    doc_res = requests.get(f"{BACKEND_URL}/doctors", params={"city": city}).json()
                                    doctors = doc_res.get("doctors", [])
                                
                                if doctors:
                                    for doc in doctors:
                                        with st.expander(f"{doc['name']} ({doc['specialization']})"):
                                            st.write(f"**Hospital:** {doc.get('hospital', doc.get('name'))}")
                                            st.write(f"**Contact:** {doc.get('contact', 'N/A')}")
                                            
                                else:
                                    st.info(f"No specialists found in {city}.")
                        
                except Exception as e:
                    # Detailed error for debugging in cloud
                    import traceback
                    st.error(f"Error during analysis: {str(e)}")
                    if DEPLOYMENT_MODE == "Cloud":
                        st.text(traceback.format_exc())
    else:
        st.info("👈 Please upload or select an MRI image to start the analysis.")


