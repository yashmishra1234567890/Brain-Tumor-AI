import streamlit as st
import requests
import os
import sys

# --- Page Config ---
st.set_page_config(
    page_title="AI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
DEPLOYMENT_MODE = "Cloud" # "Cloud" for Streamlit Share, "Local" for separate Backend
BACKEND_URL = "http://localhost:8001"

# --- Backend Integration (Cloud Mode) ---
if DEPLOYMENT_MODE == "Cloud":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    
    # Load Environment Variables
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(PROJECT_ROOT, '.env')
        if os.path.exists(env_path): load_dotenv(env_path)
    except: pass
    
    # Load Secrets
    try:
        if st.secrets:
            for k, v in st.secrets.items(): os.environ[k] = str(v)
    except: pass

    try:
        from backend.services.predictor import predict_mri
        from backend.services.llm_explainer import generate_medical_explanation
        from backend.services.doctor_finder import get_doctors_by_city
    except ImportError as e:
        st.error(f"Backend modules missing: {e}")
        st.stop()

# --- Custom Styling (Matches Clean Medical UI) ---
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain--v1.png", width=80)
    st.title("NeuroScan AI")
    st.markdown("### Intelligent Diagnostic Assistant")
    st.info("Upload an MRI scan to detect tumors and receive AI-generated medical explanations.")
    st.markdown("---")
    st.markdown("**/ Status**: 🟢 System Online")
    st.markdown("**/ Version**: 2.0.0 (Cloud)")

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload MRI Scan")
    
    # Input Selection
    input_type = st.radio("Select Input:", ["Upload Image", "Sample Image"], horizontal=True)
    
    image_bytes = None
    display_image = None
    
    if input_type == "Upload Image":
        uploaded = st.file_uploader("Choose MRI Image...", type=["jpg", "jpeg", "png"])
        if uploaded:
            image_bytes = uploaded.getvalue()
            display_image = uploaded
            st.image(uploaded, caption="Preview", use_container_width=True)
            
    else:
        sample_dir = os.path.join(os.path.dirname(__file__), "..", "image_test_sample")
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('jpg','png'))]
            idx = st.selectbox("Choose Sample:", files)
            if idx:
                path = os.path.join(sample_dir, idx)
                st.image(path, caption=idx, use_container_width=True)
                with open(path, "rb") as f:
                    image_bytes = f.read()

with col2:
    st.subheader("📊 Analysis Results")
    
    if image_bytes:
        # Use session state to persist results across reruns (e.g. when typing city name)
        if st.button("🔍 Analyze Scan", type="primary", use_container_width=True):
            with st.spinner("Running Deep Learning Analysis..."):
                try:
                    # 1. Prediction
                    if DEPLOYMENT_MODE == "Cloud":
                        res = predict_mri(image_bytes)
                    else:
                        files = {"file": ("img.jpg", image_bytes, "image/jpeg")}
                        res = requests.post(f"{BACKEND_URL}/predict", files=files).json()
                    
                    # 2. AI Explanation
                    if DEPLOYMENT_MODE == "Cloud":
                        expl_res = generate_medical_explanation(res)
                    else:
                        expl_res = requests.post(f"{BACKEND_URL}/explain", json=res).json()
                        
                    # Store in session state
                    st.session_state['analysis_done'] = True
                    st.session_state['prediction_result'] = res
                    st.session_state['explanation_result'] = expl_res
                    
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

        # Display content if analysis is done
        if st.session_state.get('analysis_done'):
            res = st.session_state['prediction_result']
            expl_res = st.session_state['explanation_result']

            # Display Prediction Card
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style='margin:0; color:#2c3e50;'>{res['prediction']}</h2>
                <p style='color:#7f8c8d;'>Confidence Score</p>
                <h1 style='color:#27ae60;'>{res['confidence']}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            
            # Probabilities
            st.caption("Class Probabilities")
            probs = res['all_probabilities']
            for label, prob in probs.items():
                st.progress(int(prob), text=f"{label}: {prob}%")

            st.divider()

            # AI Explanation
            st.subheader("🤖 AI Medical Insight")
            with st.expander("Read AI Explanation", expanded=True):
                st.markdown(expl_res.get("explanation", "Narrative unavailable."))
                st.warning(f"Note: {expl_res.get('disclaimer')}")

            st.divider()

            # Doctor Finder
            with st.expander("👨‍⚕️ Locate Specialists Nearby"):
                with st.form("doctor_finder_form"):
                    city = st.text_input("Enter City Name (e.g., Mumbai, Indore)")
                    submit_search = st.form_submit_button("Find Doctors")
                
                if submit_search and city:
                    if DEPLOYMENT_MODE == "Cloud":
                        docs = get_doctors_by_city(city)
                    else:
                        try:
                            resp = requests.get(f"{BACKEND_URL}/doctors?city={city}", timeout=5)
                            if resp.status_code == 200:
                                docs = resp.json().get("doctors", [])
                            else:
                                docs = []
                        except:
                            docs = []
                    
                    if docs:
                        for d in docs:
                            with st.container():
                                st.markdown(f"### {d.get('name')}")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown(f"**Type:** {d.get('type')}")
                                    st.markdown(f"**Specialization:** {d.get('specialization')}")
                                with col_b:
                                    st.markdown(f"**Confidence:** {d.get('confidence_level')}")
                                
                                st.markdown(f"_{d.get('reason')}_")
                                st.divider()
                    else:
                        st.info(f"No specialists found in '{city}'.")
    else:
        st.info("👈 Please select or upload an MRI image to begin.")
        # Reset state if no image
        if 'analysis_done' in st.session_state:
            del st.session_state['analysis_done']



