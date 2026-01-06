import streamlit as st
import requests
import time
import os

# --- Configuration ---
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://localhost:8001"

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
        # relative path to image_test_sample from frontend directory (where app runs presumably?)
        # Actually standard is running from root. Let's check.
        # If running `streamlit run frontend/streamlit_app.py` from root `D:\Brain_tumar`
        # Then `image_test_sample` is in `image_test_sample`.
        sample_dir = "image_test_sample"
        # Handle case if running from frontend dir
        if not os.path.exists(sample_dir) and os.path.exists("../image_test_sample"):
            sample_dir = "../image_test_sample"
            
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
                    # Send request to backend
                    # Send bytes with a filename
                    files = {"file": (file_name, image_data, "image/jpeg")}
                    response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
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
                                doc_res = requests.get(f"{BACKEND_URL}/doctors", params={"city": city}).json()
                                doctors = doc_res.get("doctors", [])
                                
                                if doctors:
                                    for doc in doctors:
                                        with st.expander(f"{doc['name']} ({doc['specialization']})"):
                                            st.write(f"**Hospital:** {doc.get('name')}") # Data json structure seemed mixed in context, adjusting generic
                                            st.write(f"**Type:** {doc.get('type')}")
                                            st.write(f"**Confidence:** {doc.get('confidence_level')}")
                                            st.write(f"**Reason:** {doc.get('reason')}")
                                else:
                                    st.info(f"No specialists found in {city}.")
                                    
                    else:
                        st.error(f"Prediction failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("👈 Please upload or select an MRI image to start the analysis.")


