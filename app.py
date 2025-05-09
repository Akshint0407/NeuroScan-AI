import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Model download and loading function
@st.cache_resource
def load_model():
    # Google Drive file ID from your link
    file_id = "16TsTtcWskH8HTiM1K3FI12EPK1hR3Fb0"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "brain_tumor_model.h5"
    
    try:
        gdown.download(url, output, quiet=False)
        model = tf.keras.models.load_model(output)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    
# Header Section
st.title("🧠 NeuroScan AI - Brain Tumor Detection")
st.markdown("---")

# Main Content
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan...",
        type=["png", "jpg", "jpeg"],
        help="Upload a brain MRI scan for tumor detection"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

with col2:
    st.header("Analysis Results")
    if uploaded_file is not None:
        with st.spinner('Analyzing MRI scan...'):
            model = load_model()
            
            if model:
                # Preprocess image
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                prediction = model.predict(img_array)
                probability = prediction[0][0]
                
                st.subheader("Diagnosis Report")
                
                if probability > 0.5:
                    st.error("⚠️ Tumor Detected")
                    st.markdown(f"""
                    <div class="result-box tumor">
                        <h3>Abnormality Detected</h3>
                        <p>Confidence: {probability*100:.2f}%</p>
                        <p>Recommendation: Consult a specialist immediately</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ No Tumor Detected")
                    st.markdown(f"""
                    <div class="result-box healthy">
                        <h3>Normal Scan Detected</h3>
                        <p>Confidence: {(1-probability)*100:.2f}%</p>
                        <p>Recommendation: Regular checkups recommended</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence meter
                st.subheader("Confidence Level")
                target_probability = probability if probability > 0.5 else 1 - probability
                st.progress(float(target_probability))

# Sidebar
with st.sidebar:
    st.markdown("# About NeuroScan AI")
    st.markdown("""
    **NeuroScan AI** is a deep learning-powered diagnostic tool that analyzes MRI scans 
    to detect potential brain tumors using convolutional neural networks.
    """)
    
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a brain MRI scan (axial view preferred)
    2. Our AI model processes the image
    3. Get instant results with confidence levels
    4. Review recommendations
    """)
    
    st.markdown("### Model Information")
    st.markdown("""
    - **Architecture**: Custom CNN
    - **Training Data**: 4000+ MRI scans
    - **Accuracy**: 92.68% validation accuracy
    - **Last Updated**: October 2023
    """)

    st.markdown("---")
    st.markdown("Developed by Akshint, Dhruv and Pranav  \nContact: [Akshint0407](https://github.com/Akshint0407)")

# Add some empty space
st.markdown("<br><br>", unsafe_allow_html=True)
