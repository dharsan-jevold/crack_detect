import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Crack Segmentation",
    layout="centered",
    page_icon="🧱"
)

# ------------------------------- UI STYLING -------------------------------
st.markdown("""
<style>
    .main { background-color: #0f1116; }
    .title { 
        text-align: center; 
        color: white; 
        font-size: 36px; 
        font-weight: 700;
        margin-top: -40px;
    }
    .subtitle {
        text-align: center; 
        color: #bbbbbb; 
        font-size: 18px; 
        margin-bottom: 25px;
    }
    .upload-card {
        background: #1a1d23;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🧱 Crack Segmentation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to detect and classify crack severity</div>", unsafe_allow_html=True)

# ------------------------------- LOAD MODEL -------------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/segment/train6/weights/best.pt")

model = load_model()

# ------------------------------- FILE UPLOAD -------------------------------
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------- FUNCTION: SEVERITY LEVEL -------------------------------
def classify_severity(mask):
    # mask = segmentation mask (0 or 255)
    crack_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    ratio = crack_pixels / total_pixels

    if ratio < 0.01:
        return "🟢 STABLE", ratio
    elif ratio < 0.05:
        return "🟡 GROWING", ratio
    else:
        return "🔴 CRITICAL", ratio


# ------------------------------- PROCESS IMAGE -------------------------------
if uploaded:
    # Show original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

    # Predict with YOLO
    results = model.predict(img, imgsz=640, conf=0.5)
    plotted = results[0].plot()  # overlay segmentation mask

    with col2:
        st.subheader("Predicted Mask")
        st.image(plotted, use_column_width=True)

    # Extract the segmentation mask
    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        # Classify crack severity
        status, ratio = classify_severity(mask)

        st.markdown("### Crack Severity Status")
        st.info(f"{status}  \nCrack Coverage: **{ratio*100:.2f}%**")

    else:
        st.error("No crack detected ❌")

else:
    st.info("⬆ Upload an image to begin crack detection.")
