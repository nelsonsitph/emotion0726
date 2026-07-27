import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fer import FER

# Page Config
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

st.title("🎭 Real-Time Emotion Detection")
st.write("Take a snapshot with your camera or upload a photo to analyze facial expressions.")

# Load AI model
@st.cache_resource
def load_detector():
    return FER(mtcnn=True)

detector = load_detector()

# Input choice
choice = st.radio("Choose Input Method:", ("Webcam Snapshot", "Upload Image"), horizontal=True)
image_input = None

if choice == "Webcam Snapshot":
    cam_file = st.camera_input("Take a picture")
    if cam_file:
        image_input = Image.open(cam_file)
else:
    up_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if up_file:
        image_input = Image.open(up_file)

# Detection logic
if image_input:
    img_array = np.array(image_input.convert("RGB"))
    
    with st.spinner("Analyzing faces..."):
        results = detector.detect_emotions(img_array)

    if not results:
        st.warning("No faces found. Try direct lighting or closer distance!")
        st.image(image_input, caption="Uploaded Image", use_container_width=True)
    else:
        annotated_img = img_array.copy()
        st.subheader("Analysis Results")

        for idx, face in enumerate(results, start=1):
            box = face["box"]
            emotions = face["emotions"]
            
            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]

            x, y, w, h = box
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 128), 3)
            cv2.putText(annotated_img, f"{top_emotion.capitalize()} ({score*100:.0f}%)", 
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)

            st.markdown(f"**Face #{idx}: Primary Emotion — `{top_emotion.upper()}` ({score*100:.1f}%)**")
            
            cols = st.columns(2)
            for i, (emo, val) in enumerate(emotions.items()):
                target_col = cols[i % 2]
                target_col.text(f"{emo.capitalize()}: {val*100:.1f}%")
                target_col.progress(float(val))
            st.divider()

        st.image(annotated_img, caption="Processed Image", use_container_width=True)
