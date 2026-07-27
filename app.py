import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Page Config
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

st.title("🎭 Real-Time Emotion Detection")
st.write("Take a snapshot with your camera or upload a photo to analyze facial expressions.")

# Choose Input
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
    
    with st.spinner("Analyzing facial expressions..."):
        try:
            # Analyze emotions using DeepFace
            results = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
            
            if not isinstance(results, list):
                results = [results]

            annotated_img = img_array.copy()
            st.subheader("Analysis Results")

            for idx, face in enumerate(results, start=1):
                region = face["region"]
                emotions = face["emotion"]
                top_emotion = face["dominant_emotion"]
                score = emotions[top_emotion]

                # Draw rectangle
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 128), 3)
                cv2.putText(annotated_img, f"{top_emotion.capitalize()} ({score:.0f}%)", 
                            (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)

                st.markdown(f"**Face #{idx}: Primary Emotion — `{top_emotion.upper()}` ({score:.1f}%)**")
                
                # Show confidence breakdown
                cols = st.columns(2)
                for i, (emo, val) in enumerate(emotions.items()):
                    target_col = cols[i % 2]
                    target_col.text(f"{emo.capitalize()}: {val:.1f}%")
                    target_col.progress(float(val) / 100.0)
                st.divider()

            st.image(annotated_img, caption="Processed Image", use_container_width=True)

        except Exception as e:
            st.error("Could not process image. Please try another photo with clear lighting.")
