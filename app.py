import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Lightweight Face & Smile Detector", page_icon="😊", layout="centered")

st.title("😊 Face & Smile Expression Detector")
st.write("A lightweight, fast expression analyzer running on OpenCV.")

# Load built-in OpenCV Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Choice input
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

if image_input:
    # Convert image format
    img_array = np.array(image_input.convert("RGB"))
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please ensure clear lighting and direct camera angle.")
        st.image(image_input, caption="Uploaded Image", use_container_width=True)
    else:
        annotated_img = img_array.copy()
        st.subheader("Analysis Results")

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            # Extract face region for smile analysis
            roi_gray = gray_img[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            
            # Determine expression
            if len(smiles) > 0:
                detected_emotion = "HAPPY / SMILING 😊"
                color = (0, 255, 0) # Green box
            else:
                detected_emotion = "NEUTRAL / SERIOUS 😐"
                color = (255, 165, 0) # Orange box

            # Draw box around face
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(annotated_img, detected_emotion, (x, max(y - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            st.markdown(f"**Face #{idx}: Detected Expression — `{detected_emotion}`**")

        st.image(annotated_img, caption="Processed Image", use_container_width=True)
