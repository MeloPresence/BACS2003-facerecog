# flake8: noqa

import warnings
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np
import streamlit as st

from utils import get_all_encodings

warnings.filterwarnings("ignore")

DEFAULT_TOLERANCE = 0.6
KNOWN_FACE_PATHS: list[Path | np.ndarray] = []
KNOWN_FACE_ENCODINGS: list[Any] = []
KNOWN_FACE_NAMES: list[str] = []


def recognize_face(
    image: np.ndarray,
    *,
    locations_model: str = "hog",
    tolerance: float = DEFAULT_TOLERANCE,
):
    name = "???"

    face_locations = face_recognition.face_locations(image, model=locations_model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        matches = face_recognition.compare_faces(
            KNOWN_FACE_ENCODINGS, face_encoding, tolerance=tolerance
        )
        face_distances = face_recognition.face_distance(
            KNOWN_FACE_ENCODINGS, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        name = "???"

        if matches[best_match_index]:
            name = KNOWN_FACE_NAMES[best_match_index]
            distance = face_distances[best_match_index]
            cv2.putText(
                image,
                f"Diff: {distance:.2%}",
                (left, top - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (127, 0, 255),
                2,
            )
        cv2.rectangle(image, (left, top), (right, bottom), (127, 0, 255), 2)
        cv2.putText(
            image,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (127, 0, 255),
            2,
        )
    return image, name


st.set_page_config(layout="wide")
KNOWN_FACE_PATHS, KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES = get_all_encodings()
st.sidebar.title("Settings")

# Create a menu bar
menu = ["Upload Image", "Webcam"]
choice = st.sidebar.selectbox("Input type", menu)
# Put slide to adjust tolerance
tolerance = st.sidebar.slider("Tolerance", 0.0, 1.0, DEFAULT_TOLERANCE, 0.01)
st.sidebar.info(
    "How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance."
)

if choice == "Upload Image":
    st.title("Face Recognition (Upload Image)")
    st.write("Change Input type to 'Webcam' in sidebar to use webcam feed.")
    uploaded_images = st.file_uploader(
        "Upload", type=["jpg", "png", "jpeg", "pgm"], accept_multiple_files=True
    )
    if len(uploaded_images) != 0:
        for image in uploaded_images:
            image = face_recognition.load_image_file(image)
            image, name = recognize_face(image, tolerance=tolerance)
            st.image(image)
    else:
        st.info("Please upload an image")

elif choice == "Webcam":
    st.title("Face Recognition (Webcam)")
    st.write("Change Input type to 'Upload Image' in sidebar to upload image.")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error(
                "Failed to capture frame from camera. Enable camera permissions or ensure no other applications are using the camera."
            )
            st.stop()
        image, name = recognize_face(frame, tolerance=tolerance)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(image)
