# flake8: noqa

from pathlib import Path
from typing import Any
import cv2
import numpy as np
import streamlit as st

from utils import get_all_encodings, add_or_update_encoding, delete_encoding

st.set_page_config(layout="wide")
st.title("Face Database: Add/Delete")
st.write("Select the option at the sidebar to add/delete subjects.")

KNOWN_FACE_PATHS: list[Path | np.ndarray] = []
KNOWN_FACE_ENCODINGS: list[Any] = []
KNOWN_FACE_NAMES: list[str] = []

KNOWN_FACE_PATHS, KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES = get_all_encodings()

menu = ["Add", "Delete"]
choice = st.sidebar.selectbox("Options", menu)

if choice == "Add":
    name = st.text_input("Name", placeholder="Enter name")

    upload = st.radio("Upload image or use webcam", ("Upload", "Webcam"))

    if upload == "Upload":
        uploaded_image = st.file_uploader("Upload", type=["jpg", "png", "jpeg", "pgm"])
        if uploaded_image is not None:
            st.image(uploaded_image)
            submit_btn = st.button("Submit", key="submit_btn")
            if submit_btn:
                if name == "":
                    st.error("Please enter name")
                else:
                    is_successful = add_or_update_encoding(uploaded_image, name)
                    if is_successful:
                        st.success("Subject added/updated")
                    else:
                        st.error(
                            "There is no face or more than one face in the picture"
                        )

    elif upload == "Webcam":
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        FRAME_WINDOW = st.image([])

        def submit_btn_callback():
            ret, frame = cam.read()
            
            if name == "":
                st.error("Please enter name")
            else:
                is_successful = add_or_update_encoding(frame, name)
                if is_successful:
                    st.success("Subject added/updated")
                else:
                    st.error(
                        "There is no face or more than one face in the picture"
                    )

        submit_btn = st.button("Take a picture", key="submit_btn", on_click=submit_btn_callback)

        while True:
            ret, frame = cam.read()
            if not ret:
                st.error(
                    "Failed to capture frame from camera. Enable camera permissions or ensure no other applications are using the camera."
                )
                st.stop()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(image)

elif choice == "Delete":

    def del_btn_callback(name):
        delete_encoding(name)
        st.success("Subject deleted")

    name = st.text_input("Subject Name", placeholder="Enter name")
    submit_btn = st.button("Submit", key="submit_btn")

    if submit_btn:
        try:
            index = KNOWN_FACE_NAMES.index(name)
        except ValueError:
            st.error("Subject does not exist")
        else:
            st.warning(
                "Recorded image of the subject. Confirm deletion?"
            )
            if type(KNOWN_FACE_PATHS[index]) == np.ndarray:
                image = cv2.cvtColor(KNOWN_FACE_PATHS[index], cv2.COLOR_BGR2RGB)
                st.image(image, width=256)
            else:
                st.image(str(KNOWN_FACE_PATHS[index]), width=256)
            del_btn = st.button(
                "Delete", key="del_btn", on_click=del_btn_callback, args=(name,)
            )
