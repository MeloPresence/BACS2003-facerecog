# flake8: noqa

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from utils import get_all_encodings

st.set_page_config(layout="wide")

KNOWN_FACE_PATHS: list[Path | np.ndarray] = []
KNOWN_FACE_ENCODINGS: list[Any] = []
KNOWN_FACE_NAMES: list[str] = []

KNOWN_FACE_PATHS, KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES = get_all_encodings()


st.title("Face Database")

for path, encoding, name in zip(
    KNOWN_FACE_PATHS, KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
):
    if type(path) == np.ndarray:
        image = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        st.image(image, width=256)
        st.write(name)
    else:
        st.image(str(path), width=256)
        st.write(f"{name} ({path.name})")
