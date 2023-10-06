# flake8: noqa

import pickle
import random
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import streamlit as st

FACES_PATH = Path("faces")
PICKLE_DB_PATH = Path("faceDatabase.pickle")


@st.cache_data
def get_builtin_encodings():
    known_face_paths = []
    known_face_encodings = []
    known_face_names = []

    for subject_path in FACES_PATH.iterdir():
        is_learnable = False
        face_locations = []
        image_paths = list(subject_path.iterdir())
        while not is_learnable:
            random_image_path = random.choice(image_paths)
            image = face_recognition.load_image_file(str(random_image_path))
            face_locations = face_recognition.face_locations(image, model="hog")
            is_learnable = len(face_locations) == 1
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_face_paths.append(random_image_path)
        known_face_encodings.append(face_encoding)
        known_face_names.append(subject_path.name)
        print(f"Learned subject {subject_path.name} using {random_image_path.name}")
    return (known_face_paths, known_face_encodings, known_face_names)


def get_all_encodings():
    if PICKLE_DB_PATH.exists():
        with PICKLE_DB_PATH.open("rb") as f:
            db = pickle.load(f)
            return (
                db["known_face_paths"],
                db["known_face_encodings"],
                db["known_face_names"],
            )

    known_face_paths, known_face_encodings, known_face_names = get_builtin_encodings()
    db = {
        "known_face_paths": known_face_paths,
        "known_face_encodings": known_face_encodings,
        "known_face_names": known_face_names,
    }
    with PICKLE_DB_PATH.open("wb") as f:
        pickle.dump(db, f)
    return (db["known_face_paths"], db["known_face_encodings"], db["known_face_names"])


def add_or_update_encoding(image, name: str, *, locations_model: str = "hog") -> bool:
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    face_locations = face_recognition.face_locations(image, model=locations_model)
    if len(face_locations) != 1:
        return False
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    with PICKLE_DB_PATH.open("rb") as f:
        db = pickle.load(f)

    try:
        index = db["known_face_names"].index(name)
    except ValueError:
        db["known_face_names"].append(name)
        db["known_face_paths"].append(image)
        db["known_face_encodings"].append(face_encoding)
    else:
        db["known_face_paths"][index] = image
        db["known_face_encodings"][index] = face_encoding

    with PICKLE_DB_PATH.open("wb") as f:
        pickle.dump(db, f)

    return True


def delete_encoding(name: str) -> bool:
    with PICKLE_DB_PATH.open("rb") as f:
        db = pickle.load(f)

    try:
        index = db["known_face_names"].index(name)
    except ValueError:
        return False
    else:
        del db["known_face_names"][index]
        del db["known_face_paths"][index]
        del db["known_face_encodings"][index]

    with PICKLE_DB_PATH.open("wb") as f:
        pickle.dump(db, f)

    return True
