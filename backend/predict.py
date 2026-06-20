import pickle
import numpy as np
import os

BASE          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE, "../models/best_model.pkl")
SCALER_PATH   = os.path.join(BASE, "../models/scaler.pkl")
FEATURES_PATH = os.path.join(BASE, "../models/feature_names.pkl")

with open(MODEL_PATH,    "rb") as f: model    = pickle.load(f)
with open(SCALER_PATH,   "rb") as f: scaler   = pickle.load(f)
with open(FEATURES_PATH, "rb") as f: FEATURES = pickle.load(f)

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def scale_input(input_data: dict) -> np.ndarray:
    """Arrange a user input dict into the correct feature order and scale it."""
    values = [input_data.get(f, 0) for f in FEATURES]
    return scaler.transform([values])


def predict_burnout(input_data: dict, return_scaled: bool = False):
    """
    Takes a dictionary of user inputs from the frontend, scales it,
    and returns the predicted burnout level as a string.

    If return_scaled=True, also returns the scaled numpy array —
    needed by agent.py to compute SHAP values without re-scaling.
    """
    scaled     = scale_input(input_data)
    prediction = model.predict(scaled)[0]
    label      = LABEL_MAP.get(int(prediction), "Medium")

    if return_scaled:
        return label, scaled
    return label