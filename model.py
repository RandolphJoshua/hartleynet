"""HartleyNet model loader and inference helper.

``load_model`` builds the architecture via ``hartleynet_arch.hartleynet_builder``,
loads the selected state_dict from ``models/`` (keyed by display name in
``MODEL_OPTIONS``), and caches one model instance per key. ``predict`` runs a
single image through the selected model and returns the predicted label plus
the raw AI probability.
"""

import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from hartleynet_arch import hartleynet_builder
from utils import preprocess_image


MODELS_DIR = Path(__file__).parent / "models"

MODEL_OPTIONS: dict[str, str] = {
    "HartleyNet — current": "hartleynet.pth",
    "HartleyNet — previous": "hartleynet_old.pth",
}
DEFAULT_MODEL = "HartleyNet — current"

LABEL_AI = "AI-generated"
LABEL_REAL = "Real"
DECISION_THRESHOLD = 0.5


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_models() -> dict[str, str]:
    """Return the subset of MODEL_OPTIONS whose weight file exists on disk."""
    return {name: fname for name, fname in MODEL_OPTIONS.items() if (MODELS_DIR / fname).exists()}


def _weights_path(model_key: str) -> Path:
    if model_key not in MODEL_OPTIONS:
        raise KeyError(f"Unknown model '{model_key}'. Expected one of: {list(MODEL_OPTIONS)}.")
    return MODELS_DIR / MODEL_OPTIONS[model_key]


@st.cache_resource(show_spinner="Loading HartleyNet weights...")
def load_model(model_key: str) -> torch.nn.Module:
    path = _weights_path(model_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. "
            f"Place {path.name} in the models/ directory."
        )

    device = _device()
    model = hartleynet_builder(pretrained_backbone=False)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image: Image.Image, model_key: str = DEFAULT_MODEL) -> tuple[str, float, float]:
    """Run the selected HartleyNet model on a PIL image.

    Returns ``(label, prob_ai, inference_ms)`` where ``inference_ms``
    is the wall-clock time spent inside the model forward pass.
    """
    model = load_model(model_key)
    device = _device()

    tensor = preprocess_image(image).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor).view(-1)
        prob = torch.sigmoid(logit).item()
    inference_ms = (time.perf_counter() - start) * 1000.0

    label = LABEL_AI if prob >= DECISION_THRESHOLD else LABEL_REAL
    return label, prob, inference_ms
