"""HartleyNet model loader and inference helper.

``load_model`` builds the architecture via ``hartleynet_arch.hartleynet_builder``,
loads the fine-tuned state_dict from ``models/hartleynet.pth``, and caches the
result. ``predict`` runs a single image through the model and returns the
predicted label plus the raw AI probability.
"""

import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from hartleynet_arch import hartleynet_builder
from utils import preprocess_image


WEIGHTS_PATH = Path(__file__).parent / "models" / "hartleynet.pth"

LABEL_AI = "AI-generated"
LABEL_REAL = "Real"
DECISION_THRESHOLD = 0.5


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading HartleyNet weights...")
def load_model() -> torch.nn.Module:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Place hartleynet.pth in the models/ directory."
        )

    device = _device()
    model = hartleynet_builder(pretrained_backbone=False)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image: Image.Image) -> tuple[str, float, float]:
    """Run HartleyNet on a PIL image.

    Returns ``(label, prob_ai, inference_ms)`` where ``inference_ms``
    is the wall-clock time spent inside the model forward pass.
    """
    model = load_model()
    device = _device()

    tensor = preprocess_image(image).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor).view(-1)
        prob = torch.sigmoid(logit).item()
    inference_ms = (time.perf_counter() - start) * 1000.0

    label = LABEL_AI if prob >= DECISION_THRESHOLD else LABEL_REAL
    return label, prob, inference_ms
