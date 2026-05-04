"""HartleyNet model loader.

Builds the model architecture via ``hartleynet_arch.hartleynet_builder``,
loads the fine-tuned state_dict from ``models/hartleynet.pth``, and
caches the result so Streamlit reuses one instance across reruns.
"""

from pathlib import Path

import streamlit as st
import torch

from hartleynet_arch import hartleynet_builder


WEIGHTS_PATH = Path(__file__).parent / "models" / "hartleynet.pth"


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
