"""Shared helpers for HartleyNet inference.

``preprocess_image`` reproduces the training-time transform pipeline:
resize to 224x224, convert to a float32 tensor in [0, 1], then normalise
with the ImageNet statistics the model was trained on.

The history helpers manage a bounded ring of recent predictions stored
in Streamlit's session state.
"""

import torch
from PIL import Image
from torchvision.transforms import v2

IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MAX_HISTORY = 10
THUMBNAIL_SIZE = 64

_to_tensor = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMG_SIZE, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Return a (1, 3, 224, 224) tensor ready for HartleyNet."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = _to_tensor(image)
    return tensor.unsqueeze(0)


def make_thumbnail(image: Image.Image, size: int = THUMBNAIL_SIZE) -> Image.Image:
    """Return a square-ish thumbnail copy suitable for the history list."""
    thumb = image.copy()
    thumb.thumbnail((size, size))
    return thumb


def append_history(history: list, entry: dict, max_items: int = MAX_HISTORY) -> list:
    """Append ``entry`` and trim the list to the most recent ``max_items``."""
    history.append(entry)
    overflow = len(history) - max_items
    if overflow > 0:
        del history[:overflow]
    return history
