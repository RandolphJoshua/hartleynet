from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model import predict
from utils import append_history, make_thumbnail

st.set_page_config(
    page_title="HartleyNet - AI Image Detector",
    page_icon=None,
    layout="centered",
)

ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp"]
PREVIEW_WIDTH = 400
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def _ensure_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None


def render_header():
    st.markdown(
        "<h1 style='text-align:center;'>HartleyNet — AI Image Detector</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Upload an image to check whether it is AI-generated or real.</p>",
        unsafe_allow_html=True,
    )


def render_uploader():
    return st.file_uploader(
        "Drop an image here",
        type=ACCEPTED_TYPES,
        accept_multiple_files=False,
    )


def render_preview(image: Image.Image):
    st.image(image, width=PREVIEW_WIDTH, caption="Uploaded image")


def render_result(label: str, prob: float, inference_ms: float):
    confidence = prob if label == "AI-generated" else 1.0 - prob

    st.subheader("The verdict")
    st.markdown(f"**Classification:** {label}")
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")
    st.markdown(f"**Raw probability (AI):** `{prob:.4f}`")
    st.markdown(f"**Inference time:** `{inference_ms:.1f} ms`")


def render_history_sidebar():
    with st.sidebar:
        st.header("History")
        history = st.session_state.history
        if not history:
            st.caption("No predictions yet this session.")
            return

        for entry in reversed(history):
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(entry["thumbnail"], width=64)
            with cols[1]:
                st.markdown(f"**{entry['label']}**")
                st.caption(
                    f"{entry['confidence'] * 100:.1f}% · {entry['timestamp']}"
                )


def _record_history(uploaded, image: Image.Image, label: str, prob: float):
    if st.session_state.last_file_id == uploaded.file_id:
        return
    st.session_state.last_file_id = uploaded.file_id

    confidence = prob if label == "AI-generated" else 1.0 - prob
    append_history(
        st.session_state.history,
        {
            "thumbnail": make_thumbnail(image),
            "label": label,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        },
    )


def _load_image(uploaded) -> Image.Image | None:
    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        st.error(
            f"That image is {len(raw) / (1024 * 1024):.1f} MB - the demo "
            f"caps uploads at {limit_mb} MB to keep things responsive. "
            "Try a smaller version."
        )
        return None

    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError:
        st.error(
            "That file does not look like an image we can decode. "
            f"Please upload a {', '.join(ACCEPTED_TYPES)} file."
        )
    except OSError as exc:
        st.error(f"Could not read the image - the file may be truncated or corrupt. ({exc})")
    return None


def _run_inference(image: Image.Image):
    try:
        return predict(image)
    except FileNotFoundError as exc:
        st.error(
            "HartleyNet weights are missing from this deployment. "
            "Place hartleynet.pth under models/ and redeploy."
        )
        st.caption(str(exc))
    except Exception as exc:
        st.error("Something went wrong while running the model.")
        st.caption(f"{type(exc).__name__}: {exc}")
    return None


def main():
    _ensure_session_state()
    render_header()
    st.divider()
    render_history_sidebar()

    uploaded = render_uploader()
    if uploaded is None:
        st.caption("Supported formats: " + ", ".join(ACCEPTED_TYPES))
        return

    image = _load_image(uploaded)
    if image is None:
        return

    render_preview(image)

    result = _run_inference(image)
    if result is None:
        return

    label, prob, inference_ms = result
    render_result(label, prob, inference_ms)
    _record_history(uploaded, image, label, prob)


if __name__ == "__main__":
    main()
