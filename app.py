from io import BytesIO

import streamlit as st
from PIL import Image

from model import predict

st.set_page_config(
    page_title="HartleyNet - AI Image Detector",
    page_icon=None,
    layout="centered",
)

ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp"]
PREVIEW_WIDTH = 400


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


def render_result(label: str, prob: float):
    st.subheader("The verdict")
    st.markdown(f"**Classification:** {label}")
    st.markdown(f"**Raw probability (AI):** `{prob:.4f}`")


def main():
    render_header()
    st.divider()

    uploaded = render_uploader()
    if uploaded is None:
        st.caption("Supported formats: " + ", ".join(ACCEPTED_TYPES))
        return

    image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
    render_preview(image)

    label, prob = predict(image)
    render_result(label, prob)


if __name__ == "__main__":
    main()
