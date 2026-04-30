import streamlit as st

st.set_page_config(
    page_title="HartleyNet - AI Image Detector",
    page_icon=None,
    layout="centered",
)


def render_header():
    st.markdown(
        "<h1 style='text-align:center;'>HartleyNet — AI Image Detector</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Upload an image to check whether it is AI-generated or real.</p>",
        unsafe_allow_html=True,
    )


def main():
    render_header()
    st.divider()
    st.write("App scaffolding in place. Uploader coming next.")


if __name__ == "__main__":
    main()
