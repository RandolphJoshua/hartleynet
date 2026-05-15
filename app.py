import base64
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
    initial_sidebar_state="expanded",
)

ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp"]
PREVIEW_WIDTH = 400
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

LABEL_AI = "AI-generated"
LABEL_REAL = "Real"


# --- Editorial-print styling -------------------------------------------------

_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"] {
    font-family: 'Newsreader', Georgia, serif;
    color: #111111;
    background: #FAF7F0;
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.main .block-container {
    max-width: 720px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    font-family: 'Playfair Display', 'Newsreader', serif;
    color: #111111;
    letter-spacing: -0.01em;
}

code, kbd, samp, pre,
[data-testid="stMarkdownContainer"] code {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    background: transparent !important;
    color: #111111 !important;
    padding: 0 2px !important;
    font-size: 0.95em;
}

/* masthead --------------------------------------------------------------- */

.hn-masthead {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #555555;
    border-top: 1px solid #111111;
    border-bottom: 1px solid #111111;
    padding: 8px 2px;
    margin-bottom: 36px;
}

.hn-masthead .right { color: #862B30; }

.hn-title {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 64px;
    line-height: 1.02;
    text-align: center;
    margin: 24px 0 6px 0;
    color: #111111;
}

.hn-subtitle {
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 19px;
    text-align: center;
    color: #444444;
    margin: 0 0 6px 0;
}

.hn-byline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    text-align: center;
    color: #777777;
    margin: 0 0 28px 0;
}

.hn-ornament {
    text-align: center;
    letter-spacing: 0.6em;
    color: #862B30;
    margin: 20px 0 28px 0;
    font-size: 14px;
}

.hn-rule {
    border: 0;
    border-top: 1px solid #111111;
    margin: 28px 0;
}

.hn-rule.thin {
    border-top: 1px solid #C9C0AE;
    margin: 16px 0;
}

/* directive line above uploader / preview */
.hn-directive {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #862B30;
    margin: 8px 0 8px 0;
    text-align: center;
}

/* file uploader ---------------------------------------------------------- */

[data-testid="stFileUploaderDropzone"] {
    background: #FAF7F0 !important;
    border: 1px solid #111111 !important;
    border-radius: 0 !important;
    padding: 32px 24px !important;
    transition: border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #862B30 !important;
    background: #FAF7F0 !important;
}

/* hide Streamlit's default cloud-upload icon (scoped to instructions
   so the delete-button X on the uploaded-file pill stays visible) */
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none !important; }

[data-testid="stFileUploaderDropzoneInstructions"] {
    align-items: center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div {
    text-align: left;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-family: 'Newsreader', serif !important;
    font-style: italic !important;
    font-size: 17px !important;
    color: #222222 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #888888 !important;
    margin-top: 4px;
    display: inline-block;
}

[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    color: #111111 !important;
    border: 1px solid #111111 !important;
    border-radius: 0 !important;
    font-family: 'Newsreader', serif !important;
    font-style: italic !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
    padding: 8px 22px !important;
    margin-left: 18px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #862B30 !important;
    color: #FAF7F0 !important;
    border-color: #862B30 !important;
}
[data-testid="stFileUploaderDropzone"] button:focus,
[data-testid="stFileUploaderDropzone"] button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px #FAF7F0, 0 0 0 3px #862B30 !important;
}

[data-testid="stFileUploader"] label {
    display: none;
}

/* uploaded-file pill that Streamlit drops in/under the dropzone */
[data-testid="stFileUploaderFile"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #555555 !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDeleteBtn"] {
    background: transparent !important;
    color: #862B30 !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDeleteBtn"] svg {
    display: inline-block !important;
    visibility: visible !important;
    fill: #862B30 !important;
    color: #862B30 !important;
    width: 18px !important;
    height: 18px !important;
}
[data-testid="stFileUploaderDeleteBtn"]:hover svg {
    fill: #111111 !important;
    color: #111111 !important;
}

/* preview (main canvas only) -------------------------------------------- */

.main [data-testid="stImage"] {
    border: 1px solid #111111;
    padding: 8px;
    background: #FFFFFF;
}
.main [data-testid="stImage"] figcaption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #555555;
    text-align: center;
    padding-top: 8px;
}

/* verdict block ---------------------------------------------------------- */

.hn-verdict {
    margin-top: 36px;
    padding: 28px 24px 24px 24px;
    border: 1px solid #111111;
    background: #FAF7F0;
    position: relative;
}
.hn-verdict::before {
    content: "THE VERDICT";
    position: absolute;
    top: -10px;
    left: 24px;
    background: #FAF7F0;
    padding: 0 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.35em;
    color: #862B30;
}

.hn-verdict-prelude {
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 16px;
    text-align: center;
    color: #555555;
    margin: 4px 0 6px 0;
}

.hn-verdict-class {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 48px;
    text-align: center;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 4px 0 10px 0;
    line-height: 1.05;
}
.hn-verdict-class.ai { color: #862B30; }
.hn-verdict-class.real { color: #1E3A5F; }

.hn-verdict-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #333333;
    margin: 12px 4px 14px 4px;
    border-top: 1px solid #111111;
    padding-top: 14px;
    letter-spacing: 0.04em;
}
.hn-verdict-meta .label {
    color: #862B30;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 2px;
}
.hn-verdict-meta .value {
    font-size: 16px;
    color: #111111;
    font-feature-settings: "tnum" 1;
}

/* progress bar ----------------------------------------------------------- */

[data-testid="stProgress"] {
    margin: 0;
}
[data-testid="stProgress"] > div > div {
    background: transparent !important;
    border-radius: 0 !important;
}
[data-testid="stProgress"] > div > div > div {
    background: #E8DFC9 !important;
    border-radius: 0 !important;
    height: 14px !important;
}
[data-testid="stProgress"] > div > div > div > div {
    background: #862B30 !important;
    border-radius: 0 !important;
}
[data-testid="stProgressLabel"], [data-testid="stProgress"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #555555 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* sidebar / history ------------------------------------------------------ */

[data-testid="stSidebar"] {
    background: #F2EBDC !important;
    border-right: 1px solid #111111 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 26px !important;
    border-bottom: 1px solid #111111;
    padding-bottom: 6px;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .hn-issue {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #862B30;
    margin-bottom: 18px;
}

/* strip the heavy main-canvas image frame from sidebar thumbnails */
[data-testid="stSidebar"] [data-testid="stImage"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stElementContainer"] {
    background: transparent !important;
}

.hn-history-list {
    margin-top: 4px;
}
.hn-history-row {
    display: flex;
    gap: 12px;
    align-items: center;
    border-top: 1px solid #C9C0AE;
    padding: 12px 2px;
}
.hn-history-row:last-child {
    border-bottom: 1px solid #C9C0AE;
}
.hn-history-thumb {
    width: 52px;
    height: 52px;
    object-fit: cover;
    border: 1px solid #1A1A1A;
    flex-shrink: 0;
    display: block;
}
.hn-history-info {
    display: flex;
    flex-direction: column;
    min-width: 0;
}
.hn-history-label {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 15px;
    color: #111111;
    line-height: 1.1;
    letter-spacing: 0.01em;
}
.hn-history-label.ai { color: #862B30; }
.hn-history-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    color: #555555;
    margin-top: 4px;
    text-transform: uppercase;
}

/* footer ----------------------------------------------------------------- */

.hn-footer {
    text-align: center;
    margin-top: 56px;
    padding-top: 14px;
    border-top: 1px solid #111111;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #777777;
}
.hn-footer .accent { color: #862B30; }

/* misc Streamlit overrides ---------------------------------------------- */

[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-left: 4px solid #862B30 !important;
    background: #F5ECD9 !important;
    color: #111111 !important;
    font-family: 'Newsreader', serif !important;
}
[data-testid="stAlert"] svg { color: #862B30 !important; }

[data-testid="stCaptionContainer"], .stCaption {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10.5px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #777777 !important;
}

hr { display: none; }
</style>
"""


def inject_style():
    st.markdown(_STYLE, unsafe_allow_html=True)


# --- Session state ----------------------------------------------------------


def _ensure_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None
    if "issue_no" not in st.session_state:
        st.session_state.issue_no = datetime.now().strftime("%j")


# --- Rendering --------------------------------------------------------------


def render_masthead():
    today = datetime.now().strftime("%B %d, %Y").upper()
    issue = st.session_state.issue_no
    st.markdown(
        f"""
        <div class="hn-masthead">
            <span>Vol. I &middot; Issue {issue}</span>
            <span class="right">{today}</span>
        </div>
        <div class="hn-title">HartleyNet</div>
        <div class="hn-subtitle">An apparatus for the detection of synthetic faces.</div>
        <div class="hn-byline">undergraduate thesis &middot; spectral pooling for image forensics</div>
        <div class="hn-ornament">&middot; &middot; &middot; &nbsp;&nbsp; &middot; &middot; &middot; &nbsp;&nbsp; &middot; &middot; &middot;</div>
        """,
        unsafe_allow_html=True,
    )


def render_uploader_section():
    st.markdown(
        "<div class='hn-directive'>— Drop a specimen below —</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop an image here",
        type=ACCEPTED_TYPES,
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.markdown(
            f"<div style='text-align:center;font-family:Newsreader,serif;"
            f"font-style:italic;font-size:14px;color:#777;margin-top:8px;'>"
            f"Accepted: {', '.join(ACCEPTED_TYPES)} &middot; up to "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB</div>",
            unsafe_allow_html=True,
        )
    return uploaded


def render_preview(image: Image.Image):
    st.markdown("<div class='hn-rule thin'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 4, 1])
    with cols[1]:
        st.image(image, width=PREVIEW_WIDTH, caption="SPECIMEN")


def render_verdict(label: str, prob: float, inference_ms: float):
    is_ai = label == LABEL_AI
    confidence = prob if is_ai else 1.0 - prob
    label_class = "ai" if is_ai else "real"
    spaced_label = "S Y N T H E T I C" if is_ai else "A U T H E N T I C"

    st.markdown(
        f"""
        <div class="hn-verdict">
            <p class="hn-verdict-prelude">The image is determined to be</p>
            <div class="hn-verdict-class {label_class}">{spaced_label}</div>
            <div class="hn-verdict-meta">
                <span>
                    <span class="label">p (AI)</span>
                    <span class="value">{prob:.4f}</span>
                </span>
                <span style="text-align:right;">
                    <span class="label">Latency</span>
                    <span class="value">{inference_ms:.1f} ms</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence, text=f"CONFIDENCE  {confidence * 100:.1f}%")


def _thumbnail_data_uri(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_history_sidebar():
    with st.sidebar:
        st.markdown(
            "<div class='hn-issue'>Ledger &middot; this session</div>",
            unsafe_allow_html=True,
        )
        st.header("History")
        history = st.session_state.history
        if not history:
            st.caption("No specimens analysed yet.")
            return

        rows = []
        for entry in reversed(history):
            label_class = "ai" if entry["label"] == LABEL_AI else "real"
            rows.append(
                f"""<div class="hn-history-row">
                    <img class="hn-history-thumb" src="{_thumbnail_data_uri(entry['thumbnail'])}" />
                    <div class="hn-history-info">
                        <div class="hn-history-label {label_class}">{entry['label']}</div>
                        <div class="hn-history-meta">{entry['confidence'] * 100:.1f}% &middot; {entry['timestamp']}</div>
                    </div>
                </div>"""
            )
        st.markdown(
            "<div class='hn-history-list'>" + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )


def render_footer():
    st.markdown(
        "<div class='hn-footer'>"
        "HartleyNet <span class='accent'>&middot;</span> "
        "Undergraduate Thesis <span class='accent'>&middot;</span> "
        "Spectral Pooling for Image Forensics"
        "</div>",
        unsafe_allow_html=True,
    )


# --- Pipeline ---------------------------------------------------------------


def _record_history(uploaded, image: Image.Image, label: str, prob: float):
    if st.session_state.last_file_id == uploaded.file_id:
        return
    st.session_state.last_file_id = uploaded.file_id

    confidence = prob if label == LABEL_AI else 1.0 - prob
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
            f"That image is {len(raw) / (1024 * 1024):.1f} MB — the demo "
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
        st.error(f"Could not read the image — the file may be truncated or corrupt. ({exc})")
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
    inject_style()
    _ensure_session_state()
    render_masthead()
    render_history_sidebar()

    uploaded = render_uploader_section()
    if uploaded is None:
        render_footer()
        return

    image = _load_image(uploaded)
    if image is None:
        render_footer()
        return

    render_preview(image)

    result = _run_inference(image)
    if result is None:
        render_footer()
        return

    label, prob, inference_ms = result
    render_verdict(label, prob, inference_ms)
    _record_history(uploaded, image, label, prob)
    render_footer()


if __name__ == "__main__":
    main()
