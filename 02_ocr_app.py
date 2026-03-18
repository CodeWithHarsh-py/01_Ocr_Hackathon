"""
🔍 OCR App - Image to Text
--------------------------
Install karo:
    pip install streamlit pytesseract opencv-python pillow numpy

Run karo:
    streamlit run ocr_app.py
"""

import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import time
import os

# ── FIX: Tesseract path Windows pe manually set karo ─────────────────────────
# Agar tesseract alag jagah install hai toh yeh path change karo
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.environ.get("USERNAME", "")),
]

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR - Image to Text",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700;800&display=swap');

    html, body, .stApp { background-color: #080c10 !important; color: #d4e8f0 !important; }
    h1 { font-family: 'Syne', sans-serif !important; color: #00e5ff !important; }
    h3 { color: #a5b4fc !important; }

    .stButton > button {
        background: linear-gradient(135deg, #00e5ff, #7c83fd) !important;
        color: #080c10 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        width: 100% !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    .stButton > button:disabled { background: #1e2a38 !important; color: #4a6070 !important; }

    .result-box {
        background: #0f1419;
        border: 1px solid #1e2a38;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Space Mono', monospace;
        font-size: 0.88rem;
        line-height: 1.9;
        white-space: pre-wrap;
        word-break: break-word;
        color: #d4e8f0;
        max-height: 480px;
        overflow-y: auto;
    }

    .stat-box {
        background: #0f1419;
        border: 1px solid #1e2a38;
        border-radius: 10px;
        padding: 14px 8px;
        text-align: center;
        margin-bottom: 16px;
    }
    .stat-val { font-size: 1.5rem; font-weight: 800; font-family: 'Space Mono', monospace; }
    .stat-lbl { font-size: 0.65rem; color: #4a6070; letter-spacing: 1.5px; font-family: 'Space Mono', monospace; margin-top: 2px; }

    .tag {
        display: inline-block;
        background: rgba(0,229,255,0.08);
        border: 1px solid rgba(0,229,255,0.25);
        color: #00e5ff;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 1px;
        margin-right: 6px;
    }

    .empty-box {
        min-height: 320px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 14px;
        background: #0f1419;
        border: 1px dashed #1e2a38;
        border-radius: 12px;
    }

    .error-box {
        background: rgba(255,80,80,0.08);
        border: 1px solid rgba(255,80,80,0.3);
        border-radius: 10px;
        padding: 16px 20px;
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        color: #ff6b6b;
    }

    footer, #MainMenu, header { visibility: hidden; }
    .stSelectbox label, .stFileUploader label { color: #a5b4fc !important; font-size: 0.82rem !important; }
    div[data-baseweb="select"] { background: #0f1419 !important; }
    div[data-baseweb="select"] * { background: #0f1419 !important; color: #d4e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Check Tesseract ───────────────────────────────────────────────────────────
def check_tesseract():
    try:
        ver = pytesseract.get_tesseract_version()
        return True, str(ver)
    except Exception as e:
        return False, str(e)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(pil_img, mode):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    if mode == "auto":
        # Upscale small images
        if w < 1000:
            scale = 1500 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        # Adaptive threshold
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

    elif mode == "printed":
        if w < 1200:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif mode == "handwritten":
        if w < 1000:
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
        )

    elif mode == "screenshot":
        if w < 800:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(gray)


def run_ocr(pil_img, lang, psm, mode):
    processed = preprocess(pil_img, mode)
    config = f"--psm {psm} --oem 3"

    data = pytesseract.image_to_data(
        processed, lang=lang, config=config,
        output_type=pytesseract.Output.DICT
    )
    full_text = pytesseract.image_to_string(processed, lang=lang, config=config)

    confs = [int(c) for c in data["conf"] if int(c) > 0]
    avg_conf = int(sum(confs) / len(confs)) if confs else 0

    return full_text.strip(), avg_conf


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 OCR — Image to Text")
st.markdown(
    '<span class="tag">100% FREE</span>'
    '<span class="tag">LOCAL</span>'
    '<span class="tag">TESSERACT v5</span>'
    '<span class="tag">OPENCV</span>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Tesseract Check ───────────────────────────────────────────────────────────
tess_ok, tess_info = check_tesseract()
if not tess_ok:
    st.markdown(f"""
    <div class="error-box">
    ⚠️ <b>Tesseract nahi mila!</b><br><br>
    Windows pe install karo:<br>
    👉 <a href="https://github.com/UB-Mannheim/tesseract/wiki" style="color:#00e5ff">
    https://github.com/UB-Mannheim/tesseract/wiki</a><br><br>
    Install ke baad default path hoga:<br>
    <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code><br><br>
    Ya agar alag path pe install kiya toh <b>ocr_app.py</b> mein
    <code>TESSERACT_PATHS</code> list mein apna path add karo.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
else:
    st.success(f"✅ Tesseract ready — v{tess_info}")

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with col1:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader(
        "PNG · JPG · WEBP · BMP · TIFF",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"],
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        # FIX: use_container_width deprecated → width='stretch'
        st.image(pil_img,
                 caption=f"{uploaded.name}  ·  {pil_img.size[0]}×{pil_img.size[1]}px",
                 width="stretch")

    st.markdown("### ⚙️ Settings")
    c1, c2 = st.columns(2)

    with c1:
        img_type = st.selectbox("Image Type", [
            "auto", "printed", "handwritten", "screenshot"
        ], format_func=lambda x: {
            "auto":        "🤖 Auto (Best)",
            "printed":     "🖨️ Printed",
            "handwritten": "✍️ Handwritten",
            "screenshot":  "🖥️ Screenshot",
        }[x])

    with c2:
        psm = st.selectbox("OCR Mode", [3, 6, 4, 7, 11],
        format_func=lambda x: {
            3:  "Full Page",
            6:  "Text Block",
            4:  "Single Column",
            7:  "Single Line",
            11: "Sparse Text",
        }[x])

    lang = st.selectbox("Language", ["eng", "hin", "eng+hin"],
    format_func=lambda x: {
        "eng":     "🇬🇧 English",
        "hin":     "🇮🇳 Hindi",
        "eng+hin": "🇮🇳 Hindi + English",
    }[x])

    extract = st.button("⚡ EXTRACT TEXT", disabled=(uploaded is None))

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col2:
    st.markdown("### 📄 Extracted Text")

    if extract and uploaded:
        with st.spinner("🔍 Reading image..."):
            t0 = time.time()
            try:
                text, conf = run_ocr(pil_img, lang, psm, img_type)
                elapsed = round(time.time() - t0, 2)

                words = len(text.split()) if text else 0
                lines = len([l for l in text.split("\n") if l.strip()]) if text else 0
                chars = len(text.replace(" ", "").replace("\n", "")) if text else 0
                conf_color = "#00ff9d" if conf > 80 else "#00e5ff" if conf > 50 else "#ff6b35"

                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{conf_color}">{conf}%</div><div class="stat-lbl">CONFIDENCE</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00e5ff">{words}</div><div class="stat-lbl">WORDS</div></div>', unsafe_allow_html=True)
                s3.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00e5ff">{lines}</div><div class="stat-lbl">LINES</div></div>', unsafe_allow_html=True)
                s4.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00e5ff">{elapsed}s</div><div class="stat-lbl">TIME</div></div>', unsafe_allow_html=True)

                if text:
                    st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
                    st.download_button(
                        "⬇️ Download .txt",
                        data=text,
                        file_name=uploaded.name.rsplit(".", 1)[0] + "_ocr.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("⚠️ Koi text nahi mila. Alag Image Type ya OCR Mode try karo.")

            except pytesseract.TesseractNotFoundError:
                st.markdown("""
                <div class="error-box">
                ⚠️ <b>Tesseract nahi mila!</b> Path check karo.<br>
                ocr_app.py mein <code>TESSERACT_PATHS</code> mein apna path add karo.
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    else:
        st.markdown("""
        <div class="empty-box">
            <div style="font-size:52px;opacity:0.25">📄</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.78rem;
                        color:#4a6070;letter-spacing:2px">
                IMAGE UPLOAD KARO AUR EXTRACT DABAO
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-family:monospace;font-size:0.72rem;color:#4a6070">'
    'Powered by <b style="color:#00e5ff">Tesseract OCR v5</b> + '
    '<b style="color:#00e5ff">OpenCV</b> · 100% Free · Local · Zero data sent anywhere'
    '</div>',
    unsafe_allow_html=True
)
