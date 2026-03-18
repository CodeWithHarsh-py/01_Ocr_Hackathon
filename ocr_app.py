"""
🔍 Best OCR App — PaddleOCR (Fixed)
-------------------------------------
Install:
    pip install paddlepaddle paddleocr streamlit opencv-python pillow numpy

Run:
    streamlit run ocr_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

# ── CRITICAL FIX: Disable PDX connectivity check (causes reinit error) ───────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR — PaddleOCR",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

    html, body, .stApp         { background-color: #07090f !important; color: #d4e8f0 !important; }
    h1                         { font-family: 'Syne', sans-serif !important; color: #00e5ff !important; letter-spacing:-1px; }
    h3                         { color: #a5b4fc !important; font-family: 'Syne', sans-serif !important; }

    .stButton > button {
        background: linear-gradient(135deg, #00e5ff, #7c83fd) !important;
        color: #07090f !important; font-weight: 800 !important;
        font-size: 1rem !important; border: none !important;
        border-radius: 10px !important; padding: 0.7rem 2rem !important;
        width: 100% !important; letter-spacing: 0.5px !important;
    }
    .stButton > button:hover    { opacity: .82 !important; }
    .stButton > button:disabled { background: #1a2030 !important; color: #3a4a5a !important; }

    .result-box {
        background: #0d1117; border: 1px solid #1e2a3a;
        border-radius: 12px; padding: 22px;
        font-family: 'Space Mono', monospace; font-size: .9rem;
        line-height: 2.0; white-space: pre-wrap; word-break: break-word;
        color: #d4e8f0; max-height: 500px; overflow-y: auto;
    }

    .stat-box {
        background: #0d1117; border: 1px solid #1e2a3a;
        border-radius: 10px; padding: 14px 6px;
        text-align: center; margin-bottom: 14px;
    }
    .stat-val { font-size: 1.45rem; font-weight: 800; font-family: 'Space Mono', monospace; }
    .stat-lbl { font-size: .62rem; color: #3a5060; letter-spacing: 2px;
                font-family: 'Space Mono', monospace; margin-top: 3px; text-transform: uppercase; }

    .tag {
        display: inline-block; background: rgba(0,229,255,.07);
        border: 1px solid rgba(0,229,255,.22); color: #00e5ff;
        border-radius: 4px; padding: 2px 10px; font-size: .7rem;
        font-family: 'Space Mono', monospace; letter-spacing: 1px; margin-right: 6px;
    }
    .tag-green {
        display: inline-block; background: rgba(0,255,150,.07);
        border: 1px solid rgba(0,255,150,.22); color: #00ff96;
        border-radius: 4px; padding: 2px 10px; font-size: .7rem;
        font-family: 'Space Mono', monospace; letter-spacing: 1px; margin-right: 6px;
    }

    .log-box {
        background: #020408; border: 1px solid #1e2a3a;
        border-radius: 10px; padding: 14px 16px;
        font-family: 'Space Mono', monospace; font-size: .75rem;
        line-height: 1.8; color: #4a9060; max-height: 200px;
        overflow-y: auto; margin-top: 10px;
    }

    .log-line-ok   { color: #00ff96; }
    .log-line-info { color: #00e5ff; }
    .log-line-warn { color: #ffaa00; }
    .log-line-err  { color: #ff6b35; }

    .empty-box {
        min-height: 340px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 14px;
        background: #0d1117; border: 1px dashed #1e2a3a; border-radius: 12px;
    }

    .word-chip {
        display: inline-block;
        font-family: 'Space Mono', monospace; font-size: .75rem;
        padding: 3px 9px; border-radius: 5px; border: 1px solid #1e2a3a;
        margin: 3px;
    }

    footer, #MainMenu, header { visibility: hidden; }
    .stSelectbox label, .stFileUploader label,
    .stRadio label { color: #a5b4fc !important; font-size: .82rem !important; }
    div[data-baseweb="select"]   { background: #0d1117 !important; }
    div[data-baseweb="select"] * { background: #0d1117 !important; color: #d4e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── PaddleOCR singleton — only init ONCE, never reinit ───────────────────────
# Using st.session_state instead of cache_resource to avoid PDX reinit error
def get_ocr_engine(lang: str):
    key = f"paddle_ocr_{lang}"
    if key not in st.session_state:
        from paddleocr import PaddleOCR
        st.session_state[key] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,
            show_log=False,
        )
    return st.session_state[key]


# ── Smart preprocessing — handles ANY image ───────────────────────────────────
def preprocess_any(pil_img: Image.Image) -> list:
    """
    Returns multiple preprocessed versions.
    PaddleOCR tries all and picks best result.
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    versions = []

    # 1. Original (upscaled if small)
    if w < 1000:
        scale = 1500 / w
        img_up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        img_up = img.copy()
    versions.append(("original", img_up))

    # 2. Contrast enhanced (great for dark bg / screenshots)
    lab = cv2.cvtColor(img_up, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    versions.append(("contrast_enhanced", enhanced))

    # 3. Denoised
    denoised = cv2.fastNlMeansDenoisingColored(img_up, None, 6, 6, 7, 21)
    versions.append(("denoised", denoised))

    # 4. Grayscale → RGB (for printed/scanned text)
    gray = cv2.cvtColor(img_up, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    versions.append(("binarized", bw_rgb))

    return versions


# ── Run OCR on all versions, return best result ───────────────────────────────
def run_ocr(pil_img: Image.Image, lang: str, log_placeholder):
    engine = get_ocr_engine(lang)
    versions = preprocess_any(pil_img)

    logs = []
    best_text = ""
    best_conf = 0.0
    best_lines = []

    def push_log(msg, kind="info"):
        logs.append((msg, kind))
        html = '<div class="log-box">'
        for m, k in logs:
            css = {"ok": "log-line-ok", "info": "log-line-info",
                   "warn": "log-line-warn", "err": "log-line-err"}.get(k, "log-line-info")
            html += f'<div class="log-line-{k}">▸ {m}</div>'
        html += "</div>"
        log_placeholder.markdown(html, unsafe_allow_html=True)

    push_log("PaddleOCR engine ready ✓", "ok")
    push_log(f"Image size: {pil_img.size[0]}×{pil_img.size[1]}px", "info")
    push_log(f"Trying {len(versions)} preprocessing modes...", "info")

    for name, img_arr in versions:
        push_log(f"Processing: {name}...", "info")
        try:
            result = engine.ocr(img_arr, cls=True)
            if not result or not result[0]:
                push_log(f"  {name}: no text found", "warn")
                continue

            lines = []
            confs = []
            for line in result[0]:
                text, conf = line[1]
                lines.append((text, float(conf)))
                confs.append(float(conf))

            avg_conf = sum(confs) / len(confs) if confs else 0
            push_log(f"  {name}: {len(lines)} lines, avg conf {avg_conf*100:.1f}%", "ok")

            if avg_conf > best_conf:
                best_conf = avg_conf
                best_lines = lines
                best_text = "\n".join(t for t, _ in lines)
                push_log(f"  ✓ New best result from: {name}", "ok")

        except Exception as e:
            push_log(f"  {name}: error — {e}", "err")

    push_log(f"Done! Best confidence: {best_conf*100:.1f}%", "ok")
    return best_text.strip(), int(best_conf * 100), best_lines


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 OCR — Image to Text")
st.markdown(
    '<span class="tag-green">PADDLEOCR</span>'
    '<span class="tag">100% FREE</span>'
    '<span class="tag">LOCAL</span>'
    '<span class="tag">DEEP LEARNING</span>'
    '<span class="tag">ANY IMAGE</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with col1:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Koi bhi image — screenshot, photo, scan, dark bg sab chalega",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"],
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img,
                 caption=f"{uploaded.name}  ·  {pil_img.size[0]}×{pil_img.size[1]}px",
                 width="stretch")

    st.markdown("### ⚙️ Settings")

    lang = st.selectbox(
        "Language",
        ["en", "ch", "hi"],
        format_func=lambda x: {
            "en": "🇬🇧 English",
            "ch": "🇨🇳 Chinese",
            "hi": "🇮🇳 Hindi",
        }[x],
    )

    show_conf = st.toggle("Per-line confidence dikhao", value=False)
    extract   = st.button("⚡ EXTRACT TEXT", disabled=(uploaded is None))

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col2:
    st.markdown("### 📄 Extracted Text")

    if extract and uploaded:
        log_placeholder = st.empty()

        with st.spinner(""):
            t0 = time.time()
            try:
                text, conf, lines = run_ocr(pil_img, lang, log_placeholder)
                elapsed = round(time.time() - t0, 2)

                words     = len(text.split())                                       if text else 0
                num_lines = len([l for l in text.split("\n") if l.strip()])         if text else 0
                chars     = len(text.replace(" ", "").replace("\n", ""))            if text else 0
                conf_color = (
                    "#00ff96" if conf > 85 else
                    "#00e5ff" if conf > 60 else
                    "#ffaa00" if conf > 40 else
                    "#ff6b35"
                )

                st.markdown("---")
                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{conf_color}">{conf}%</div><div class="stat-lbl">Confidence</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00e5ff">{words}</div><div class="stat-lbl">Words</div></div>', unsafe_allow_html=True)
                s3.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00e5ff">{num_lines}</div><div class="stat-lbl">Lines</div></div>', unsafe_allow_html=True)
                s4.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#00e5ff">{elapsed}s</div><div class="stat-lbl">Time</div></div>', unsafe_allow_html=True)

                if text:
                    st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)

                    if show_conf and lines:
                        st.markdown("**Per-line confidence:**")
                        chips = ""
                        for t, c in lines:
                            pct = int(c * 100)
                            color = (
                                "#00ff96" if pct > 85 else
                                "#00e5ff" if pct > 60 else
                                "#ffaa00" if pct > 40 else
                                "#ff6b35"
                            )
                            chips += f'<span class="word-chip" style="color:{color};border-color:{color}44">{t} <small style="opacity:.55">{pct}%</small></span>'
                        st.markdown(chips, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        "⬇️ Download as .txt",
                        data=text,
                        file_name=uploaded.name.rsplit(".", 1)[0] + "_ocr.txt",
                        mime="text/plain",
                    )
                else:
                    st.warning("⚠️ Koi text nahi mila. Image clear hai? Ya doosri language try karo.")

            except ImportError:
                st.error("❌ PaddleOCR install nahi hai!\n\n```\npip install paddlepaddle paddleocr\n```")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    else:
        st.markdown("""
        <div class="empty-box">
            <div style="font-size:54px;opacity:.2">📄</div>
            <div style="font-family:'Space Mono',monospace;font-size:.78rem;
                        color:#3a5060;letter-spacing:2px">
                IMAGE UPLOAD KARO AUR EXTRACT DABAO
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-family:monospace;font-size:.7rem;color:#3a5060">'
    'Powered by <b style="color:#00ff96">PaddleOCR</b> + '
    '<b style="color:#00e5ff">OpenCV</b> · '
    '100% Free · Runs Locally · No API · No Cost Ever'
    '</div>',
    unsafe_allow_html=True,
)
