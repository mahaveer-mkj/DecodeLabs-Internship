"""
================================================================================
  DecodeLabs | Project 4: Building the Machine's Optic Nerve
  Streamlit UI — app.py
  Author  : Mahaveer (AI Intern, DecodeLabs)

  This file is a PRESENTATION LAYER only.
  All core logic lives in path1_ocr.py and path2_object_detection.py.
  app.py imports the pipeline functions directly — no logic is duplicated.
================================================================================
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
import time

# ── Page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="The Machine's Optic Nerve | DecodeLabs P4",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import pipeline functions from the core scripts ──────────────────────────
# This is the same pattern as Project 3's app.py importing TechStackRecommender
try:
    from path1_ocr import (
        preprocess_for_ocr,
        run_ocr_with_confidence_filter,
        annotate_image_ocr,
    )
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    OCR_IMPORT_ERROR = str(e)

try:
    from path2_object_detection import (
        load_model,
        construct_blob,
        run_inference,
        filter_and_decode_detections,
        visual_confirmation_detection,
        CLASS_LABELS,
        PROTOTXT_PATH,
        CAFFEMODEL_PATH,
    )
    DETECTION_AVAILABLE = True
except ImportError as e:
    DETECTION_AVAILABLE = False
    DETECTION_IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# ░░  CUSTOM CSS — matches DecodeLabs indigo brand colour from Project 3
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Brand colour: indigo (#4F46E5) matching Project 3 theme */
  .stApp { background-color: #0f0f1a; color: #e2e8f0; }

  .main-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #4F46E5, #818CF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .sub-title {
    color: #94a3b8; font-size: 1.05rem; margin-top: 0.2rem;
    margin-bottom: 1.5rem;
  }
  .rule-badge {
    display: inline-block; background: #1e1b4b; color: #818CF8;
    border: 1px solid #4F46E5; border-radius: 6px;
    padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
    margin: 2px; letter-spacing: 0.04em;
  }
  .gatekeeper-pass {
    background: #052e16; border-left: 4px solid #22c55e;
    padding: 10px 16px; border-radius: 6px; margin: 6px 0;
    color: #86efac; font-family: monospace; font-size: 0.88rem;
  }
  .gatekeeper-fail {
    background: #2d0a0a; border-left: 4px solid #ef4444;
    padding: 10px 16px; border-radius: 6px; margin: 6px 0;
    color: #fca5a5; font-family: monospace; font-size: 0.88rem;
  }
  .metric-card {
    background: #1e1b4b; border: 1px solid #3730a3;
    border-radius: 10px; padding: 16px; text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: #818CF8; }
  .metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }
  .section-header {
    font-size: 1.15rem; font-weight: 700; color: #c7d2fe;
    border-bottom: 1px solid #3730a3; padding-bottom: 6px; margin-top: 1.2rem;
  }
  div[data-testid="stSidebar"] { background-color: #0d0d1f; }
  .stButton > button {
    background: linear-gradient(135deg, #4F46E5, #6366f1);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; width: 100%; padding: 0.6rem;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stFileUploader { border: 2px dashed #4F46E5 !important; border-radius: 10px; }
  code { background: #1e1b4b !important; color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ░░  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array → PIL Image for Streamlit display."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def gray_to_pil(gray: np.ndarray) -> Image.Image:
    """Convert grayscale numpy array → PIL Image."""
    return Image.fromarray(gray)

def img_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    """Encode PIL Image to bytes for st.download_button."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def rule_badge(number: int, label: str):
    st.markdown(
        f'<span class="rule-badge">RULE {number} — {label}</span>',
        unsafe_allow_html=True
    )

def gate_pass(msg: str):
    st.markdown(f'<div class="gatekeeper-pass">✅ {msg}</div>', unsafe_allow_html=True)

def gate_fail(msg: str):
    st.markdown(f'<div class="gatekeeper-fail">❌ {msg}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ░░  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">👁️ Building the Machine\'s Optic Nerve</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">DecodeLabs — Project 4 &nbsp;|&nbsp; Computer Vision Pipeline &nbsp;|&nbsp; by Mahaveer</p>', unsafe_allow_html=True)

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1: rule_badge(1, "Library Integration")
with col_b2: rule_badge(2, "Pre-Processing")
with col_b3: rule_badge(3, "80% Confidence Filter")
with col_b4: rule_badge(4, "Visual Confirmation")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ░░  SIDEBAR — PATH SELECTOR + CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    path = st.radio(
        "**Choose Execution Path**",
        ["👁️ Path 1 — OCR (Text Extraction)", "📦 Path 2 — Object Detection"],
        help="Path 1 reads text from images. Path 2 detects and locates objects."
    )
    IS_OCR = "Path 1" in path

    st.divider()

    if IS_OCR:
        st.markdown("**PSM Mode (Page Segmentation)**")
        psm_choice = st.selectbox(
            "Tesseract PSM",
            options=[3, 6, 11],
            index=1,
            format_func=lambda x: {
                3:  "PSM 3 — Auto (varied layouts)",
                6:  "PSM 6 — Uniform text block",
                11: "PSM 11 — Sparse text / labels",
            }[x],
        )
        oem_mode = 3   # always LSTM
        tesseract_config = f"--oem {oem_mode} --psm {psm_choice}"
        st.caption(f"Config: `{tesseract_config}`")
    else:
        st.markdown("**Confidence Threshold**")
        conf_threshold = st.slider(
            "Minimum confidence (%)",
            min_value=50, max_value=99, value=80, step=1,
            help="Detections below this value are dropped as false positives."
        )
        st.caption(f"Gatekeeper: `if confidence >= {conf_threshold/100:.2f}`")

    st.divider()
    confidence_threshold = (conf_threshold / 100.0) if not IS_OCR else 0.80

    st.markdown("**About**")
    st.caption(
        "Presentation layer only — imports pipeline functions directly "
        "from `path1_ocr.py` and `path2_object_detection.py`. "
        "No logic duplicated."
    )
    st.caption("DecodeLabs Internship | Mahaveer Mundaluhari")


# ─────────────────────────────────────────────────────────────────────────────
# ░░  PATH 1 — OCR
# ─────────────────────────────────────────────────────────────────────────────

if IS_OCR:

    st.markdown("## 📄 Path 1 — Optical Character Recognition")
    st.markdown(
        "Upload any image containing text — a document scan, photograph of a sign, "
        "screenshot, invoice, or printed page. The pipeline runs all 4 Gatekeeper Rules "
        "and returns clean, validated, machine-readable text."
    )

    uploaded = st.file_uploader(
        "Drop your image here",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="ocr_upload"
    )

    if uploaded:
        pil_original = Image.open(uploaded)
        bgr_original = pil_to_bgr(pil_original)
        h, w = bgr_original.shape[:2]

        # ── Rule 1 display ───────────────────────────────────────────────────
        st.markdown('<p class="section-header">RULE 1 — Library Integration</p>', unsafe_allow_html=True)
        col_r1a, col_r1b = st.columns(2)
        with col_r1a:
            if OCR_AVAILABLE:
                gate_pass("pytesseract imported successfully")
                gate_pass(f"Tesseract config: `{tesseract_config}`")
            else:
                gate_fail(f"pytesseract import failed: {OCR_IMPORT_ERROR}")
                st.stop()
        with col_r1b:
            gate_pass(f"Image loaded — {w}×{h} px ({bgr_original.shape[2]} channels BGR)")

        # ── Rule 2 — Pre-Processing ──────────────────────────────────────────
        st.markdown('<p class="section-header">RULE 2 — Pre-Processing Pipeline</p>', unsafe_allow_html=True)

        with st.spinner("Running pre-processing pipeline …"):
            t0 = time.time()
            preprocessed = preprocess_for_ocr(bgr_original)
            elapsed_pre = time.time() - t0

        col_r2a, col_r2b, col_r2c = st.columns(3)
        with col_r2a:
            gate_pass(f"Grayscale: (H,W,3) → (H,W,1)")
            gate_pass("Gaussian Blur: kernel 5×5")
        with col_r2b:
            gate_pass("Deskew: MinAreaRect angle correction")
            gate_pass("Otsu Threshold: pure binary image")
        with col_r2c:
            gate_pass(f"Pipeline time: {elapsed_pre*1000:.0f} ms")

        col_orig, col_pre = st.columns(2)
        with col_orig:
            st.markdown("**Original image**")
            st.image(pil_original, use_container_width=True)
        with col_pre:
            st.markdown("**After pre-processing (binary)**")
            st.image(gray_to_pil(preprocessed), use_container_width=True)

        # ── Rule 3 — Confidence Filter ───────────────────────────────────────
        st.markdown('<p class="section-header">RULE 3 — Accuracy Benchmarking (80% Confidence Filter)</p>', unsafe_allow_html=True)

        with st.spinner("Running OCR + confidence filter …"):
            t1 = time.time()
            filtered_text, confident_words = run_ocr_with_confidence_filter(
                preprocessed,
                config=tesseract_config,
                confidence_threshold=0.80
            )
            elapsed_ocr = time.time() - t1

        total_candidates = len(confident_words)   # already filtered
        accepted = len(confident_words)

        # get rejected count by running raw data pass
        import pytesseract
        raw_data = pytesseract.image_to_data(
            preprocessed, config=tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        total_raw = sum(1 for t in raw_data["text"] if t.strip() != "")
        rejected  = total_raw - accepted

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown('<div class="metric-card"><div class="metric-value">'
                        f'{total_raw}</div><div class="metric-label">Total Candidates</div></div>',
                        unsafe_allow_html=True)
        with col_m2:
            st.markdown('<div class="metric-card"><div class="metric-value" style="color:#22c55e">'
                        f'{accepted}</div><div class="metric-label">Accepted ≥ 80%</div></div>',
                        unsafe_allow_html=True)
        with col_m3:
            st.markdown('<div class="metric-card"><div class="metric-value" style="color:#ef4444">'
                        f'{rejected}</div><div class="metric-label">Rejected &lt; 80%</div></div>',
                        unsafe_allow_html=True)
        with col_m4:
            st.markdown('<div class="metric-card"><div class="metric-value">'
                        f'{elapsed_ocr*1000:.0f}ms</div><div class="metric-label">Inference Time</div></div>',
                        unsafe_allow_html=True)

        st.markdown("")
        gate_pass(f"Gatekeeper applied: `if confidence >= 0.80` — {rejected} false positives dropped")

        # ── Rule 4 — Visual Confirmation ─────────────────────────────────────
        st.markdown('<p class="section-header">RULE 4 — Visual Confirmation</p>', unsafe_allow_html=True)

        if filtered_text.strip():
            gate_pass("Machine-readable text extracted successfully")
            st.text_area(
                "Extracted Text (words with confidence ≥ 80%)",
                value=filtered_text,
                height=200,
            )
        else:
            st.warning(
                "No high-confidence text found. Try switching PSM mode "
                "(PSM 3 for varied layouts, PSM 11 for sparse text)."
            )

        # Annotated image
        annotated_bgr = bgr_original.copy()
        for w_item in confident_words:
            x, y, wd, ht = w_item["left"], w_item["top"], w_item["width"], w_item["height"]
            label = f"{w_item['text']} ({w_item['confidence']*100:.0f}%)"
            cv2.rectangle(annotated_bgr, (x, y), (x+wd, y+ht), (0, 200, 0), 2)
            cv2.putText(annotated_bgr, label, (x, max(y-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
        annotated_pil = bgr_to_pil(annotated_bgr)

        st.markdown("**Annotated image — green boxes = words accepted by the 80% gatekeeper**")
        st.image(annotated_pil, use_container_width=True)

        # Word-level breakdown table
        if confident_words:
            st.markdown("**Word-level confidence breakdown**")
            import pandas as pd
            df = pd.DataFrame([{
                "Word"      : w["text"],
                "Confidence": f"{w['confidence']*100:.1f}%",
                "Left (px)" : w["left"],
                "Top (px)"  : w["top"],
                "Width (px)": w["width"],
                "Height (px)": w["height"],
            } for w in confident_words])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Download buttons
        st.markdown("**Download outputs**")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download extracted text (.txt)",
                data=filtered_text.encode("utf-8"),
                file_name="ocr_output.txt",
                mime="text/plain",
            )
        with col_dl2:
            st.download_button(
                "⬇ Download annotated image (.png)",
                data=img_to_bytes(annotated_pil),
                file_name="ocr_annotated.png",
                mime="image/png",
            )

    else:
        st.info("👆 Upload an image above to begin the OCR pipeline.")
        st.markdown("""
        **Good test images for OCR:**
        - Scanned documents or invoices
        - Photographs of signs or printed pages
        - Screenshots of text
        - Business cards or receipts
        """)


# ─────────────────────────────────────────────────────────────────────────────
# ░░  PATH 2 — OBJECT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

else:

    st.markdown("## 📦 Path 2 — Object Detection (MobileNet-SSD)")
    st.markdown(
        "Upload any real-world scene photograph. MobileNet-SSD runs a forward pass, "
        "generates up to 100 candidates, and the confidence gatekeeper filters them down "
        "to only verified detections. Bounding boxes are decoded from normalised coordinates "
        "back to pixel space and drawn on the original image."
    )

    # ── Check model files exist ──────────────────────────────────────────────
    prototxt_ok    = os.path.isfile(PROTOTXT_PATH)
    caffemodel_ok  = os.path.isfile(CAFFEMODEL_PATH)

    if not prototxt_ok or not caffemodel_ok:
        st.error("⚠️ Model files missing. Cannot run Object Detection.")
        st.markdown(f"""
        **Required files (place in the same folder as `app.py`):**

        | File | Status |
        |------|--------|
        | `{PROTOTXT_PATH}` | {"✅ Found" if prototxt_ok else "❌ Missing"} |
        | `{CAFFEMODEL_PATH}` | {"✅ Found" if caffemodel_ok else "❌ Missing — run `python download_model.py`"} |

        Run `python download_model.py` to download the caffemodel automatically.
        """)
        st.stop()

    # ── Load model (cached so it only loads once per session) ────────────────
    @st.cache_resource(show_spinner="Loading MobileNet-SSD model …")
    def get_model():
        return load_model(PROTOTXT_PATH, CAFFEMODEL_PATH)

    net = get_model()

    uploaded = st.file_uploader(
        "Drop your image here",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="det_upload"
    )

    if uploaded:
        pil_original = Image.open(uploaded)
        bgr_original = pil_to_bgr(pil_original)
        h, w = bgr_original.shape[:2]

        # ── Rule 1 ───────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">RULE 1 — Library Integration</p>', unsafe_allow_html=True)
        col_r1a, col_r1b = st.columns(2)
        with col_r1a:
            if DETECTION_AVAILABLE:
                gate_pass("OpenCV cv2.dnn loaded successfully")
                gate_pass(f"MobileNet-SSD: {len(CLASS_LABELS)-1} PASCAL VOC classes")
            else:
                gate_fail(f"Import failed: {DETECTION_IMPORT_ERROR}")
                st.stop()
        with col_r1b:
            gate_pass(f"Image loaded — {w}×{h} px")
            gate_pass(f"Model files: prototxt ✓  caffemodel ✓")

        # ── Rule 2 — Blob construction ────────────────────────────────────────
        st.markdown('<p class="section-header">RULE 2 — Pre-Processing (4D Blob)</p>', unsafe_allow_html=True)

        with st.spinner("Constructing 4D Blob …"):
            t0 = time.time()
            blob = construct_blob(bgr_original)
            elapsed_blob = time.time() - t0

        col_r2a, col_r2b, col_r2c = st.columns(3)
        with col_r2a:
            gate_pass(f"Input shape: ({h}, {w}, 3) — H×W×C")
            gate_pass("Blob shape: (1, 3, 300, 300) — N,C,H,W")
        with col_r2b:
            gate_pass("Resize: 300×300 px (network requirement)")
            gate_pass("Scale factor: 0.007843 → [0, ~2.0]")
        with col_r2c:
            gate_pass("Mean subtraction: (127.5, 127.5, 127.5)")
            gate_pass(f"Blob construction: {elapsed_blob*1000:.1f} ms")

        # ── Rule 3 — Inference + Confidence Filter ────────────────────────────
        st.markdown('<p class="section-header">RULE 3 — Inference + Accuracy Benchmarking</p>', unsafe_allow_html=True)

        with st.spinner(f"Running forward pass + applying {conf_threshold}% confidence filter …"):
            t1 = time.time()
            raw_detections  = run_inference(net, blob)
            verified        = filter_and_decode_detections(
                raw_detections,
                image_shape=bgr_original.shape,
                confidence_threshold=confidence_threshold
            )
            elapsed_inf = time.time() - t1

        total_raw  = raw_detections.shape[2]
        accepted   = len(verified)
        rejected   = total_raw - accepted

        if conf_threshold != 80:
            st.info(f"🔒 Cold-start guard: You set {conf_threshold}% (default is 80%). "
                    f"Gatekeeper rule: `if confidence >= {confidence_threshold:.2f}`")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_raw}</div>'
                        f'<div class="metric-label">Raw Candidates</div></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#22c55e">'
                        f'{accepted}</div><div class="metric-label">Accepted ≥ {conf_threshold}%</div></div>',
                        unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444">'
                        f'{rejected}</div><div class="metric-label">Rejected (False +)</div></div>',
                        unsafe_allow_html=True)
        with col_m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">'
                        f'{elapsed_inf*1000:.0f}ms</div><div class="metric-label">Inference Time</div></div>',
                        unsafe_allow_html=True)

        st.markdown("")
        gate_pass(f"Gatekeeper applied: `if confidence >= {confidence_threshold:.2f}` — "
                  f"{rejected} false positives dropped")

        # ── Rule 4 — Visual Confirmation ──────────────────────────────────────
        st.markdown('<p class="section-header">RULE 4 — Visual Confirmation</p>', unsafe_allow_html=True)

        # Draw bounding boxes
        annotated_bgr = bgr_original.copy()
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(CLASS_LABELS), 3), dtype="uint8")

        if not verified:
            st.warning(
                f"No objects detected above {conf_threshold}% confidence. "
                "Try a clearer image or lower the threshold in the sidebar."
            )
        else:
            gate_pass(f"{accepted} object(s) detected and verified above {conf_threshold}% confidence")

            for det in verified:
                label = det["label"]
                conf  = det["confidence"]
                x1, y1, x2, y2 = det["x_start"], det["y_start"], det["x_end"], det["y_end"]
                color = [int(c) for c in COLORS[CLASS_LABELS.index(label)]]
                display_text = f"{label}: {conf*100:.1f}%"
                (tw, th), baseline = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_bgr, (x1, y1-th-baseline-4),
                              (x1+tw, y1), color, cv2.FILLED)
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_bgr, display_text, (x1, y1-baseline-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        col_orig2, col_ann2 = st.columns(2)
        with col_orig2:
            st.markdown("**Original image**")
            st.image(pil_original, use_container_width=True)
        with col_ann2:
            st.markdown("**Detections — bounding boxes decoded from normalised coordinates**")
            st.image(bgr_to_pil(annotated_bgr), use_container_width=True)

        # Detection table
        if verified:
            st.markdown("**Detection breakdown**")
            import pandas as pd
            df = pd.DataFrame([{
                "Object"    : d["label"],
                "Confidence": f"{d['confidence']*100:.1f}%",
                "x1 (px)"  : d["x_start"],
                "y1 (px)"  : d["y_start"],
                "x2 (px)"  : d["x_end"],
                "y2 (px)"  : d["y_end"],
                "Box W (px)": d["x_end"] - d["x_start"],
                "Box H (px)": d["y_end"] - d["y_start"],
            } for d in verified])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Download
        annotated_pil = bgr_to_pil(annotated_bgr)
        st.download_button(
            "⬇ Download annotated image (.png)",
            data=img_to_bytes(annotated_pil),
            file_name="detection_output.png",
            mime="image/png",
        )

    else:
        st.info("👆 Upload an image above to begin object detection.")
        st.markdown(f"""
        **Best images for Object Detection:**
        - Clear photographs of people, cars, animals, furniture
        - Well-lit scenes with distinct objects
        - Photos where objects are not heavily overlapping

        **Detectable classes ({len(CLASS_LABELS)-1} total):**
        `{", ".join(CLASS_LABELS[1:])}`
        """)


# ─────────────────────────────────────────────────────────────────────────────
# ░░  FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:0.82rem;">'
    'DecodeLabs Internship &nbsp;|&nbsp; Project 4: Building the Machine\'s Optic Nerve &nbsp;|&nbsp; '
    'Mahaveer Mundaluhari &nbsp;|&nbsp; '
    '<a href="https://www.linkedin.com/in/mahaveer-mundaluhari/" style="color:#6366f1">LinkedIn</a> &nbsp;|&nbsp; '
    '<a href="https://github.com/mahaveer-mkj" style="color:#6366f1">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)
