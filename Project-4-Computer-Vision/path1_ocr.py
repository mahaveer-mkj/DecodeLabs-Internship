"""
================================================================================
  DecodeLabs Internship | Project 4: Building the Machine's Optic Nerve
  PATH 1: Optical Character Recognition (OCR)
  Author  : Mahaveer (AI Intern, DecodeLabs)
  Engine  : Google Tesseract via pytesseract
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 1 — LIBRARY INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import sys
import traceback

# ── Point pytesseract to the Tesseract binary ────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Force immediate version check (caches a SUCCESS result) ──────────────────
# Also add the Tesseract folder to PATH so the subprocess finds its own DLLs
os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + os.pathsep + os.environ.get('PATH', '')
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract {version} is ready.")
except pytesseract.TesseractNotFoundError:
    import sys
    print("❌ Tesseract not found at the expected location. Exiting.")
    sys.exit(1)

# ── PSM (Page Segmentation Mode) selector ────────────────────────────────────
PSM_MODE = 6
TESSERACT_CONFIG = f"--oem 3 --psm {PSM_MODE}"

print("=" * 70)
print("  DecodeLabs | Project 4 | PATH 1: Optical Character Recognition")
print("=" * 70)
print(f"[✓] RULE 1 — Library Integration : pytesseract loaded successfully.")
print(f"            Tesseract config      : {TESSERACT_CONFIG}")


# ─────────────────────────────────────────────────────────────────────────────
# ░░  HELPER — IMAGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and validate it exists.
    Returns a BGR numpy array (OpenCV's native format).
    """
    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Image not found at: {image_path}")
        print("        Please update IMAGE_PATH at the bottom of this script.")
        sys.exit(1)

    image = cv2.imread(image_path)

    if image is None:
        print(f"\n[ERROR] OpenCV could not decode the file: {image_path}")
        sys.exit(1)

    h, w, c = image.shape
    print(f"\n[✓] Image loaded  : {image_path}")
    print(f"    Dimensions     : {w}px wide × {h}px tall × {c} channels (BGR)")
    return image


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 2 — PRE-PROCESSING INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Transforms a raw colour image into a clean binary (black/white) image
    that Tesseract can read with maximum accuracy.

    Steps
    -----
    1. Grayscale conversion  — collapse RGB depth into 1D intensity values
    2. Gaussian Blur         — suppress high-frequency noise
    3. Deskew                — rotate tilted text to horizontal baseline
    4. Otsu Thresholding     — force every pixel to pure black OR pure white
    """

    print("\n[►] RULE 2 — Pre-Processing Pipeline starting …")

    # ── STEP 2a: GRAYSCALE CONVERSION ────────────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"    [2a] Grayscale : shape changed from {image_bgr.shape} → {gray.shape}")

    # ── STEP 2b: GAUSSIAN BLUR ───────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    print(f"    [2b] Gaussian Blur applied  : kernel 5×5")

    # ── STEP 2c: DESKEWING ───────────────────────────────────────────────────
    blurred_for_skew = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh_for_skew = cv2.threshold(
        blurred_for_skew, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = np.column_stack(np.where(thresh_for_skew > 0))

    if coords.size == 0:
        print(f"    [2c] Deskew : no foreground pixels found — skipping rotation")
        deskewed = blurred
    else:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = blurred.shape[:2]
        centre = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
        deskewed = cv2.warpAffine(
            blurred, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        print(f"    [2c] Deskew : corrected {angle:.2f}° skew")

    # ── STEP 2d: OTSU ADAPTIVE THRESHOLDING ──────────────────────────────────
    _, binary = cv2.threshold(
        deskewed, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"    [2d] Otsu Thresholding : image is now pure black-and-white")
    print(f"[✓] RULE 2 — Pre-Processing complete.  Output shape: {binary.shape}")

    return binary


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 3 — ACCURACY BENCHMARKING (80% CONFIDENCE FILTER)
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr_with_confidence_filter(
    binary_image: np.ndarray,
    config: str,
    confidence_threshold: float = 0.80
) -> tuple:
    """
    Runs Tesseract in data mode (returns word-level bounding boxes + confidence)
    then filters out every word whose confidence is below the threshold.

    Parameters
    ----------
    binary_image         : pre-processed binary numpy array
    config               : Tesseract config string (PSM + OEM flags)
    confidence_threshold : minimum accepted confidence (default 0.80 = 80%)

    Returns
    -------
    filtered_text        : clean string of high-confidence words only
    confident_words      : list of dicts with word metadata
    """

    print(f"\n[►] RULE 3 — Accuracy Benchmarking  (threshold ≥ {confidence_threshold*100:.0f}%) …")

    # ── CRITICAL FIX: Clear pytesseract's internal cache ─────────────────────
    # pytesseract remembers a failed version check; this forces a fresh lookup
    # so the binary at tesseract_cmd is re‑tested every time.
    pytesseract.pytesseract._tesseract_version = None
    pytesseract.pytesseract._tesseract_version_date = None

    try:
        raw_data = pytesseract.image_to_data(
            binary_image,
            config=config,
            output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        msg = (
            "Tesseract OCR engine not found.\n"
            f"Expected at: {pytesseract.pytesseract.tesseract_cmd}\n"
            "Please ensure Tesseract is installed and the path is correct.\n"
            "On Windows, you may also need the Visual C++ Redistributable."
        )
        print(f"\n[ERROR] {msg}")
        traceback.print_exc()
        # Raise a RuntimeError so the caller (Streamlit or CLI) can handle it
        raise RuntimeError(msg)

    total_words     = 0
    confident_words = []

    n_boxes = len(raw_data["text"])
    for i in range(n_boxes):
        word_text = raw_data["text"][i].strip()
        if word_text == "":
            continue

        total_words += 1
        raw_conf = raw_data["conf"][i]

        # Tesseract returns -1 for non-text regions — skip those
        if raw_conf == -1:
            continue

        confidence = raw_conf / 100.0

        # ── THE GATEKEEPER IF-STATEMENT ──────────────────────────────────────
        if confidence >= confidence_threshold:
            confident_words.append({
                "text"      : word_text,
                "confidence": confidence,
                "left"      : raw_data["left"][i],
                "top"       : raw_data["top"][i],
                "width"     : raw_data["width"][i],
                "height"    : raw_data["height"][i],
            })

    filtered_text = " ".join([w["text"] for w in confident_words])

    accepted = len(confident_words)
    rejected = total_words - accepted
    print(f"    Total word candidates : {total_words}")
    print(f"    Accepted (≥ 80%)      : {accepted}")
    print(f"    Rejected (< 80%)      : {rejected}")
    print(f"[✓] RULE 3 — Confidence filter applied.")

    return filtered_text, confident_words


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 4 — VISUAL CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────

def visual_confirmation_ocr(
    filtered_text: str,
    confident_words: list,
    output_txt_path: str = "ocr_output.txt"
) -> None:
    """
    Proves the machine can READ by printing a clean, formatted text block
    and persisting it as a machine-readable .txt file.
    """

    print("\n[►] RULE 4 — Visual Confirmation …")

    separator = "─" * 70
    print(f"\n{separator}")
    print("  EXTRACTED TEXT (words with confidence ≥ 80%)")
    print(separator)
    if filtered_text.strip():
        print(f"\n{filtered_text}\n")
    else:
        print("\n  [!] No high-confidence text found in this image.\n")
        print("      Try lowering confidence_threshold or switching PSM mode.\n")
    print(separator)

    print("\n  Word-level breakdown:")
    print(f"  {'WORD':<25} {'CONFIDENCE':>12}  BBOX (L, T, W, H)")
    print("  " + "─" * 65)
    for w in confident_words:
        bbox = f"({w['left']}, {w['top']}, {w['width']}, {w['height']})"
        print(f"  {w['text']:<25} {w['confidence']*100:>10.1f}%  {bbox}")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("DecodeLabs | Project 4 | PATH 1: OCR Output\n")
        f.write("=" * 70 + "\n\n")
        f.write(filtered_text)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"Words accepted (≥80% confidence): {len(confident_words)}\n")

    print(f"\n[✓] RULE 4 — Text saved to : {os.path.abspath(output_txt_path)}")


def annotate_image_ocr(
    original_image_bgr: np.ndarray,
    confident_words: list,
    output_image_path: str = "ocr_annotated.jpg"
) -> None:
    """Draw green bounding boxes around every high-confidence word."""

    annotated = original_image_bgr.copy()

    for w in confident_words:
        x, y, wd, ht = w["left"], w["top"], w["width"], w["height"]
        label = f"{w['text']} ({w['confidence']*100:.0f}%)"

        cv2.rectangle(annotated, (x, y), (x + wd, y + ht),
                      color=(0, 200, 0), thickness=2)
        cv2.putText(annotated, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_image_path, annotated)
    print(f"[✓] Annotated image saved : {os.path.abspath(output_image_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# ░░  MAIN EXECUTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    IMAGE_PATH = "sample_text.png"   # ← change to your image filename

    original    = load_image(IMAGE_PATH)
    preprocessed = preprocess_for_ocr(original)

    try:
        text, words = run_ocr_with_confidence_filter(
            preprocessed,
            config=TESSERACT_CONFIG,
            confidence_threshold=0.80
        )
    except RuntimeError as e:
        print(f"\n[FATAL] OCR failed: {e}")
        sys.exit(1)

    visual_confirmation_ocr(text, words, output_txt_path="ocr_output.txt")
    annotate_image_ocr(original, words, output_image_path="ocr_annotated.jpg")

    cv2.imwrite("ocr_preprocessed.jpg", preprocessed)
    print(f"[✓] Preprocessed image  : ocr_preprocessed.jpg\n")

    print("=" * 70)
    print("  PATH 1 COMPLETE — Ready for DecodeLabs portal submission.")
    print("=" * 70)
