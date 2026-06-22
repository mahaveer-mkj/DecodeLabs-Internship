"""
================================================================================
  DecodeLabs Internship | Project 4: Building the Machine's Optic Nerve
  PATH 1: Optical Character Recognition (OCR)
  Author  : Mahaveer (AI Intern, DecodeLabs)
  Engine  : Google Tesseract via pytesseract
================================================================================

PROJECT OBJECTIVE
-----------------
Bridge the gap between physical reality and computational logic by extracting
machine-readable intelligence from unstructured visual data (raw image pixels).
This script ingests a raw image, aggressively pre-processes it to remove noise,
feeds it into the Tesseract OCR engine, and outputs clean, formatted text.

GATEKEEPER RULES IMPLEMENTED
  [1] Library Integration   — pytesseract with PSM tuning
  [2] Pre-Processing        — Grayscale → Gaussian Blur → Deskew → Otsu Threshold
  [3] Accuracy Benchmarking — 80% confidence filter on word-level detections
  [4] Visual Confirmation   — Formatted, machine-readable text output to console + file

INSTALLATION (run once in your terminal)
-----------------------------------------
  pip install pytesseract opencv-python numpy pillow
  # Also install the Tesseract binary:
  # Windows : https://github.com/UB-Mannheim/tesseract/wiki
  # Linux   : sudo apt-get install tesseract-ocr
  # macOS   : brew install tesseract
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 1 — LIBRARY INTEGRATION
# ░░  Import every dependency and verify Tesseract is reachable.
# ─────────────────────────────────────────────────────────────────────────────

import cv2                        # OpenCV  — image I/O and pre-processing
import numpy as np                # NumPy   — matrix arithmetic for deskewing
import pytesseract                # Wrapper — calls Google's Tesseract binary
from PIL import Image             # Pillow  — alternative image loader (optional)
import os
import sys

# ── OPTIONAL: point to your Tesseract binary if it is not on PATH ──────────
# Uncomment and edit the line below if Tesseract is not found automatically.
# Windows example:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── PSM (Page Segmentation Mode) selector ──────────────────────────────────
# PSM tells Tesseract how the text is laid out BEFORE it tries to read it.
#   --psm 3  → Fully automatic page segmentation (varied layouts, mixed content)
#   --psm 6  → Assume a single uniform block of text (clean documents, invoices)
#   --psm 11 → Sparse text — find as much text as possible (receipts, labels)
PSM_MODE = 6   # ← change to 3 for general-purpose documents

TESSERACT_CONFIG = f"--oem 3 --psm {PSM_MODE}"
# --oem 3 = use the LSTM neural-network OCR engine (most accurate)

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

    # cv2.imread reads the image as a 3D array: (Height, Width, 3 channels BGR)
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
# ░░  Pipeline: BGR → Grayscale → Gaussian Blur → Deskew → Otsu Threshold
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
    # A colour image is a 3D array: shape = (H, W, 3).  BGR channels carry red,
    # green, blue intensity per pixel.  Tesseract reads intensity only — colour
    # is irrelevant noise.  We collapse the 3 channels into 1 using the ITU-R
    # perceptual formula:  Y = 0.114B + 0.587G + 0.299R
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"    [2a] Grayscale : shape changed from {image_bgr.shape} → {gray.shape}")

    # ── STEP 2b: GAUSSIAN BLUR ───────────────────────────────────────────────
    # Scanner artifacts, compression noise, and paper grain appear as random
    # high-frequency pixel spikes.  A Gaussian kernel (5×5) averages each pixel
    # with its neighbours using a bell-curve weighting — edges survive, spikes
    # are smoothed.  Kernel size must be ODD integers.
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    print(f"    [2b] Gaussian Blur applied  : kernel 5×5")

    # ── STEP 2c: DESKEWING ───────────────────────────────────────────────────
    # A photograph of a tilted document makes OCR fail because Tesseract reads
    # horizontally.  We use the Minimum Area Rectangle of all non-white pixels
    # to compute the skew angle and then rotate the image to compensate.
    blurred_for_skew = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh_for_skew = cv2.threshold(
        blurred_for_skew, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Collect coordinates of all "dark" (foreground text) pixels
    coords = np.column_stack(np.where(thresh_for_skew > 0))

    if coords.size == 0:
        print(f"    [2c] Deskew : no foreground pixels found — skipping rotation")
        deskewed = blurred
    else:
        # cv2.minAreaRect returns the smallest enclosing rectangle and its angle
        angle = cv2.minAreaRect(coords)[-1]
        # Tesseract convention: angle is in [-90, 0), correct toward 0°
        if angle < -45:
            angle = 90 + angle
        # Build affine rotation matrix around image centre
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
    # After blur and deskew the image still has grey gradients.  Otsu's Method
    # automatically finds the OPTIMAL single threshold value that minimises
    # intra-class variance between black and white pixel populations.
    # Result: every pixel is forced to EXACTLY 0 (black) or 255 (white).
    # THRESH_BINARY_INV makes text BLACK on a WHITE background — Tesseract
    # expects this polarity for best accuracy.
    _, binary = cv2.threshold(
        deskewed, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"    [2d] Otsu Thresholding : image is now pure black-and-white")
    print(f"[✓] RULE 2 — Pre-Processing complete.  Output shape: {binary.shape}")

    return binary


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 3 — ACCURACY BENCHMARKING (80% CONFIDENCE FILTER)
# ░░  Use pytesseract.image_to_data() to get per-word confidence scores.
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr_with_confidence_filter(
    binary_image: np.ndarray,
    config: str,
    confidence_threshold: float = 0.80
) -> tuple[str, list[dict]]:
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

    # image_to_data() returns a TSV string with columns:
    #   level | page_num | block_num | par_num | line_num | word_num |
    #   left  | top      | width     | height  | conf     | text
    raw_data = pytesseract.image_to_data(
        binary_image,
        config=config,
        output_type=pytesseract.Output.DICT  # parse directly into Python dict
    )

    total_words    = 0
    confident_words = []

    n_boxes = len(raw_data["text"])
    for i in range(n_boxes):
        word_text = raw_data["text"][i].strip()
        if word_text == "":          # Tesseract returns empty strings for spaces
            continue

        total_words += 1
        raw_conf = raw_data["conf"][i]

        # Tesseract returns conf as integer 0–100; convert to 0.0–1.0
        # It also returns -1 for non-text regions — skip those
        if raw_conf == -1:
            continue

        confidence = raw_conf / 100.0

        # ── THE GATEKEEPER IF-STATEMENT ──────────────────────────────────────
        # Any detection below 80% is treated as noise and DROPPED.
        if confidence >= confidence_threshold:
            confident_words.append({
                "text"      : word_text,
                "confidence": confidence,
                "left"      : raw_data["left"][i],
                "top"       : raw_data["top"][i],
                "width"     : raw_data["width"][i],
                "height"    : raw_data["height"][i],
            })
        # else: silently discard — the machine is not sure enough

    # Reconstruct readable text from validated words only
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
# ░░  Print the pristine machine-readable text and save it to a file.
# ─────────────────────────────────────────────────────────────────────────────

def visual_confirmation_ocr(
    filtered_text: str,
    confident_words: list[dict],
    output_txt_path: str = "ocr_output.txt"
) -> None:
    """
    Proves the machine can READ by printing a clean, formatted text block
    and persisting it as a machine-readable .txt file.
    Also annotates the original image with bounding boxes around each
    high-confidence word and saves the annotated image.
    """

    print("\n[►] RULE 4 — Visual Confirmation …")

    # ── CONSOLE OUTPUT ───────────────────────────────────────────────────────
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

    # ── PER-WORD DETAIL TABLE ────────────────────────────────────────────────
    print("\n  Word-level breakdown:")
    print(f"  {'WORD':<25} {'CONFIDENCE':>12}  BBOX (L, T, W, H)")
    print("  " + "─" * 65)
    for w in confident_words:
        bbox = f"({w['left']}, {w['top']}, {w['width']}, {w['height']})"
        print(f"  {w['text']:<25} {w['confidence']*100:>10.1f}%  {bbox}")

    # ── SAVE TO TXT FILE ─────────────────────────────────────────────────────
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("DecodeLabs | Project 4 | PATH 1: OCR Output\n")
        f.write("=" * 70 + "\n\n")
        f.write(filtered_text)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"Words accepted (≥80% confidence): {len(confident_words)}\n")

    print(f"\n[✓] RULE 4 — Text saved to : {os.path.abspath(output_txt_path)}")


def annotate_image_ocr(
    original_image_bgr: np.ndarray,
    confident_words: list[dict],
    output_image_path: str = "ocr_annotated.jpg"
) -> None:
    """Draw green bounding boxes around every high-confidence word."""

    annotated = original_image_bgr.copy()

    for w in confident_words:
        x, y, wd, ht = w["left"], w["top"], w["width"], w["height"]
        label = f"{w['text']} ({w['confidence']*100:.0f}%)"

        # Draw rectangle around each validated word
        cv2.rectangle(annotated, (x, y), (x + wd, y + ht),
                      color=(0, 200, 0), thickness=2)

        # Draw label above the box
        cv2.putText(annotated, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_image_path, annotated)
    print(f"[✓] Annotated image saved : {os.path.abspath(output_image_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# ░░  MAIN EXECUTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── UPDATE THIS PATH to point to your test image ─────────────────────────
    IMAGE_PATH = "sample_text.png"   # ← change to your image filename

    # ── PIPELINE EXECUTION ───────────────────────────────────────────────────
    # 1. Load
    original = load_image(IMAGE_PATH)

    # 2. Pre-process (Rule 2)
    preprocessed = preprocess_for_ocr(original)

    # 3. OCR + confidence filter (Rule 3)
    text, words = run_ocr_with_confidence_filter(
        preprocessed,
        config=TESSERACT_CONFIG,
        confidence_threshold=0.80          # THE 80% GATEKEEPER
    )

    # 4. Visual confirmation (Rule 4)
    visual_confirmation_ocr(text, words, output_txt_path="ocr_output.txt")
    annotate_image_ocr(original, words, output_image_path="ocr_annotated.jpg")

    # ── OPTIONAL: Save the preprocessed image for inspection ─────────────────
    cv2.imwrite("ocr_preprocessed.jpg", preprocessed)
    print(f"[✓] Preprocessed image  : ocr_preprocessed.jpg\n")

    print("=" * 70)
    print("  PATH 1 COMPLETE — Ready for DecodeLabs portal submission.")
    print("=" * 70)
