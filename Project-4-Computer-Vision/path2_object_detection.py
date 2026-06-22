"""
================================================================================
  DecodeLabs Internship | Project 4: Building the Machine's Optic Nerve
  PATH 2: Object Detection
  Author  : Mahaveer (AI Intern, DecodeLabs)
  Engine  : MobileNet-SSD via OpenCV cv2.dnn
================================================================================

PROJECT OBJECTIVE
-----------------
Enable the machine to locate and identify physical objects in raw image data.
This script loads a pre-trained MobileNet-SSD neural network, constructs a
4D Blob from the input image, runs a forward pass through the network, applies
an 80% confidence filter, decodes normalized spatial coordinates into pixel
bounding boxes, and overlays annotated detections onto the original image.

GATEKEEPER RULES IMPLEMENTED
  [1] Library Integration   — OpenCV cv2.dnn + MobileNet-SSD Caffe model
  [2] Pre-Processing        — 4D Blob via cv2.dnn.blobFromImage (300×300, mean subtraction)
  [3] Accuracy Benchmarking — 80% confidence filter (if confidence >= 0.80)
  [4] Visual Confirmation   — Bounding boxes + labels drawn on original image

MODEL FILES REQUIRED (download once)
--------------------------------------
  MobileNetSSD_deploy.prototxt
      https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt

  MobileNetSSD_deploy.caffemodel
      https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc
  OR from the OpenCV zoo:
      https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_mobilenet

  Place both files in the SAME directory as this script.

INSTALLATION (run once in terminal)
--------------------------------------
  pip install opencv-python numpy
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 1 — LIBRARY INTEGRATION
# ░░  Import libraries and load the MobileNet-SSD Caffe model via cv2.dnn
# ─────────────────────────────────────────────────────────────────────────────

import cv2      # OpenCV — image I/O, dnn module, drawing utilities
import numpy as np
import os
import sys

# ── CLASS LABELS for MobileNet-SSD (trained on PASCAL VOC 20 classes) ───────
# The network outputs a class index (integer).  This list maps each index back
# to a human-readable label.  Index 0 is reserved as "background".
CLASS_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle",     "bus",       "car",     "cat",  "chair",
    "cow",        "diningtable", "dog",   "horse", "motorbike",
    "person",     "pottedplant", "sheep", "sofa",  "train",
    "tvmonitor"
]

# ── COLOUR PALETTE — one distinct BGR colour per class for visual clarity ────
# np.random.seed ensures the same colour map every run (reproducible output)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASS_LABELS), 3), dtype="uint8")

# ── MODEL FILE PATHS ─────────────────────────────────────────────────────────
# Update these if your model files are in a different directory.
PROTOTXT_PATH   = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = "MobileNetSSD_deploy.caffemodel"


def load_model(prototxt: str, caffemodel: str) -> cv2.dnn_Net:
    """
    Load the MobileNet-SSD Caffe model into OpenCV's DNN backend.
    cv2.dnn.readNetFromCaffe() returns a Net object that holds the
    network architecture (prototxt) and learned weights (caffemodel).
    """
    for path in [prototxt, caffemodel]:
        if not os.path.isfile(path):
            print(f"\n[ERROR] Model file not found: {path}")
            print("        See the file header for download instructions.")
            sys.exit(1)

    # Load the neural network from disk — no GPU required; runs on CPU
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    print("=" * 70)
    print("  DecodeLabs | Project 4 | PATH 2: Object Detection")
    print("=" * 70)
    print("[✓] RULE 1 — Library Integration : MobileNet-SSD loaded via cv2.dnn")
    print(f"            Prototxt  : {prototxt}")
    print(f"            Weights   : {caffemodel}")
    print(f"            Classes   : {len(CLASS_LABELS) - 1} object categories")
    return net


# ─────────────────────────────────────────────────────────────────────────────
# ░░  IMAGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load and validate an image from disk; return BGR array."""
    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"\n[ERROR] OpenCV could not decode: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"\n[✓] Image loaded  : {image_path}  ({w}×{h} px)")
    return image


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 2 — PRE-PROCESSING INTEGRITY
# ░░  Construct a 4D Blob — the exact tensor format MobileNet-SSD expects.
# ─────────────────────────────────────────────────────────────────────────────

def construct_blob(image_bgr: np.ndarray) -> np.ndarray:
    """
    Transforms a raw HxWx3 BGR image into the 4D tensor (Blob) that the
    MobileNet-SSD neural network requires as input.

    cv2.dnn.blobFromImage() performs THREE simultaneous operations:
    ┌────────────────────────────────────────────────────────────────────┐
    │ 1. RESIZE   → scales the image to exactly 300×300 pixels          │
    │              (the input dimension MobileNet-SSD was trained on)    │
    │ 2. SCALE    → multiplies each pixel by scalefactor=0.007843       │
    │              This normalises values from [0, 255] → [0, 2.0]     │
    │              (matches the training data normalisation)             │
    │ 3. MEAN SUB → subtracts the training-set mean RGB per channel     │
    │              mean=(127.5, 127.5, 127.5) centres the distribution  │
    │              around zero, which stabilises gradient computations.  │
    └────────────────────────────────────────────────────────────────────┘
    Output shape: (1, 3, 300, 300)
                   ↑  ↑   ↑    ↑
                   N  C   H    W   ← standard NCHW (Batch, Channel, H, W)
    """

    print("\n[►] RULE 2 — Pre-Processing : constructing 4D Blob …")

    blob = cv2.dnn.blobFromImage(
        image   = image_bgr,
        scalefactor = 0.007843,           # pixel normalisation: 1/127.5
        size    = (300, 300),             # MobileNet-SSD input resolution
        mean    = (127.5, 127.5, 127.5),  # ImageNet mean subtraction (BGR)
        swapRB  = False,                  # OpenCV is already BGR — no swap needed
        crop    = False                   # stretch to fit; do not centre-crop
    )

    print(f"    Input image shape : {image_bgr.shape}  (H × W × C)")
    print(f"    Blob shape        : {blob.shape}  (N, C, H, W)")
    print(f"    Resize target     : 300 × 300 px")
    print(f"    Mean subtraction  : (127.5, 127.5, 127.5)")
    print(f"    Scale factor      : 0.007843  → pixels normalised to ~[0, 2.0]")
    print(f"[✓] RULE 2 — 4D Blob constructed successfully.")

    return blob


# ─────────────────────────────────────────────────────────────────────────────
# ░░  NEURAL NETWORK FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(net: cv2.dnn_Net, blob: np.ndarray) -> np.ndarray:
    """
    Feed the blob into the network and retrieve raw detection output.

    net.setInput() loads the blob into the network's input layer.
    net.forward() executes all layers from input → output (one forward pass).

    detections shape: (1, 1, N, 7)
                       ↑  ↑  ↑  ↑
                       |  |  |  └── 7 values per detection:
                       |  |  |        [0] batch index (always 0)
                       |  |  |        [1] class label index
                       |  |  |        [2] confidence score  ← THE KEY VALUE
                       |  |  |        [3] x_start (normalised 0–1)
                       |  |  |        [4] y_start (normalised 0–1)
                       |  |  |        [5] x_end   (normalised 0–1)
                       |  |  |        [6] y_end   (normalised 0–1)
                       |  |  └────── N candidate detections
                       └──┴───────── batch and channel wrappers
    """
    net.setInput(blob)
    detections = net.forward()
    print(f"\n[✓] Forward pass complete — {detections.shape[2]} raw candidates generated.")
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 3 — ACCURACY BENCHMARKING (80% CONFIDENCE FILTER)
# ░░  Drop every detection below 80% confidence.  Decode spatial coordinates.
# ─────────────────────────────────────────────────────────────────────────────

def filter_and_decode_detections(
    detections: np.ndarray,
    image_shape: tuple,
    confidence_threshold: float = 0.80
) -> list[dict]:
    """
    Iterates over every raw detection candidate and applies the Gatekeeper.

    Coordinate Decoding Logic
    -------------------------
    MobileNet-SSD outputs NORMALISED coordinates in range [0.0, 1.0].
    These are fractions of the image dimensions, not pixel values.

    To convert:
        pixel_x = normalised_x * image_width
        pixel_y = normalised_y * image_height

    This recovers the actual pixel position relative to the ORIGINAL image
    (not the 300×300 resized blob), so bounding boxes align correctly.
    """

    print(f"\n[►] RULE 3 — Accuracy Benchmarking (threshold ≥ {confidence_threshold*100:.0f}%) …")

    H, W = image_shape[:2]     # original pixel dimensions
    accepted_detections = []
    total = detections.shape[2]

    for i in range(total):
        # detections[0, 0, i] is a 7-element vector
        detection_vector = detections[0, 0, i]

        # Element [2] = confidence score (float between 0.0 and 1.0)
        confidence = float(detection_vector[2])

        # ── THE GATEKEEPER IF-STATEMENT ──────────────────────────────────────
        # This is the core accuracy benchmark.  Any detection the model is
        # less than 80% certain about is silently discarded as a false positive.
        if confidence >= confidence_threshold:

            # Element [1] = integer class index (maps to CLASS_LABELS list)
            class_idx = int(detection_vector[1])

            # Guard against invalid class indices
            if class_idx < 0 or class_idx >= len(CLASS_LABELS):
                continue

            label = CLASS_LABELS[class_idx]
            color = [int(c) for c in COLORS[class_idx]]

            # ── COORDINATE DECODING: normalised → pixel ───────────────────
            # Multiply each normalised coordinate by the corresponding
            # dimension of the ORIGINAL image (not the 300×300 blob).
            x_start = int(detection_vector[3] * W)   # left edge
            y_start = int(detection_vector[4] * H)   # top edge
            x_end   = int(detection_vector[5] * W)   # right edge
            y_end   = int(detection_vector[6] * H)   # bottom edge

            # Clamp to image boundaries (prevents drawing outside canvas)
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end   = min(W - 1, x_end)
            y_end   = min(H - 1, y_end)

            accepted_detections.append({
                "label"     : label,
                "confidence": confidence,
                "x_start"   : x_start,
                "y_start"   : y_start,
                "x_end"     : x_end,
                "y_end"     : y_end,
                "color"     : color,
            })

    rejected = total - len(accepted_detections)
    print(f"    Total raw candidates  : {total}")
    print(f"    Accepted (≥ 80%)      : {len(accepted_detections)}")
    print(f"    Rejected (< 80%)      : {rejected}  (false positives dropped)")
    print(f"[✓] RULE 3 — Confidence filter applied.")

    return accepted_detections


# ─────────────────────────────────────────────────────────────────────────────
# ░░  GATEKEEPER RULE 4 — VISUAL CONFIRMATION
# ░░  Draw bounding boxes + labels on the original image.  Save + print.
# ─────────────────────────────────────────────────────────────────────────────

def visual_confirmation_detection(
    image_bgr: np.ndarray,
    detections: list[dict],
    output_image_path: str = "detection_output.jpg"
) -> None:
    """
    Proves the machine can SEE by physically annotating the original image
    with coloured bounding boxes and class + confidence labels.

    The boxes are drawn using the DECODED pixel coordinates (Rule 3 output),
    not the raw normalised values from the network.
    """

    print(f"\n[►] RULE 4 — Visual Confirmation …")

    annotated = image_bgr.copy()  # work on a copy; preserve the original

    if not detections:
        print("    [!] No objects detected above the 80% confidence threshold.")
        print("        Try a different image or lower the threshold for testing.")
    else:
        print(f"\n    Detected objects:")
        print(f"    {'OBJECT':<18} {'CONFIDENCE':>12}  PIXEL BBOX (x1,y1) → (x2,y2)")
        print("    " + "─" * 65)

        for det in detections:
            label      = det["label"]
            conf       = det["confidence"]
            x1, y1     = det["x_start"], det["y_start"]
            x2, y2     = det["x_end"],   det["y_end"]
            color      = det["color"]

            # ── DRAW FILLED RECTANGLE as background for the label text ───────
            display_text = f"{label}: {conf * 100:.1f}%"
            # Measure text size to create a properly fitted filled background
            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            # Filled rectangle behind text for readability
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color, thickness=cv2.FILLED
            )

            # ── DRAW BOUNDING BOX around detected object ─────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                          color=color, thickness=3)

            # ── DRAW LABEL TEXT ───────────────────────────────────────────────
            cv2.putText(
                annotated, display_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255),   # white text on coloured background
                thickness=2, lineType=cv2.LINE_AA
            )

            bbox_str = f"({x1},{y1}) → ({x2},{y2})"
            print(f"    {label:<18} {conf*100:>10.1f}%  {bbox_str}")

    # ── SAVE ANNOTATED IMAGE ─────────────────────────────────────────────────
    cv2.imwrite(output_image_path, annotated)

    print(f"\n[✓] RULE 4 — Annotated image saved : {os.path.abspath(output_image_path)}")
    print(f"            Total verified detections : {len(detections)}")


# ─────────────────────────────────────────────────────────────────────────────
# ░░  MAIN EXECUTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── UPDATE THESE PATHS ───────────────────────────────────────────────────
    IMAGE_PATH = "sample_image.jpg"   # ← your test image

    # ── PIPELINE EXECUTION ───────────────────────────────────────────────────
    # 1. Load model (Rule 1)
    net = load_model(PROTOTXT_PATH, CAFFEMODEL_PATH)

    # 2. Load image
    image = load_image(IMAGE_PATH)

    # 3. Construct 4D Blob (Rule 2)
    blob = construct_blob(image)

    # 4. Run forward pass through the neural network
    raw_detections = run_inference(net, blob)

    # 5. Apply 80% confidence filter + decode coordinates (Rule 3)
    verified_detections = filter_and_decode_detections(
        raw_detections,
        image_shape=image.shape,
        confidence_threshold=0.80    # THE 80% GATEKEEPER
    )

    # 6. Draw boxes + labels → save output image (Rule 4)
    visual_confirmation_detection(
        image,
        verified_detections,
        output_image_path="detection_output.jpg"
    )

    print("\n" + "=" * 70)
    print("  PATH 2 COMPLETE — Ready for DecodeLabs portal submission.")
    print("=" * 70)
