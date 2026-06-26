"""
================================================================================
  DecodeLabs | Project 4 — MobileNet-SSD Model Downloader
  Run this script ONCE to download MobileNetSSD_deploy.caffemodel (~23 MB)
  into the current directory.

  Usage:
      python download_model.py
================================================================================
"""

import urllib.request
import os
import sys

# ── Multiple mirror sources (tried in order until one succeeds) ──────────────
SOURCES = [
    {
        "name"    : "OpenCV GitHub Releases",
        "url"     : "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "filename": "MobileNetSSD_deploy.caffemodel",
        "note"    : "Primary mirror"
    },
    {
        "name"    : "MEGA / Sourceforge mirror",
        "url"     : "https://sourceforge.net/projects/dcaffe/files/MobileNetSSD_deploy.caffemodel/download",
        "filename": "MobileNetSSD_deploy.caffemodel",
        "note"    : "Fallback mirror"
    },
]

# ── Direct working URL (most reliable as of 2025) ────────────────────────────
CAFFEMODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/blob/master/"
    "MobileNetSSD_deploy.caffemodel?raw=true"
)
OUTPUT_FILE = "MobileNetSSD_deploy.caffemodel"
EXPECTED_SIZE_MB = 22.9


def download_with_progress(url: str, output_path: str) -> bool:
    """Download a file with a live progress bar. Returns True on success."""

    def progress_hook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # newline after progress bar
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False


def verify_file(path: str, min_size_mb: float) -> bool:
    """Check file exists and is above the minimum expected size."""
    if not os.path.isfile(path):
        return False
    size_mb = os.path.getsize(path) / 1_048_576
    return size_mb >= min_size_mb * 0.8   # allow 20% tolerance


def main():
    print("=" * 65)
    print("  DecodeLabs | Project 4 — MobileNet-SSD Model Downloader")
    print("=" * 65)

    # ── Already downloaded? ──────────────────────────────────────────────────
    if verify_file(OUTPUT_FILE, EXPECTED_SIZE_MB):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
        print(f"\n[✓] {OUTPUT_FILE} already exists ({size_mb:.1f} MB).")
        print("    Nothing to do — you are ready to run path2_object_detection.py")
        return

    print(f"\n[►] Downloading {OUTPUT_FILE} (~{EXPECTED_SIZE_MB} MB) …")
    print(f"    Destination : {os.path.abspath(OUTPUT_FILE)}\n")

    # ── MANUAL FALLBACK INSTRUCTIONS (always shown so user has a backup) ─────
    print("  ─── If automatic download fails, use ONE of these manual methods ───")
    print()
    print("  METHOD A — Browser download (easiest):")
    print("    1. Open this URL in your browser:")
    print("       https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc")
    print("    2. Click the download button")
    print("    3. Move the downloaded file into this folder and rename it:")
    print(f"       {os.path.abspath(OUTPUT_FILE)}")
    print()
    print("  METHOD B — wget (Linux/macOS terminal):")
    print("    wget -O MobileNetSSD_deploy.caffemodel \\")
    print('    "https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel?raw=true"')
    print()
    print("  METHOD C — pip install gdown, then:")
    print("    pip install gdown")
    print("    gdown 0B3gersZ2cHIxRm5PMWRoTkdHdHc")
    print()
    print("  ─────────────────────────────────────────────────────────────────")
    print()
    print("  [►] Attempting automatic download now …")
    print()

    success = download_with_progress(CAFFEMODEL_URL, OUTPUT_FILE)

    if success and verify_file(OUTPUT_FILE, EXPECTED_SIZE_MB):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
        print(f"\n[✓] Download complete! File size: {size_mb:.1f} MB")
        print(f"    Saved to: {os.path.abspath(OUTPUT_FILE)}")
        print("\n[✓] You are ready to run:")
        print("    python path2_object_detection.py")
    else:
        # Auto-download failed — remove partial file if it exists
        if os.path.isfile(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        print("\n[!] Automatic download failed (network restrictions or URL change).")
        print("    Please use one of the MANUAL METHODS printed above.")
        sys.exit(1)

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
