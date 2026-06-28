
<div align="center">

<img src="https://img.shields.io/badge/DecodeLabs-AI%20Internship-6366F1?style=for-the-badge" />

# 🧠 DecodeLabs AI Internship
### Mahaveer Mundaluhari — Artificial Intelligence Intern

![Duration](https://img.shields.io/badge/Duration-May%2027%20–%20June%2027%2C%202026-F59E0B?style=for-the-badge)
![Projects](https://img.shields.io/badge/Projects-4%20Total-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)

**A structured, week-by-week AI project portfolio built during my internship at DecodeLabs —  
progressing from rule-based logic to supervised machine learning and beyond.**

[Projects](#-projects) · [Progress](#-internship-progress) · [Tech Stack](#-tech-stack) · [About Me](#-about-me)

---

</div>

## 📖 About This Repository

This repository contains all projects assigned and completed during my **Artificial Intelligence Internship at [DecodeLabs](https://www.decodelabs.tech/)**. Each project is self-contained in its own folder with full source code, output files, and a dedicated README.

The internship follows a deliberate learning arc — starting from the absolute foundations of AI (rule-based logic) and progressively advancing toward core machine learning algorithms, building real, working systems at every step.

---

## 📂 Projects

| # | Project | Concept | Status | Folder |
|---|---------|---------|--------|--------|
| 1 | [🤖 WavYy — Rule-Based AI Chatbot](#project-1--wayyy--rule-based-ai-chatbot) | Control Flow, Hash Maps, Decision Logic | ✅ Complete | [`Project-1-Chatbot`](./Project-1-Chatbot/) |
| 2 | [🌸 Iris KNN Classification Pipeline](#project-2--iris-knn-classification-pipeline) | Supervised Learning, KNN, F1 Score | ✅ Complete | [`Project-2-KNN-Classification`](./Project-2-KNN-Classification/) |
| 3 | [🎯 Tech Stack Recommender — "The Digital Matchmaker"](#project-3--tech-stack-recommender--the-digital-matchmaker) · [🔗 Live Demo](https://thedigitalmatchmaker.streamlit.app/) | Content-Based Filtering, TF-IDF, Cosine Similarity | ✅ Complete | [`Project-3-Recommendation-Engine`](./Project-3-Recommendation-Engine/) |
| 4 | [👁️ Building the Machine's Optic Nerve](#project-4--building-the-machines-optic-nerve) | Computer Vision, Pre-Trained AI, OCR, Object Detection | ✅ Complete | [`Project-4-Computer-Vision`](./Project-4-Computer-Vision/) |

---

## 🚀 Internship Progress

```
Week 1  ██████████  Project 1 — Rule-Based Chatbot              ✅ Complete
Week 2  ██████████  Project 2 — KNN Classification              ✅ Complete
Week 3  ██████████  Project 3 — Recommendation Engine           ✅ Complete
Week 4  ██████████  Project 4 — Computer Vision                 ✅ Complete
```

---

## 🗂️ Project Details

### Project 1 · WavYy — Rule-Based AI Chatbot
> **Folder:** [`Project-1-Chatbot/`](./Project-1-Chatbot/)

WavYy is a deterministic, terminal-based chatbot built entirely on predefined rules and a structured knowledge base — no ML, no external APIs. It demonstrates the foundational AI principles that every learning algorithm is built upon: control flow, conditional logic, and dictionary-based O(1) lookup.

**Key concepts covered:**
- Rule-based decision making vs. statistical learning
- Input sanitization (`.strip().lower()`) for robust string matching
- Hash map architecture for scalable Q&A lookup
- Graceful fallback handling for unknown queries

```bash
cd Project-1-Chatbot
python chatbot.py
```

---

### Project 2 · Iris KNN Classification Pipeline
> **Folder:** [`Project-2-KNN-Classification/`](./Project-2-KNN-Classification/)

A production-grade supervised machine learning pipeline that classifies Iris flower species using the K-Nearest Neighbors algorithm. Built with a strict **Input → Process → Output (IPO)** architecture, with every design decision documented and justified.

**Key concepts covered:**
- Supervised Learning vs. rule-based heuristics
- Feature scaling with `StandardScaler` (mean=0, variance=1)
- Hyperparameter tuning via the Elbow Method
- Honest evaluation: Confusion Matrix, Precision, Recall, F1 Score

```bash
cd Project-2-KNN-Classification
pip install -r requirements.txt
python iris_knn_pipeline.py
```

**Output previews:**

| Elbow Method | Confusion Matrix |
|---|---|
| ![Elbow Plot](./Project-2-KNN-Classification/outputs/elbow_plot.png) | ![Confusion Matrix](./Project-2-KNN-Classification/outputs/confusion_matrix.png) |

---

### Project 3 · Tech Stack Recommender — "The Digital Matchmaker"
> **Folder:** [`Project-3-Recommendation-Engine/`](./Project-3-Recommendation-Engine/)  
> **Live Demo:** [thedigitalmatchmaker.streamlit.app](https://thedigitalmatchmaker.streamlit.app/)

A content-based recommendation engine that maps user skills/interests to platforms, tools, and job roles using TF-IDF vectorization and Cosine Similarity — strictly avoiding Euclidean distance and binary overlap. Includes dual cold-start handling (structural and semantic) so a user is never left with a zero vector, plus a bonus 1-5 preference-weighting layer.

**Key concepts covered:**
- Content-Based Filtering vs. Collaborative Filtering
- TF-IDF: penalizing generic terms, rewarding specific ones
- Cosine Similarity — angle, not distance — and why that matters at scale
- Strict vocabulary alignment between user and item feature spaces

```bash
cd Project-3-Recommendation-Engine
pip install -r requirements.txt
python main.py
```

**Sample output:**

```
#1  Machine Learning Engineer Role  [JobRole]  -> similarity = 0.406
#2  Cloud DevOps Engineer Role     [JobRole]  -> similarity = 0.4039
#3  MLOps Engineer Role            [JobRole]  -> similarity = 0.3785
```

Full run, including both cold-start recovery paths: [`outputs/sample_run_output.txt`](./Project-3-Recommendation-Engine/outputs/sample_run_output.txt)

---

### Project 4 · Building the Machine's Optic Nerve
> **Folder:** [`Project-4-Computer-Vision/`](./Project-4-Computer-Vision/)

A production-grade computer vision pipeline that teaches a machine to extract intelligence from raw pixel arrays. Two independent paths — **Optical Character Recognition (OCR)** and **Object Detection** — each pass through four Gatekeeper Rule validations before producing verified, annotated output.

**Path 1 (OCR):** raw image → grayscale, blur, deskew, Otsu threshold → Tesseract → word-level confidence filter (≥80%) → machine-readable text + annotated image.  
**Path 2 (Object Detection):** raw image → 4D blob → MobileNet-SSD forward pass → 80% confidence gate → decoded bounding boxes drawn on the original scene.

**Key concepts covered:**
- Images as 3D arrays and pre-processing pipelines (grayscale, blur, deskew, Otsu)
- OCR engine integration (Tesseract) and Page Segmentation Mode tuning
- Pre-trained deep learning models: MobileNet-SSD via OpenCV's `dnn` module
- 4D blob construction: resizing, mean subtraction, pixel scaling
- Normalised coordinate decoding and confidence-based filtering

```bash
cd Project-4-Computer-Vision

# Path 1 — OCR
pip install pytesseract opencv-python numpy pillow
sudo apt-get install tesseract-ocr   # Linux; see README for Windows/macOS
python path1_ocr.py

# Path 2 — Object Detection
pip install opencv-python numpy
# Download model files: MobileNetSSD_deploy.prototxt and .caffemodel (see README)
python path2_object_detection.py
```

**Sample output — Object Detection (80% filter applied):**
```
    Total raw candidates  : 100
    Accepted (≥ 80%)      : 3
    Rejected (< 80%)      : 97  (false positives dropped)

    OBJECT             CONFIDENCE  PIXEL BBOX (x1,y1) → (x2,y2)
    ─────────────────────────────────────────────────────────────
    person                   96.2%  (142,80)  → (398,695)
    car                      92.7%  (620,210) → (1100,590)
    dog                      87.4%  (44,350)  → (280,680)
```

---

## 🛠️ Tech Stack

| Tool / Library | Used In | Purpose |
|----------------|---------|---------|
| `Python 3.9+` | All projects | Core language |
| `scikit-learn` | Projects 2, 3 | KNN model / TF-IDF vectorizer, scaler, metrics, cosine similarity |
| `numpy` | Projects 2, 4 | Numerical computation, pixel array manipulation |
| `pandas` | Projects 2, 3 | Data handling, ranking and sorting |
| `matplotlib` | Project 2 | Elbow plot, confusion matrix |
| `seaborn` | Project 2 | Plot styling |
| `scipy` | Project 3 | Sparse vector math (cosine similarity) |
| `opencv-python` | Project 4 | Image I/O, pre-processing, `cv2.dnn` inference, annotation |
| `pytesseract` | Project 4 | OCR engine wrapper (Tesseract) |
| `pillow` | Project 4 | Optional image loading fallback |
| `Git + GitHub` | All projects | Version control and portfolio |

---

## 📈 Learning Arc

```
Project 1                Project 2                Project 3                Project 4
────────────             ────────────             ────────────             ────────────
Rule-Based AI      →     Supervised ML      →     Content-Based      →     Computer Vision
                                                    Filtering              + Pre-Trained AI

No learning              Learns from data         Matches by feature       Understands raw
Explicit rules           Finds patterns           similarity, not labels    pixel arrays
Deterministic            Probabilistic            Vector-space geometry    Deep neural nets
O(1) lookup              Distance-based           Angle-based (cosine)     80% confidence gate
```

> Each project builds directly on the mental models established by the previous one. You cannot fully appreciate *why* feature scaling matters in KNN without first understanding deterministic logic in Project 1 — and you cannot appreciate why cosine similarity beats Euclidean distance without first seeing how KNN actually uses Euclidean distance in Project 2. Project 4 completes the arc by moving from structured tabular/text data into the unstructured world of images, introducing pre-trained neural architectures that none of the previous projects could leverage.

---

## 🏢 About DecodeLabs

[DecodeLabs](https://www.decodelabs.tech/) is an AI-focused organisation that provides structured, hands-on internship programs designed to bridge the gap between academic theory and real-world AI development. Projects are assigned progressively, each introducing a new layer of complexity.

---

## 👤 About Me

**Mahaveer Mundaluhari**  
*Artificial Intelligence Intern @ [DecodeLabs](https://www.decodelabs.tech/)*  
*B.S. Data Science & Applications — IIT Madras*  
*B.Tech CSE (AI & ML) — OUTR Bhubaneswar*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mahaveer%20Mundaluhari-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/mahaveer-mundaluhari/)
[![GitHub](https://img.shields.io/badge/GitHub-mahaveer--mkj-181717?style=flat&logo=github)](https://github.com/mahaveer-mkj)
[![Email](https://img.shields.io/badge/Email-mahaveer%40maxiwoxi.com-EA4335?style=flat&logo=gmail)](mailto:mahaveer@maxiwoxi.com)

---

<div align="center">

*This repository is actively updated throughout the internship (May 27 – June 27, 2026).*  
**If this work helped you, consider giving it a ⭐**

</div>
