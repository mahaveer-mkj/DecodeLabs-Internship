<div align="center">

<img src="https://img.shields.io/badge/DecodeLabs-AI%20Internship-6366F1?style=for-the-badge" />

# 🧠 DecodeLabs AI Internship
### Mahaveer Mundaluhari — Artificial Intelligence Intern

![Duration](https://img.shields.io/badge/Duration-May%2027%20–%20June%2027%2C%202026-F59E0B?style=for-the-badge)
![Projects](https://img.shields.io/badge/Projects-4%20Total-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Progress-3B82F6?style=for-the-badge)
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
| 4 | 🔒 Coming Soon | — | ⏳ Upcoming | — |

---

## 🚀 Internship Progress

```
Week 1  ██████████  Project 1 — Rule-Based Chatbot         ✅ Complete
Week 2  ██████████  Project 2 — KNN Classification         ✅ Complete
Week 3  ██████████  Project 3 — Recommendation Engine      ✅ Complete
Week 4  ░░░░░░░░░░  Project 4 — TBA                        ⏳ Upcoming
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

## 🛠️ Tech Stack

| Tool / Library | Used In | Purpose |
|----------------|---------|---------|
| `Python 3.9+` | All projects | Core language |
| `scikit-learn` | Projects 2, 3 | KNN model / TF-IDF vectorizer, scaler, metrics, cosine similarity |
| `numpy` | Project 2 | Numerical computation |
| `pandas` | Projects 2, 3 | Data handling, ranking and sorting |
| `matplotlib` | Project 2 | Elbow plot, confusion matrix |
| `seaborn` | Project 2 | Plot styling |
| `scipy` | Project 3 | Sparse vector math (cosine similarity) |
| `Git + GitHub` | All projects | Version control and portfolio |

---

## 📈 Learning Arc

```
Project 1                Project 2                Project 3                Project 4
────────────             ────────────             ────────────             ────────────
Rule-Based AI      →     Supervised ML      →     Content-Based      →     TBA
                                                    Filtering

No learning              Learns from data         Matches by feature
Explicit rules           Finds patterns           similarity, not labels
Deterministic            Probabilistic            Vector-space geometry
O(1) lookup              Distance-based           Angle-based (cosine)
```

> Each project builds directly on the mental models established by the previous one. You cannot fully appreciate *why* feature scaling matters in KNN without first understanding deterministic logic in Project 1 — and you cannot appreciate why cosine similarity beats Euclidean distance without first seeing how KNN actually uses Euclidean distance in Project 2.

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
