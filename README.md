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
| 3 | 🔒 Coming Soon | — | 🔄 In Progress | — |
| 4 | 🔒 Coming Soon | — | ⏳ Upcoming | — |

---

## 🚀 Internship Progress

```
Week 1  ██████████  Project 1 — Rule-Based Chatbot         ✅ Complete
Week 2  ██████████  Project 2 — KNN Classification         ✅ Complete
Week 3  ░░░░░░░░░░  Project 3 — TBA                        🔄 In Progress
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

## 🛠️ Tech Stack

| Tool / Library | Used In | Purpose |
|----------------|---------|---------|
| `Python 3.9+` | All projects | Core language |
| `scikit-learn` | Project 2 | KNN model, scaler, metrics |
| `numpy` | Project 2 | Numerical computation |
| `pandas` | Project 2 | Data handling |
| `matplotlib` | Project 2 | Elbow plot, confusion matrix |
| `seaborn` | Project 2 | Plot styling |
| `Git + GitHub` | All projects | Version control and portfolio |

---

## 📈 Learning Arc

```
Project 1                Project 2                Project 3                Project 4
────────────             ────────────             ────────────             ────────────
Rule-Based AI      →     Supervised ML      →     TBA                →     TBA
                         
No learning              Learns from data
Explicit rules           Finds patterns
Deterministic            Probabilistic
O(1) lookup              Distance-based
```

> Each project builds directly on the mental models established by the previous one. You cannot fully appreciate *why* feature scaling matters in KNN without first understanding deterministic logic in Project 1.

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
