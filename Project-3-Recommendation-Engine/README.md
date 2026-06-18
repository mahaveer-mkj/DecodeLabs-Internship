<div align="center">

# 🎯 Tech Stack Recommender — "The Digital Matchmaker"
### DecodeLabs — Project 3: AI Recommendation Logic

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)

**A production-grade, fully documented Content-Based Recommendation Engine —  
moving beyond classification to mathematical pattern-matching across user and item feature spaces.**

[View Code](#-project-structure) · [How to Run](#-quick-start) · [Results](#-results--outputs) · [Key Concepts](#-key-concepts-explained)

---

</div>

## 📌 What This Project Does

This pipeline maps a user's skills and career interests to the most aligned tech **platforms, tools, and job roles** using **Content-Based Filtering** — one of the two foundational recommender paradigms. It is built to the same professional, scalable standard as Project 2, following a strict **Input → Process → Output (IPO)** architecture.

| Stage | Task | Key Decision |
|-------|------|-------------|
| **Input** | Capture user skills (min. 3) and optional 1–5 preference weights; validate against the Cold‑Start rules | Dual cold‑start guard — structural (too few skills) **and** semantic (zero‑vector / out‑of‑vocabulary) — a user is never allowed to proceed empty‑handed |
| **Process** | Fit one shared TF‑IDF vocabulary across all items, project the user profile into that *same* space, score every item via Cosine Similarity | Cosine, not Euclidean — angle‑based and scale‑invariant, so it doesn't punish a richly‑described profile |
| **Output** | Sort all similarity scores descending, truncate to Top‑3 | Top‑N truncation only — the full ranked list is never exposed, by design |

> **Design Philosophy:** Every step is justified, not just functional. Inline comments explain *why* each decision is made — this is an educational‑grade professional tool.

---

## 📊 Results & Outputs

The full console capture from an actual run is saved at `outputs/sample_run_output.txt`.

### Standard Match — Weighted Skill Query
> Querying `["Python", "Cloud Computing", "Automation"]` with preference weights `{Python: 5, Cloud Computing: 4, Automation: 3}` returns three job‑role matches, ranked purely by cosine similarity — no manual curation involved.

```
#1  Machine Learning Engineer Role  [JobRole]  -> similarity = 0.406
#2  Cloud DevOps Engineer Role     [JobRole]  -> similarity = 0.4039
#3  MLOps Engineer Role            [JobRole]  -> similarity = 0.3785
```

---

### Cold‑Start Recovery — Zero‑Vector Guard in Action
> Querying three deliberately nonsense skills (`"Quantum Macrame"`, `"Astral Welding"`, `"Vibe Coding"`) produces a 100% out‑of‑vocabulary vector. Instead of returning empty or random results, the semantic cold‑start guard detects `nnz == 0` and silently repopulates the profile from the dataset's own trending terms before re‑scoring.

```
WARNING: Zero-vector detected post-vectorization (skills have no overlap with the known vocabulary).
         Forcing onboarding survey.
INFO:    Onboarding survey populated profile with:
         ['quantum macrame', 'astral welding', 'vibe coding', 'automation', 'python', 'cloud', 'computing', 'learning']

#1  Machine Learning Engineer Role  [JobRole]   -> similarity = 0.4962
#2  MLOps Engineer Role            [JobRole]   -> similarity = 0.4626
#3  Google Cloud Professional      [Platform]  -> similarity = 0.3849
```

> ⚠️ **Important Note:** Similarity scores reflect vocabulary overlap with the *sample* dataset shipped in this repo — they are a measure of relative ranking, not an absolute "compatibility %". Swapping `raw_skills.csv` for a larger, denser real‑world dataset will produce more nuanced scores without changing a single line of code.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mahaveer-mkj/DecodeLabs-Internship.git
cd DecodeLabs-Internship/Project-3-Recommendation-Engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run the CLI pipeline (the graded deliverable)
```bash
python main.py
```

The script will:
- Print a dataset summary to the console (items loaded, TF‑IDF vocabulary size)
- Run a standard weighted match and print the Top‑3 results
- Run both cold‑start scenarios (structural + semantic) and print the recovery logs and Top‑3 results
- The full text capture is also saved at `outputs/sample_run_output.txt`

### 3b. Or launch the interactive UI (optional, same logic underneath)
```bash
streamlit run app.py
```

`app.py` is a presentation layer only — it imports `TechStackRecommender` directly and calls the exact same `.recommend()` used by `main.py`. No recommendation logic is duplicated or reimplemented for the UI. It adds: a skill picker seeded with real in‑vocabulary suggestions, live 1–5 preference sliders (the bonus weighting feature, made tangible), and a visible "❄️ Cold‑start guard activated" banner whenever that safety net actually fires — so the same mechanism described below isn't just tested in a log line, it's something you can trigger and watch happen.

---

## 📁 Project Structure

```
Project-3-Recommendation-Engine/
│
├── main.py                      # CLI entry point — demo harness, all 3 scenarios
├── app.py                       # Optional Streamlit UI — same pipeline, presentation only
├── .streamlit/
│   └── config.toml              # Theme (reuses this repo's indigo brand color)
├── pipeline.py                  # TechStackRecommender — orchestrates the strict IPO call
├── data_ingestion.py            # Stage 1 — Input: load dataset, validate, cold-start survey
├── feature_extractor.py         # Stage 2a — Process: shared TF-IDF vector space
├── recommendation_engine.py     # Stage 2b/3/4 — Process/Output: cosine scoring, sort, Top-N
├── models.py                    # UserProfile / MatchResult data contracts
├── requirements.txt             # Python dependencies (incl. streamlit, optional)
├── raw_skills.csv                # Item dataset (Platforms / Tools / Job Roles)
├── outputs/                      # Captured run output
│   └── sample_run_output.txt
└── README.md                     # This file
```

---

## 🏗️ Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT STAGE                               │
│                                                                │
│  capture_user_profile()  →  validate (>= 3 valid skills?)    │
│                                      ↓                        │
│                    [FAIL] → cold_start_survey()               │
│                    [PASS] → UserProfile(skills, weights)      │
└─────────────────────┬──────────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────────┐
│                   PROCESS STAGE                                │
│                                                                  │
│  FeatureExtractor.fit(items)     →  shared TF-IDF vocabulary    │
│                              ↓                                   │
│  transform_user(skills, weights) →  weighted pseudo-document    │
│                              ↓                                   │
│        is_zero_vector()?  → [YES] → cold_start_survey()         │
│                              ↓ [NO]                              │
│   RecommendationEngine.score() →  cosine_similarity(user, items)│
└─────────────────────┬──────────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────────┐
│                   OUTPUT STAGE                                  │
│                                                                  │
│  rank()  →  sort_values(by=score, descending)                   │
│         →  get_top_n(n=3)  →  List[MatchResult]                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configurable Parameters

Nothing is hardcoded. All key values are exposed as parameters on `TechStackRecommender.recommend()`:

```python
engine.recommend(
    skills   = ["Python", "Cloud Computing", "Automation"],          # min. 3 required
    weights  = {"Python": 5, "Cloud Computing": 4, "Automation": 3}, # optional, 1-5 scale
    top_n    = 3,                                                    # how many matches to return
)
```

`MIN_REQUIRED_SKILLS` (in `data_ingestion.py`) and `TOP_N` (in `pipeline.py`) are also exposed as module‑level constants for easy tuning.

To scale this pipeline to a larger dataset, simply point `TechStackRecommender(dataset_path=...)` at a bigger `raw_skills.csv` — same 4‑column schema, zero other code changes required.

---

## 🧠 Key Concepts Explained

### Why Content‑Based Filtering, not Collaborative Filtering?
Collaborative filtering needs *other users'* interaction history ("people who picked AWS also picked Kubernetes"). DecodeLabs' brief is a single user with a cold profile and no peer data to lean on — content‑based filtering only needs the item's own attributes, so it works correctly from the very first user, day one.

### Why TF‑IDF instead of raw word counts or Binary Overlap?
A binary 1/0 "does this item mention Python" flag treats "python" identically whether it's the item's defining skill or a passing mention — and treats every term as equally important. TF‑IDF fixes both:

```
TF(term, doc)  =  count of term in doc
IDF(term)      =  log( N_items / N_items_containing_term )
TF-IDF score   =  TF × IDF
```

A term like "automation" that appears in most items gets a near‑zero IDF — it can't dominate a match. A term like "pytorch" appearing in only one item gets a high IDF — it's treated as a strong, specific signal.

### Why Cosine Similarity over Euclidean Distance?
KNN (Project 2) uses Euclidean distance because all four Iris features sit on comparable numeric scales. TF‑IDF vectors are nothing like that: high‑dimensional, extremely sparse, and variable in magnitude purely based on how many terms a description happens to use. Euclidean distance is sensitive to that magnitude — a richly‑weighted user profile would look artificially "far" from a short item description. Cosine similarity ignores magnitude entirely and measures only the angle between two vectors:

```
cosine(A, B)  =  (A · B) / (||A|| × ||B||)
```

Two vectors pointing in the same direction score 1.0 regardless of length — exactly the scale‑invariance sparse text vectors need.

### Why two separate Cold‑Start guards?
> **Critical rule:** a user profile can fail in two different ways, and only one of them is visible before vectorization.

- **Structural** (`data_ingestion.py`): fewer than `MIN_REQUIRED_SKILLS = 3` valid strings were supplied. Caught immediately, before any math runs.
- **Semantic** (`pipeline.py`): 3+ skills were supplied, but none exist anywhere in the fitted vocabulary — the resulting TF‑IDF vector has `nnz == 0`. This only becomes visible *after* vectorization, which is why it's checked in the orchestrator rather than at ingestion.

Both paths fall back to the same `cold_start_survey()`, which pulls the most frequent terms straight out of the item dataset itself — so the recovered profile is always grounded in real, scorable vocabulary, never a guess.

### Why weight preferences instead of just listing skills?
Weighting is implemented as repeated term frequency, not a separate formula bolted onto cosine similarity afterward. Rating "Python" 5/5 repeats that token 5× in the user's pseudo‑document before vectorization, which legitimately increases its TF component — honoring the actual TF‑IDF math instead of faking a binary flag.

### What changed to support the UI, and what didn't
Adding `app.py` required exactly two additive changes to the tested backend, both pure introspection — zero existing logic was modified:
- `DataIngestion.get_suggested_skills()` — a public wrapper around the already‑existing trending‑term extractor, so the UI can show real, in‑vocabulary skill suggestions instead of guessing.
- `TechStackRecommender.last_profile` — records which `UserProfile` produced the most recent `.recommend()` call, so the UI can detect when a cold‑start guard fired and explain it to the user. `.recommend()`'s signature and return value are unchanged; `main.py` and every existing call site behave identically (verified by re‑running the original demo before and after — byte‑identical output).

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | 1.3+ | `TfidfVectorizer`, `cosine_similarity` |
| `pandas` | 2.0+ | Item dataset handling, ranking and sorting |
| `scipy` | 1.10+ | Sparse vector representation (`csr_matrix`) |
| `streamlit` | 1.38+ | Optional interactive UI (presentation layer only) |

---

## 📚 Project Context

This project was assigned and completed during my AI internship at [DecodeLabs](https://www.decodelabs.tech/). Project 3 builds directly on Project 2: where Project 2 learned to predict a known *label* from training data (Supervised Learning), Project 3 has no labels at all — it matches an unlabelled user profile directly against item attributes using vector‑space geometry (Content‑Based Filtering).

**Dataset:** `raw_skills.csv` — an 18‑row sample catalogue of Platforms, Tools, and Job Roles (schema: `item_id, item_name, category, description`). Swap in a real DecodeLabs dataset with the same schema and zero code changes are required.

---

## 👤 Author

**Mahaveer Mundaluhari**  
*Artificial Intelligence Intern @ [DecodeLabs](https://www.decodelabs.tech/)*  
*B.S. Data Science & Applications — IIT Madras*  
*B.Tech CSE (AI & ML) — OUTR Bhubaneswar*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mahaveer%20Mundaluhari-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/mahaveer-mundaluhari/)
[![GitHub](https://img.shields.io/badge/GitHub-mahaveer--mkj-181717?style=flat&logo=github)](https://github.com/mahaveer-mkj)
[![Email](https://img.shields.io/badge/Email-mahaveer%40maxiwoxi.com-EA4335?style=flat&logo=gmail)](mailto:mahaveer@maxiwoxi.com)

---

<div align="center">

*Built with precision. Documented with purpose.*  
**If this helped you, consider giving it a ⭐**

</div>
