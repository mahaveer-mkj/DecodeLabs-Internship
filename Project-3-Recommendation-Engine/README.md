
<div align="center">

# 🎯 Tech Stack Recommender — "The Digital Matchmaker"
### DecodeLabs — Project 3: AI Recommendation Logic

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)

**A production-grade Content-Based Filtering engine —  
mathematically mapping skills to careers using TF‑IDF vectorization and Cosine Similarity.  
No rules, no randomness — every recommendation is the output of strict vector-space math.**

[View Code](#-project-structure) · [How to Run](#-quick-start) · [Sample Output](#-results--outputs) · [Key Concepts](#-key-concepts-explained)

---

</div>

## 📌 What This Project Does

This pipeline recommends the **Top‑3 most aligned tech platforms, tools, and job roles** based on a user's self‑reported skills and career interests. It implements a strict **Content‑Based Filtering** approach — matching a user profile directly against item attributes using **TF‑IDF weighting** and **Cosine Similarity**, following a rigorous **Input → Process → Output (IPO)** architecture.

| Stage | Task | Key Decision |
|-------|------|-------------|
| **Input** | Ingest user skills (min. 3), apply cold‑start guard, run weighted preference survey | Never allow a zero vector — structural *and* semantic guards |
| **Process** | Fit shared TF‑IDF vocabulary on the item catalogue, project user into the *same* vector space, score via Cosine Similarity | Vocabulary alignment enforced — user and items share one fitted vectorizer |
| **Output** | Sort descending by similarity score, truncate to Top‑3 | Prevents choice overload — the full ranked list is never shown |

> **Design Philosophy:** This is not a keyword matcher. Generic terms like *"automation"* are penalized by TF‑IDF's inverse document frequency. Rare, discriminative terms like *"Kubernetes"* or *"PyTorch"* are rewarded. The result is a precision‑oriented recommendation, not a fuzzy guess.

---

## 📊 Results & Outputs

The pipeline prints results directly to the console and logs cold‑start trigger events. A captured run is saved at [`outputs/sample_run_output.txt`](./outputs/sample_run_output.txt).

### Standard Match (Weighted Skills)
> User provides 3+ skills with optional 1‑5 importance ratings. TF‑IDF amplifies higher‑rated skills by repeating them in the synthetic user document — an honest manipulation of term frequency, not a post‑hoc score multiplier.

```
=== Standard Match (weighted) ===
#1  Machine Learning Engineer Role  [JobRole]  -> similarity = 0.406
#2  Cloud DevOps Engineer Role  [JobRole]  -> similarity = 0.4039
#3  MLOps Engineer Role  [JobRole]  -> similarity = 0.3785
```

---

### Cold‑Start Trigger: Insufficient Input
> Fewer than 3 skills supplied. The system **refuses** to proceed — this is the structural cold‑start guard. The user is looped until minimum input is met.

```
[LOG] Cold-start guard triggered: only 2 skills provided (minimum 3 required).
[LOG] Re-prompting user for additional skills...
```

---

### Cold‑Start Trigger: Semantic Zero‑Vector
> 3+ skills provided, but **none** overlap with the fitted TF‑IDF vocabulary. The resulting sparse vector has `nnz == 0` — the semantic guard catches it before scoring can produce meaningless zeros.

```
[LOG] Cold-start guard triggered: user vector has zero non-zero entries.
[LOG] All provided skills are out-of-vocabulary. Prompting for new input.
```

> ⚠️ **Important Note:** Cosine Similarity scores are bounded [0, 1] because TF‑IDF vectors are non‑negative. A score of 1.0 indicates perfect pattern alignment; 0.0 indicates orthogonal (no overlap). Real‑world scores typically range from 0.2 to 0.6 — perfect 1.0 is theoretically possible but rare outside toy datasets.

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

### 3. Run the pipeline
```bash
python main.py
```

The script will:
- Load and validate `raw_skills.csv` (falls back to a built‑in sample dataset if missing)
- Launch the interactive onboarding survey (skills + optional 1‑5 importance ratings)
- Log every pipeline stage to the console
- Print the **Top‑3 ranked matches** with similarity scores
- Trigger and report both cold‑start guard types when applicable

---

## 📁 Project Structure

```
Project-3-Recommendation-Engine/
│
├── main.py                       # Entry point — orchestrates the full IPO pipeline
├── requirements.txt              # Python dependencies
├── raw_skills.csv                # Item catalogue (Platforms, Tools, Job Roles)
│
├── src/
│   ├── __init__.py               # Package marker
│   ├── data_ingestion.py         # Stage 1 — loads CSV, validates 3-skill minimum, runs cold‑start survey
│   ├── feature_extractor.py      # Stage 2a — fits shared TF‑IDF vector space, projects user into it
│   ├── recommendation_engine.py  # Stage 2b/3/4 — cosine scoring, descending sort, Top‑N truncation
│   └── pipeline.py               # TechStackRecommender — orchestrator enforcing strict IPO call order
│
└── outputs/
    └── sample_run_output.txt     # Captured console output from an actual run
```

---

## 🏗️ Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT STAGE                              │
│                                                             │
│  User skills (min. 3)  →  Cold‑start guard #1 (structural) │
│       ↓                                                    │
│  Weighted preference survey (1–5 per skill)                 │
│       ↓                                                    │
│  Build synthetic user document (skill × weight repetitions) │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   PROCESS STAGE                             │
│                                                             │
│  TF‑IDF vectorizer (pre‑fitted on item catalogue)           │
│       ↓                                                    │
│  Transform user document → user_vector (same vocabulary)    │
│       ↓                                                    │
│  Cold‑start guard #2 (semantic) — check nnz > 0             │
│       ↓                                                    │
│  Cosine Similarity(user_vector, item_matrix) → score[i]     │
│       ↓                                                    │
│  Sort descending → ranked_items                             │
│       ↓                                                    │
│  Truncate to Top‑N (default N=3)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   OUTPUT STAGE                              │
│                                                             │
│  Ranked MatchResults  →  #1 Role/Platform/Tool + score      │
│                       →  #2 Role/Platform/Tool + score      │
│                       →  #3 Role/Platform/Tool + score      │
│                                                             │
│  Full ranked list is NEVER displayed (prevents overload)    │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configurable Parameters

Nothing is hardcoded. All key values are exposed as parameters in `TechStackRecommender`:

```python
recommender = TechStackRecommender(
    data_path     = "raw_skills.csv",   # Path to item catalogue
    top_n         = 3,                  # Number of recommendations to return
    min_skills    = 3,                  # Minimum skills required (cold‑start guard)
    enable_ratings = True,              # Toggle 1‑5 importance weighting survey
)

# TF‑IDF vectorizer parameters (inside FeatureExtractor)
vectorizer = TfidfVectorizer(
    lowercase     = True,
    stop_words    = "english",          # Remove generic filler words
    ngram_range   = (1, 2),             # Capture phrases like "cloud computing"
    max_features  = 500,                # Cap vocabulary size for large datasets
    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#./-]*\b",  # Preserves "C++", "UI/UX"
)
```

To scale this pipeline to a larger dataset, simply replace `raw_skills.csv` with any CSV containing `Job_Role` and `Required_Skills` columns — the rest of the pipeline requires **zero changes**.

---

## 🧠 Key Concepts Explained

### Why Content‑Based Filtering instead of Collaborative Filtering?
Collaborative Filtering requires **other users' behaviour data** (ratings, clicks, purchase history) to find "users like you." In a cold‑start scenario with a single new user, that data simply doesn't exist. Content‑Based Filtering works with **zero other users** — it matches the user's attributes directly against item attributes, making it the only viable choice for this problem.

---

### Why TF‑IDF instead of raw term counts?
Consider two skills: *"Python"* and *"Kubernetes"*. If *"Python"* appears in 90% of all job listings, a raw count gives it high weight in every recommendation — making all results look the same. TF‑IDF solves this:

```
TF‑IDF(term, document) = TF(term, document) × IDF(term)

where:
  TF  = (occurrences of term in document) / (total terms in document)
  IDF = log( (total documents) / (documents containing term) )
```

- **High IDF:** *"Kubernetes"* appears in few documents → rewarded as discriminative
- **Low IDF:** *"Python"* appears nearly everywhere → penalized as generic

The result: rare, specific skills drive the recommendation; common skills become tie‑breakers.

---

### Why Cosine Similarity instead of Euclidean Distance?
Euclidean distance measures **magnitude** — the straight‑line distance between two points. In high‑dimensional text vector spaces (500+ features), Euclidean distance breaks down catastrophically:

- **Sparse vectors:** Most entries are zero. Euclidean distance treats all zeros as "close," even when the non‑zero terms are completely different.
- **Length sensitivity:** A job listing with 10 required skills will always be "farther" from a user with 3 skills than a listing with 3 skills — even if the user is a perfect match for the 10‑skill role.

Cosine Similarity measures the **angle** between vectors, ignoring magnitude entirely:

```
Cosine(A, B) = (A · B) / (||A|| × ||B||)

  = 1.0  →  vectors point in the same direction (perfect match)
  = 0.0  →  vectors are orthogonal (no overlap)
```

> **Why it matters here:** A user who lists *"Python, Docker, Kubernetes"* and a job role requiring *"Python, Docker, Kubernetes, Terraform, CI/CD, Helm, AWS, Prometheus, Grafana, Istio"* will still score high on Cosine Similarity — because the user's three skills point in the *same direction* as the role's skill vector. Euclidean distance would incorrectly rank this match far lower.

---

### Why Strict Vocabulary Alignment?
If the user's skills are vectorized with a **separate** TF‑IDF instance (naively calling `fit_transform()` on user input alone), the resulting vector lives in a **different feature space** than the item matrix. The dimensions won't align, and Cosine Similarity becomes mathematically meaningless.

**The fix:** Fit the vectorizer **once** on the item catalogue. When the user arrives, call only `transform()` — projecting the user into the same vocabulary space. This guarantees every dimension means the same thing for both vectors.

> **Critical rule:** `fit()` on items only. `transform()` on the user. Never `fit_transform()` the user. Breaking this rule silently corrupts the similarity scores.

---

### The Two Cold‑Start Guards — Structural and Semantic

Most systems only check "did the user provide input?" This pipeline adds a second, deeper guard:

| Guard | Type | Condition | What It Catches |
|-------|------|-----------|-----------------|
| **#1** | Structural | `len(skills) < 3` | Empty or insufficient input — user hasn't engaged |
| **#2** | Semantic | `user_vector.nnz == 0` | User provided 3+ skills, but **none** exist in the vocabulary — e.g., `["esperanto", "underwater basket weaving", "interpretive dance"]` |

> Guard #2 is subtle but critical: without it, Cosine Similarity returns `0.0` for every item, producing a "Top‑3" that is mathematically valid but practically useless. The semantic guard detects this and re‑prompts the user — ensuring every recommendation has genuine signal.

---

### Bonus: Weighted Preference Scoring (1–5 Scale)
Most content‑based recommenders treat all user skills equally. This pipeline introduces a **pre‑similarity bias** through honest TF manipulation:

- User rates *"Python"* as **5** (critical) and *"Docker"* as **2** (nice‑to‑have)
- The synthetic user document becomes: `"python python python python python docker docker"`
- TF‑IDF naturally assigns *"Python"* a higher term frequency — no post‑hoc score multiplier needed

This keeps the math transparent: the vectorizer does exactly what it was designed to do, and the bias is fully auditable in the user document string.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | 1.3+ | TfidfVectorizer, cosine_similarity |
| `numpy` | 1.24+ | Numerical array operations, argsort |
| `pandas` | 2.0+ | Data loading, DataFrame manipulation, result formatting |
| `logging` | stdlib | Pipeline‑stage logging and cold‑start trigger events |

---

## 📚 Project Context

This project was assigned and completed during my AI internship at [DecodeLabs](https://www.decodelabs.tech/). Project 3 specifically demonstrates the transition from classification (Project 2's KNN) to **Recommendation Systems** — where the algorithm must rank items by relevance rather than assign discrete labels. It implements the same strict **IPO discipline** as Project 2, ensuring every recommendation is traceable to a mathematical operation, never a heuristic rule or random fallback.

**Methodology:** Content‑Based Filtering — matching user profiles directly to item attributes without requiring other users' behavioural data.

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

*Every recommendation is the output of vector‑space math — not a guess.*  
**If this helped you, consider giving it a ⭐**

</div>



