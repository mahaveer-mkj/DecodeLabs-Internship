"""
main.py
--------
Execution entry point / demo harness for the Digital Matchmaker.

Run directly:
    python main.py
"""

from typing import List

from models import MatchResult
from pipeline import TechStackRecommender


def print_results(title: str, results: List[MatchResult]) -> None:
    print(f"\n=== {title} ===")
    if not results:
        print("No matches found.")
        return
    for rank, match in enumerate(results, start=1):
        print(
            f"#{rank}  {match.item_name}  [{match.category}]  "
            f"-> similarity = {match.similarity_score}"
        )


def main() -> None:
    engine = TechStackRecommender("raw_skills.csv")
    engine.load()

    # --- Scenario 1: Standard run, minimum 3 valid inputs + weighting ---
    skills = ["Python", "Cloud Computing", "Automation"]
    weights = {"Python": 5, "Cloud Computing": 4, "Automation": 3}  # Bonus feature
    results = engine.recommend(skills, weights)
    print_results("Standard Match (weighted)", results)

    # --- Scenario 2: Structural cold start (fewer than 3 skills) ---
    results_cold = engine.recommend(["Python"])
    print_results("Cold Start (insufficient input)", results_cold)

    # --- Scenario 3: Semantic cold start (count OK, but all OOV terms) ---
    results_oov = engine.recommend(["Quantum Macrame", "Astral Welding", "Vibe Coding"])
    print_results("Cold Start (out-of-vocabulary input)", results_oov)


if __name__ == "__main__":
    main()
