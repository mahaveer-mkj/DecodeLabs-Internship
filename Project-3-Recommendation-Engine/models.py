"""
models.py
----------
Defines the core data contracts (schemas) used across the Digital
Matchmaker pipeline. Centralizing these as typed dataclasses keeps
Ingestion, Feature Extraction, and the Recommendation Engine strictly
decoupled -- each stage only needs to know the *shape* of the data,
not how it was produced.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UserProfile:
    """
    Represents the fully validated, ingested state of a single user.

    Attributes:
        skills: Cleaned, lowercase skill/interest tokens.
        weights: Optional 1-5 preference rating per skill (Bonus Feature).
                 Skills with no explicit rating default to a neutral 3.
        is_cold_start: Set by DataIngestion when the onboarding survey
                       bypass had to be triggered (structural or semantic
                       cold start). Downstream code can use this flag to,
                       e.g., show the user a "based on trending skills"
                       disclaimer in a real UI.
    """
    skills: List[str]
    weights: Dict[str, int] = field(default_factory=dict)
    is_cold_start: bool = False


@dataclass
class MatchResult:
    """A single scored, ranked recommendation returned by the engine."""
    item_id: int
    item_name: str
    category: str
    description: str
    similarity_score: float
