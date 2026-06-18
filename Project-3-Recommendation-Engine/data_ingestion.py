"""
data_ingestion.py
------------------
STAGE 1 of the IPO Pipeline: INPUT.

Responsible for:
  1. Loading the item-side dataset (raw_skills.csv).
  2. Capturing and validating the user-side state (skills + optional
     1-5 preference weights).
  3. Enforcing the "User Cold Start" rule: a user is NEVER allowed to
     enter the Feature Extraction stage with fewer than
     MIN_REQUIRED_SKILLS, and is never left with a final vector of
     zeros (handled jointly with pipeline.py -- see cold_start_survey).

This module deliberately knows nothing about TF-IDF or cosine similarity.
Its only job is to guarantee that whatever leaves it is a clean,
validated UserProfile.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from models import UserProfile

logger = logging.getLogger(__name__)

MIN_REQUIRED_SKILLS = 3  # Hard constraint per project spec
MIN_WEIGHT, MAX_WEIGHT = 1, 5
DEFAULT_WEIGHT = 3


class DataIngestion:
    """Handles all read access to the item dataset and user input validation."""

    REQUIRED_COLUMNS = {"item_id", "item_name", "category", "description"}

    def __init__(self, dataset_path: str = "raw_skills.csv") -> None:
        self.dataset_path = Path(dataset_path)
        self._item_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Item-side ingestion
    # ------------------------------------------------------------------
    def load_item_dataset(self) -> pd.DataFrame:
        """
        Loads raw_skills.csv and validates its schema.

        Returns:
            A DataFrame with one row per platform/tool/job-role item.

        Raises:
            FileNotFoundError: if raw_skills.csv is missing.
            ValueError: if required columns are absent.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'. Expected a "
                f"raw_skills.csv with columns: {sorted(self.REQUIRED_COLUMNS)}"
            )

        df = pd.read_csv(self.dataset_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"raw_skills.csv is missing required columns: {missing}")

        # Normalize text early so every downstream stage sees consistent casing.
        df["description"] = df["description"].str.lower().str.strip()
        self._item_df = df.reset_index(drop=True)
        logger.info("Loaded %d items from %s", len(df), self.dataset_path)
        return self._item_df

    # ------------------------------------------------------------------
    # User-side ingestion
    # ------------------------------------------------------------------
    def capture_user_profile(
        self,
        skills: List[str],
        weights: Optional[Dict[str, int]] = None,
    ) -> UserProfile:
        """
        Validates and packages raw user input into a UserProfile.

        Cold-Start Rule #1 (Structural): if fewer than MIN_REQUIRED_SKILLS
        valid, non-empty skill strings are supplied, the onboarding survey
        is triggered automatically -- the pipeline never proceeds with a
        sparse, under-specified profile.

        Args:
            skills: e.g. ["Python", "Cloud Computing", "Automation"]
            weights: optional {"Python": 5, "Cloud Computing": 4}, 1-5 scale.

        Returns:
            A validated UserProfile (never empty, never under-filled).
        """
        cleaned = [s.strip().lower() for s in skills if s and s.strip()]
        clean_weights = self._sanitize_weights(weights or {}, cleaned)

        if len(cleaned) < MIN_REQUIRED_SKILLS:
            logger.warning(
                "Cold Start triggered: only %d valid skill(s) supplied "
                "(minimum required = %d). Falling back to onboarding survey.",
                len(cleaned), MIN_REQUIRED_SKILLS,
            )
            return self.cold_start_survey(seed_skills=cleaned)

        return UserProfile(skills=cleaned, weights=clean_weights, is_cold_start=False)

    def _sanitize_weights(self, weights: Dict[str, int], skills: List[str]) -> Dict[str, int]:
        """Clamps weights to [1, 5] and fills missing skills with a neutral default."""
        clean: Dict[str, int] = {}
        for skill in skills:
            raw = weights.get(skill, weights.get(skill.title(), DEFAULT_WEIGHT))
            try:
                w = int(raw)
            except (TypeError, ValueError):
                w = DEFAULT_WEIGHT
            clean[skill] = min(max(w, MIN_WEIGHT), MAX_WEIGHT)
        return clean

    # ------------------------------------------------------------------
    # Cold-Start survey bypass
    # ------------------------------------------------------------------
    def cold_start_survey(self, seed_skills: Optional[List[str]] = None) -> UserProfile:
        """
        Cold-Start Rule #2 (Forced Ingestion): guarantees a non-zero,
        densely-populated UserProfile by pulling the most common,
        highest-signal terms straight out of the item dataset's own
        vocabulary.

        This simulates an "onboarding survey": in a real product this
        would render multi-select chips of these suggested skills; here
        we auto-select them so the pipeline can never receive a profile
        that vectorizes to all zeros.

        Args:
            seed_skills: any partial input the user already gave (kept, not discarded).

        Returns:
            A UserProfile padded with dataset-derived trending skills,
            flagged with is_cold_start=True.
        """
        if self._item_df is None:
            self.load_item_dataset()

        trending = self._extract_trending_terms(top_k=5)
        seed_skills = seed_skills or []
        merged = list(dict.fromkeys(seed_skills + trending))  # de-dup, preserve order

        # Survey-suggested skills get a neutral default weight since the
        # user never explicitly rated them.
        weights = {s: DEFAULT_WEIGHT for s in merged}

        logger.info("Onboarding survey populated profile with: %s", merged)
        return UserProfile(skills=merged, weights=weights, is_cold_start=True)

    def get_suggested_skills(self, top_k: int = 8) -> List[str]:
        """
        Public-facing version of the trending-term extractor, intended for
        UI layers (e.g. a Streamlit front-end) that want to show a user
        real, in-vocabulary skill suggestions before they type anything.
        Pure read-only convenience wrapper -- no change to ingestion logic.
        """
        if self._item_df is None:
            self.load_item_dataset()
        return self._extract_trending_terms(top_k=top_k)

    def _extract_trending_terms(self, top_k: int = 5) -> List[str]:
        """
        Naive but effective trending-term extractor: counts raw token
        frequency across every item description and returns the top_k
        most common terms (length > 2, to skip noise like "ui").

        Intentionally simple (no TF-IDF here) -- Ingestion must stay
        independent of the Feature Extraction stage's math.
        """
        tokens = " ".join(self._item_df["description"]).split()
        tokens = [t for t in tokens if len(t) > 2]
        counts = Counter(tokens)
        return [term for term, _ in counts.most_common(top_k)]
