"""
pipeline.py
------------
The TechStackRecommender ties Stages 1-4 together into a single, strict
IPO (Input -> Process -> Output) call: recommend().

This is the only class a consumer (CLI, API route, notebook, web backend,
etc.) should ever need to import directly -- DataIngestion,
FeatureExtractor, and RecommendationEngine are internal implementation
details of this orchestrator.
"""

import logging
from typing import Dict, List, Optional

from data_ingestion import DataIngestion
from feature_extractor import FeatureExtractor
from models import MatchResult, UserProfile
from recommendation_engine import RecommendationEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TOP_N = 3  # Hard constraint per project spec: never surface more than this.


class TechStackRecommender:
    """
    The Digital Matchmaker.

    Usage:
        engine = TechStackRecommender("raw_skills.csv")
        engine.load()
        results = engine.recommend(["Python", "Cloud Computing", "Automation"])
    """

    def __init__(self, dataset_path: str = "raw_skills.csv") -> None:
        self.ingestion = DataIngestion(dataset_path)
        self.extractor = FeatureExtractor()
        self._engine: Optional[RecommendationEngine] = None
        # Read-only introspection hook for UI layers: which UserProfile
        # actually produced the most recent recommend() call. Does not
        # change recommend()'s signature or return type.
        self.last_profile: Optional[UserProfile] = None

    def load(self) -> None:
        """One-time setup: load items and fit the shared TF-IDF vocabulary."""
        item_df = self.ingestion.load_item_dataset()
        item_matrix = self.extractor.fit(item_df)
        self._engine = RecommendationEngine(item_df, item_matrix)

    def recommend(
        self,
        skills: List[str],
        weights: Optional[Dict[str, int]] = None,
        top_n: int = TOP_N,
    ) -> List[MatchResult]:
        """
        Runs the full strict IPO pipeline for one user and returns the
        Top-N matches.

            INPUT   -> DataIngestion.capture_user_profile (+ cold-start guard)
            PROCESS -> FeatureExtractor.transform_user -> RecommendationEngine.score
            OUTPUT  -> RecommendationEngine.rank -> get_top_n

        Args:
            skills: minimum 3 raw skill/interest strings.
            weights: optional {skill: 1-5} preference ratings (Bonus Feature).
            top_n: defaults to 3 per spec; exposed for testing/flexibility only.

        Returns:
            A list of at most `top_n` MatchResult objects, descending by score.
        """
        if self._engine is None:
            raise RuntimeError("Call .load() before .recommend().")

        # ---------------- INPUT ----------------
        profile = self.ingestion.capture_user_profile(skills, weights)

        # ---------------- PROCESS (2a: Vectorize) ----------------
        user_vector = self.extractor.transform_user(profile.skills, profile.weights)

        # Cold-Start Rule #2: catch the zero-vector case even when the user
        # technically supplied >= 3 skills, but none of them exist anywhere
        # in the item vocabulary (fully out-of-vocabulary input). This is
        # the literal "never leave a user with a vector of zeros" guarantee.
        if self.extractor.is_zero_vector(user_vector):
            logger.warning(
                "Zero-vector detected post-vectorization (skills %s have no "
                "overlap with the known vocabulary). Forcing onboarding survey.",
                profile.skills,
            )
            profile = self.ingestion.cold_start_survey(seed_skills=profile.skills)
            user_vector = self.extractor.transform_user(profile.skills, profile.weights)

        # ---------------- PROCESS (2b: Score) ----------------
        scores = self._engine.score(user_vector)

        # ---------------- OUTPUT (3: Sort, 4: Top-N Filter) ----------------
        ranked_df = self._engine.rank(scores)
        top_matches = self._engine.get_top_n(ranked_df, n=top_n)

        if profile.is_cold_start:
            logger.info(
                "Note: these results were generated from an onboarding-survey "
                "profile, not raw user input."
            )

        self.last_profile = profile  # UI introspection hook; doesn't affect return value
        return top_matches
