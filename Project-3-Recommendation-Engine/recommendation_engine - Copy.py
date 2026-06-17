"""
recommendation_engine.py
--------------------------
STAGES 2b, 3, and 4 of the IPO Pipeline: PROCESS (Scoring) -> Sort -> Filter.

Pure math, zero I/O. Takes a user vector and an item matrix (both already
living in the same TF-IDF feature space) and returns a ranked, truncated
list of MatchResult objects.

Why Cosine Similarity, and explicitly not Euclidean Distance:
    TF-IDF vectors are high-dimensional and extremely sparse (most items
    only touch a handful of the full vocabulary). Euclidean distance is
    sensitive to vector *magnitude* -- a user profile with many weighted
    skill mentions would look "far" from a short item description purely
    because of length, even if both point in an identical semantic
    direction. Cosine similarity normalizes that away and measures only
    the *angle* between vectors -- how aligned the pattern of skills is,
    regardless of how verbose either side's text happens to be. That's
    also why it scales: it stays well-behaved as vocabulary size grows,
    where Euclidean distance degrades in high-dimensional sparse spaces.
"""

import logging
from typing import List

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from models import MatchResult

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Scores, ranks, and truncates item matches for a given user vector."""

    def __init__(self, item_df: pd.DataFrame, item_matrix: csr_matrix) -> None:
        self.item_df = item_df
        self.item_matrix = item_matrix

    def score(self, user_vector: csr_matrix) -> List[float]:
        """
        STAGE 2b -- Scoring.
        Computes cosine similarity between the user vector (1 x N_vocab)
        and every row of the item matrix (N_items x N_vocab) in a single
        vectorized call.

        Returns:
            A flat list of similarity scores, one per item, in the same
            row order as item_df.
        """
        similarities = cosine_similarity(user_vector, self.item_matrix)
        return similarities.flatten().tolist()

    def rank(self, scores: List[float]) -> pd.DataFrame:
        """
        STAGE 3 -- Sorting.
        Attaches scores to their items and sorts strictly descending
        (best match first).
        """
        ranked = self.item_df.copy()
        ranked["similarity_score"] = scores
        return ranked.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

    def get_top_n(self, ranked_df: pd.DataFrame, n: int = 3) -> List[MatchResult]:
        """
        STAGE 4 -- Filtering (Top-N).
        Truncates to the top `n` rows to avoid choice overload, per spec.
        The full ranked_df is intentionally never returned to the caller
        of the public pipeline -- only this truncated slice is.
        """
        top_slice = ranked_df.head(n)
        return [
            MatchResult(
                item_id=int(row.item_id),
                item_name=row.item_name,
                category=row.category,
                description=row.description,
                similarity_score=round(float(row.similarity_score), 4),
            )
            for row in top_slice.itertuples(index=False)
        ]
