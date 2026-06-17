"""
feature_extractor.py
---------------------
STAGE 2a of the IPO Pipeline: PROCESS (Vectorization).

Converts raw text (item descriptions + user skill profile) into a shared
TF-IDF vector space. This is the single most important file for
mathematical correctness: every later similarity score is only valid if
the user vector and the item matrix live in *exactly* the same feature
space -- the "strict vocabulary alignment" constraint from the spec.

Design rule: fit the vectorizer ONCE on the item corpus. Every later
.transform() call (including the user's) reuses that exact fitted
vocabulary. We never call fit() a second time, and we never fit a
separate vectorizer on the user's text alone.
"""

import logging
from typing import Dict, List

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Wraps a single, shared TfidfVectorizer for items and users alike."""

    def __init__(self) -> None:
        # stop_words='english' strips generic noise ("the", "and", "with")
        # so the IDF component can do its real job: penalizing terms that
        # are common ACROSS items (e.g. "automation" shows up almost
        # everywhere) while rewarding terms that are rare and specific
        # (e.g. "kubernetes", "pytorch", "selenium").
        self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        self.item_matrix: csr_matrix = None  # populated by fit()
        self._is_fitted = False

    def fit(self, item_df) -> csr_matrix:
        """
        Fits the TF-IDF vocabulary on the item corpus and returns the
        resulting item-feature matrix (N_items x N_vocab).

        Called exactly once per dataset load.
        """
        corpus = item_df["description"].tolist()
        self.item_matrix = self.vectorizer.fit_transform(corpus)
        self._is_fitted = True
        logger.info(
            "TF-IDF vocabulary fitted: %d terms across %d items.",
            len(self.vectorizer.vocabulary_), self.item_matrix.shape[0],
        )
        return self.item_matrix

    def transform_user(self, skills: List[str], weights: Dict[str, int]) -> csr_matrix:
        """
        Projects a user's skill profile into the SAME vector space as the
        item matrix, honoring per-skill preference weights.

        Bonus Feature -- Weighted Term Frequency (not binary overlap):
            Each skill token is repeated `weight` times in the user's
            pseudo-document before vectorization. TF-IDF's TF component
            is a term *count* -- repeating "python" 5x vs 1x measurably
            increases its contribution to the resulting vector, while the
            IDF component (how rare "python" is across the whole item
            corpus) stays fixed, since IDF is derived only from the
            already-fitted item corpus. This gives an honest, math-native
            way to encode "I really care about this skill" without ever
            faking the underlying TF-IDF formula or resorting to a 1/0 flag.

        Args:
            skills: validated, lowercase skill tokens.
            weights: 1-5 preference rating per skill.

        Returns:
            A 1 x N_vocab sparse vector living in the item feature space.

        Raises:
            RuntimeError: if called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor.fit() must be called before transform_user().")

        weighted_tokens: List[str] = []
        for skill in skills:
            weight = weights.get(skill, 3)
            weighted_tokens.extend([skill] * weight)

        pseudo_document = " ".join(weighted_tokens)

        # CRITICAL: .transform(), never .fit_transform(). This is what
        # guarantees vocabulary alignment with the item matrix.
        return self.vectorizer.transform([pseudo_document])

    @staticmethod
    def is_zero_vector(vector: csr_matrix) -> bool:
        """
        Detects the exact failure mode the spec calls out: a user vector
        with no overlap whatsoever with the fitted vocabulary (every term
        the user typed was Out-Of-Vocabulary). nnz == 0 means every
        dimension is literally zero.
        """
        return vector.nnz == 0
