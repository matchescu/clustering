from typing import Any

from matchescu.entity_matchers.similarity._common import Similarity


class ExactMatch(Similarity):
    def _compute_similarity(self, a: Any, b: Any) -> float:
        return 1 if a == b else 0
