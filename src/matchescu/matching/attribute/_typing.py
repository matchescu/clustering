from typing import Protocol, Any

from matchescu.matching.attribute._match_result import MatchResult


class AttrMatchCallable(Protocol):
    def __call__(self, a: Any, b: Any) -> MatchResult:
        pass
