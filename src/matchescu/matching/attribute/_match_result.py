from enum import Enum


class MatchResult(Enum):
    def __repr__(self):
        return f"{self.value} - {self.name.lower()}"

    def __str__(self):
        return str(self.value)


class TernaryResult(MatchResult):
    """Enumerates possible match results inspired by the Fellegi-Sunter model."""

    NonMatch = 0
    Match = 1
    NoComparisonData = 2


class BinaryResult(MatchResult):
    """Enumerates values that can be used with a Naive Bayes classifier."""

    Negative = -1
    Positive = 1
