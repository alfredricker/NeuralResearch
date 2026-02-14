from .base import GlobalOutputMap, LocalOutputMap, RegionOutputAssignment
from .oglobal import ClassificationVoteGlobalOutputMap
from .local import FlatLocalOutputMap

__all__ = [
    "RegionOutputAssignment",
    "LocalOutputMap",
    "GlobalOutputMap",
    "FlatLocalOutputMap",
    "ClassificationVoteGlobalOutputMap",
]
