"""Strategy name registry shared by the CLI, output directories, and config.json."""

from __future__ import annotations

from .badge import BADGEGreedyDPP, BADGEKMeansPlusPlus, BADGEMCMCDPP
from .base import QueryStrategy
from .disagreement import Disagreement
from .entropy import Entropy
from .farthest_traversal import FarthestTraversal
from .mfft import MFFT
from .random_sampling import RandomSampling

_STRATEGY_CLASSES: dict[str, type[QueryStrategy]] = {
    "random": RandomSampling,
    "entropy": Entropy,
    "farthest-traversal": FarthestTraversal,
    "disagreement": Disagreement,
    "mfft": MFFT,
    "badge-kmeans++": BADGEKMeansPlusPlus,
    "badge-mcmc-dpp": BADGEMCMCDPP,
    "badge-greedy-dpp": BADGEGreedyDPP,
}

STRATEGY_NAMES = tuple(_STRATEGY_CLASSES)

# The first round draws the shared cold-start set, so every strategy starts from the same labeled pool.
COLD_START_STRATEGY = "random"

# The full-supervised reference labels the whole pool in that single cold-start round.
FULL_SUPERVISED = "full-supervised"


def build_query_strategy(name: str, *, random_state) -> QueryStrategy:
    """Build a query strategy by name.

    Args:
        name: Strategy name in STRATEGY_NAMES.
        random_state: Seed or numpy RandomState shared across the rounds of one experiment.

    Returns:
        A new instance of the strategy class registered under name.
    """
    return _STRATEGY_CLASSES[name](random_state=random_state)
