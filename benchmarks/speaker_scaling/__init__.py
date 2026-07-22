"""Synthetic scalability benchmark for the existing speaker clusterer."""

from .runner import (
    DEFAULT_MODES,
    DEFAULT_SAMPLES_PER_SPEAKER,
    DEFAULT_SPEAKER_COUNTS,
    partition_is_correct,
    run_benchmark,
    run_case,
)
from .synthetic import SCENARIO_NAMES, SyntheticCase, generate_scenario

__all__ = [
    "DEFAULT_MODES",
    "DEFAULT_SAMPLES_PER_SPEAKER",
    "DEFAULT_SPEAKER_COUNTS",
    "SCENARIO_NAMES",
    "SyntheticCase",
    "generate_scenario",
    "partition_is_correct",
    "run_benchmark",
    "run_case",
]
