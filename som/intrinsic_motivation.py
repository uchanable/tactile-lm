"""Intrinsic motivation combining curiosity and affective touch.

Provides an intrinsic reward signal that combines:
1. Prediction error (curiosity): novelty of the current sensory input,
   measured as distance from the SOM's BMU to the input vector.
2. CT affective signal: mean CT afferent firing rate, reflecting
   the pleasantness of current touch.

This models the infant's drive to explore (curiosity) while being
drawn to affective touch experiences (CT-mediated pleasant contact).

Reference:
    Pathak et al. (2017). Curiosity-driven exploration by
    self-supervised prediction. ICML.
"""

import numpy as np
from typing import Optional
from som.core import SelfOrganizingMap


class IntrinsicMotivation:
    """Intrinsic reward from SOM novelty and CT activation.

    reward = alpha * novelty + beta * ct_signal

    Novelty is the BMU distance (how poorly the SOM represents the
    current input), normalized by a running estimate. CT signal is
    the mean CT firing rate across body parts.

    Attributes:
        alpha: Weight for curiosity (novelty) component.
        beta: Weight for CT affective component.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        novelty_window: int = 1000,
        ct_window: int = 1000,
    ):
        """Initialize intrinsic motivation module.

        Args:
            alpha: Curiosity weight. Higher = more exploration-driven.
            beta: CT reward weight. Higher = more drawn to pleasant touch.
            novelty_window: Window size for novelty running statistics.
            ct_window: Window size for CT running statistics.
        """
        self.alpha = alpha
        self.beta = beta

        # Running statistics for novelty normalization
        self._novelty_history = np.zeros(novelty_window)
        self._novelty_idx = 0
        self._novelty_count = 0
        self._novelty_window = novelty_window

        # Running statistics for CT normalization
        self._ct_history = np.zeros(ct_window)
        self._ct_idx = 0
        self._ct_count = 0
        self._ct_window = ct_window

    def compute_novelty(self, som: SelfOrganizingMap, x: np.ndarray) -> float:
        """Compute novelty as distance from input to its BMU.

        High distance = the SOM hasn't learned this pattern well = novel.

        Args:
            som: The SOM to measure novelty against.
            x: Input vector.

        Returns:
            float: Raw novelty score (Euclidean distance to BMU).
        """
        bmu = som.find_bmu(x)
        distance = np.sqrt(np.sum((x - som.weights[bmu]) ** 2))
        return float(distance)

    def compute_reward(
        self,
        som: SelfOrganizingMap,
        x: np.ndarray,
        ct_activation: float,
    ) -> dict:
        """Compute intrinsic reward.

        Args:
            som: SOM for novelty computation (typically tactile_disc).
            x: Input vector to the SOM.
            ct_activation: Mean CT firing rate (from preprocessor).

        Returns:
            dict with keys:
                "reward": Combined intrinsic reward.
                "novelty_raw": Raw novelty (BMU distance).
                "novelty_normalized": Normalized novelty [0, ~3].
                "ct_raw": Raw CT activation.
                "ct_normalized": Normalized CT activation.
        """
        # Novelty
        novelty_raw = self.compute_novelty(som, x)
        self._update_novelty(novelty_raw)
        novelty_norm = self._normalize_novelty(novelty_raw)

        # CT signal
        self._update_ct(ct_activation)
        ct_norm = self._normalize_ct(ct_activation)

        # Combined reward
        reward = self.alpha * novelty_norm + self.beta * ct_norm

        return {
            "reward": float(reward),
            "novelty_raw": float(novelty_raw),
            "novelty_normalized": float(novelty_norm),
            "ct_raw": float(ct_activation),
            "ct_normalized": float(ct_norm),
        }

    def _update_novelty(self, value: float):
        """Update novelty running statistics."""
        self._novelty_history[self._novelty_idx % self._novelty_window] = value
        self._novelty_idx += 1
        self._novelty_count = min(self._novelty_count + 1, self._novelty_window)

    def _normalize_novelty(self, value: float) -> float:
        """Normalize novelty using running mean and std."""
        if self._novelty_count < 2:
            return value
        data = self._novelty_history[:self._novelty_count]
        mean = data.mean()
        std = data.std() + 1e-8
        return (value - mean) / std

    def _update_ct(self, value: float):
        """Update CT running statistics."""
        self._ct_history[self._ct_idx % self._ct_window] = value
        self._ct_idx += 1
        self._ct_count = min(self._ct_count + 1, self._ct_window)

    def _normalize_ct(self, value: float) -> float:
        """Normalize CT using running mean and std."""
        if self._ct_count < 2:
            return value
        data = self._ct_history[:self._ct_count]
        mean = data.mean()
        std = data.std() + 1e-8
        return (value - mean) / std

    def get_state(self) -> dict:
        """Serialize state."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "novelty_history": self._novelty_history.copy(),
            "novelty_idx": self._novelty_idx,
            "novelty_count": self._novelty_count,
            "ct_history": self._ct_history.copy(),
            "ct_idx": self._ct_idx,
            "ct_count": self._ct_count,
        }

    def set_state(self, state: dict):
        """Restore state."""
        self._novelty_history = state["novelty_history"].copy()
        self._novelty_idx = state["novelty_idx"]
        self._novelty_count = state["novelty_count"]
        self._ct_history = state["ct_history"].copy()
        self._ct_idx = state["ct_idx"]
        self._ct_count = state["ct_count"]
